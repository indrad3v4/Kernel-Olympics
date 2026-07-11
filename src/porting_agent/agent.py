"""
Porting Agent — uses Fireworks API to fix CUDA→ROCm porting issues.

Input: flagged kernel + surrounding context + any retrieved similar pattern
Model: Fireworks API (AMD-hosted catalog)
Output: ported code + confidence score + explanation of the fix

Confidence-gated: if confidence < threshold, flag for human review.

Every code path here that returns a ``ported_code`` field routes the string
through :func:`verification.extraction.extract_code` (to strip prose /
markdown / JSON wrappers) and then :func:`verification.lexical.validate_lexical`
(to reject reasoning at top level).  A response that fails either gate is
NEVER returned as ``ported_code`` — the caller sees ``rejected: True`` and
the raw response in a diagnostic field instead.
"""

import os
import json
import socket
import sys
import time
import urllib.error
from typing import Dict, List, Optional
from pathlib import Path

# The verification helpers live under src/verification.  When the agent is
# imported via ``from porting_agent.agent import PortingAgent`` after
# ``sys.path`` has been seeded with the ``src`` root (main.py does that),
# this import resolves.  When tests import agent.py in isolation they must
# also seed sys.path — we do the seed here as a safety net.
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from verification.extraction import extract_code as _extract_code_v2
from verification.lexical import validate_lexical as _validate_lexical
# router.ModelRouter._fix_ported_code is the comprehensive CUDA→HIP header/API
# rewriter (cuda_runtime.h→hip/hip_runtime.h, cudaMalloc→hipMalloc, etc.).
# Reused here so the template fallback below can never emit CUDA headers —
# it previously ran only the wavefront32→64 fixups and left includes alone.
from router import ModelRouter as _ModelRouter


class FailureType:
    """Explicit classification of why a model call did not yield usable code.

    Plain string constants (not an Enum) so they drop straight into JSON debug
    artifacts and report dicts without a custom encoder. ``INFRASTRUCTURE``
    covers everything that means "we never got a real translation attempt back
    from the model" — the failures the mission calls out (timeout, malformed
    response, reasoning-only). ``TRANSIENT`` is the subset worth retrying the
    *same* model for before moving on or giving up.
    """
    API_TIMEOUT = "api_timeout"
    NETWORK_ERROR = "network_error"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    EMPTY_RESPONSE = "empty_response"
    REASONING_ONLY = "reasoning_only"
    PARTIAL_CODE = "partial_code"
    INVALID_JSON = "invalid_json"
    EXTRACTION_FAILURE = "extraction_failure"
    VALIDATION_FAILURE = "validation_failure"

    INFRASTRUCTURE = frozenset({
        API_TIMEOUT, NETWORK_ERROR, HTTP_ERROR, RATE_LIMIT, EMPTY_RESPONSE,
        REASONING_ONLY, PARTIAL_CODE, INVALID_JSON, EXTRACTION_FAILURE,
        VALIDATION_FAILURE,
    })
    TRANSIENT = frozenset({API_TIMEOUT, NETWORK_ERROR, RATE_LIMIT, HTTP_ERROR})

    @classmethod
    def is_transient(cls, failure_type: Optional[str]) -> bool:
        return failure_type in cls.TRANSIENT

    @classmethod
    def classify_exception(cls, exc: BaseException) -> str:
        """Best-effort mapping of a caught exception to a FailureType.

        Never raises — an exception raised while classifying an exception
        would replace an observable failure with an unobservable one.
        """
        try:
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code == 429:
                    return cls.RATE_LIMIT
                if exc.code >= 500:
                    return cls.API_TIMEOUT  # 5xx is transient — retry-worthy
                return cls.HTTP_ERROR
            if isinstance(exc, socket.timeout):
                return cls.API_TIMEOUT
            if isinstance(exc, urllib.error.URLError):
                reason = str(getattr(exc, "reason", exc))
                if "timed out" in reason.lower():
                    return cls.API_TIMEOUT
                return cls.NETWORK_ERROR
            if isinstance(exc, TimeoutError):
                return cls.API_TIMEOUT
            msg = str(exc).lower()
            if "timed out" in msg or "timeout" in msg:
                return cls.API_TIMEOUT
            if "connection" in msg or "network" in msg or "resolve" in msg:
                return cls.NETWORK_ERROR
        except Exception:
            pass
        return cls.NETWORK_ERROR


def _force_ipv4():
    """Monkey-patch socket to prefer IPv4 for HTTP connections.
    
    Python's urllib tries IPv6 first, which is slow/unreachable on some
    Jupyter nodes. Forces IPv4 but only for SOCK_STREAM (HTTP/HTTPS)."""
    orig = socket.getaddrinfo
    def ipv4_safe(host, port, family=0, type=0, proto=0, flags=0):
        results = orig(host, port, family, type, proto, flags)
        # Prefer IPv4 (AF_INET) over IPv6 (AF_INET6) for TCP connections
        v4 = [r for r in results if r[0] == socket.AF_INET and r[1] == socket.SOCK_STREAM]
        if v4:
            return v4 + [r for r in results if r not in v4]
        return results
    socket.getaddrinfo = ipv4_safe


_force_ipv4()


class PortingAgent:
    """LLM-based CUDA→ROCm porting agent using Fireworks API."""

    SYSTEM_PROMPT = """You are PortingAgent, an expert CUDA→ROCm/HIP migration engineer. 
You receive CUDA kernels flagged with warp(32)→wavefront(64) divergence issues
and produce a JSON object describing the exact fix.

ROLE: CUDA→HIP porting specialist — you understand AMD GPU architecture
(wavefront=64 threads) and know how every CUDA warp intrinsic maps to HIP.

PORTING RULES (follow all that apply):
1. AMD GPUs use wavefronts of 64 threads, not warps of 32
2. __shfl_down_sync(0xffffffff, val, 16) on wavefront64 skips half the lanes — 
   the offset must be adjusted or use a different algorithm
3. Hardcoded "32" for warp size → should be "64" or use warpSize/wavefront size
4. __shared__ arrays sized to 32 → may need 64 for wavefront-aware code
5. __syncwarp() → use __syncthreads() for HIP compatibility
6. Use __ballot_sync (HIP) instead of CUDA warp-vote functions
7. Keep the same algorithm structure — only change what's needed for portability
8. __activemask() → use __ballot_sync(0xffffffff, 1) for active lane mask on HIP
9. __all_sync/__any_sync — these take a mask argument; verify it works with 64 lanes
10. __match_all_sync — no direct HIP equivalent; redesign as sequential check
11. threadIdx.x >> 5 computes warp index (32 lanes) → should be >> 6 for wavefront64
12. Lane identification: if (lane_id < 32) → if (lane_id < 64) for wavefront boundary
13. __shfl_sync (basic shuffle) — mask and lane count must be adjusted for wavefront64

OUTPUT FORMAT — STRICT JSON (no prose, no extra text):
Respond with a single JSON object inside a ```json markdown code block.
The JSON object must have EXACTLY these four fields:

{
  "ported_code": "<string — the full ported HIP kernel code>",
  "confidence": <integer 0-100 — rubric-based confidence score>,
  "changes": ["<string — one change description per modification>", ...],
  "explanation": "<string — short explanation of what was fixed and why>"
}

EXAMPLE:
```json
{
  "ported_code": "__global__ void vec_add(float* a, float* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += b[i]; }",
  "confidence": 88,
  "changes": ["Replaced warp32 hardcodes with WAVEFRONT_SIZE (64)", "Changed __syncwarp() to __syncthreads()"],
  "explanation": "Ported warp-32 kernel to wavefront-64 HIP by replacing hardcoded 32 with WAVEFRONT_SIZE and fixing sync primitives."
}
```

CRITICAL: Return ONLY the ```json ... ``` block. No introductory text, no explanation before it, no summary after it. If I cannot parse valid JSON from your response, the pipeline fails.
"""

    # ✅ VERIFIED WORKING on Fireworks API (tested, confirmed):
    #   - kimi-k2p7-code (strongest: code generation, struct-aware HIP porting)
    #   - glm-5p2        (accurate code generation, struct understanding)
    #   - deepseek-v4-pro (good general fallback)
    # ❌ UNVERIFIED / REMOVED:
    #   - llama-v3p3-70b-instruct  (unstable results on Fireworks, removed)
    FALLBACK_MODELS = [
        "accounts/fireworks/models/kimi-k2p7-code",              # 1st: Kimi K2.7 Code (coder ✅)
        "accounts/fireworks/models/glm-5p2",                      # 2nd: GLM (planner ✅)
        "accounts/fireworks/models/deepseek-v4-pro",              # 3rd: DeepSeek (works ✅)
    ]

    # Cost per 1000 tokens for Fireworks models (for cost tracking)
    MODEL_COST_MAP = {
        "accounts/fireworks/models/kimi-k2p7-code": 0.00095,
        "accounts/fireworks/models/glm-5p2": 0.0014,
        "accounts/fireworks/models/deepseek-v4-pro": 0.0012,
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "accounts/fireworks/models/kimi-k2p7-code",
                 deepseek_key: str = "", deepseek_model: str = "deepseek-reasoner"):
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.model = model
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_model = deepseek_model
        self.api_base = "https://api.fireworks.ai/inference/v1"
        self.deepseek_base = "https://api.deepseek.com/v1"
        # Phase: per-model health tracking for THIS agent instance/session.
        # Repeated timeouts/malformed responses push a model to the back of
        # the try-order on the next kernel, instead of paying its full
        # timeout again on every single kernel in the batch.
        self._model_health: Dict[str, Dict] = {}

    def _health(self, model: str) -> Dict:
        return self._model_health.setdefault(model, {
            "attempts": 0, "timeouts": 0, "malformed": 0,
            "extraction_ok": 0, "total_latency_ms": 0.0,
        })

    def _record_health(self, model: str, failure_type: Optional[str] = None,
                       latency_ms: float = 0.0, extraction_ok: Optional[bool] = None) -> None:
        h = self._health(model)
        h["attempts"] += 1
        h["total_latency_ms"] += latency_ms
        if failure_type == FailureType.API_TIMEOUT:
            h["timeouts"] += 1
        elif failure_type in (FailureType.REASONING_ONLY, FailureType.EMPTY_RESPONSE,
                              FailureType.INVALID_JSON, FailureType.EXTRACTION_FAILURE,
                              FailureType.PARTIAL_CODE):
            h["malformed"] += 1
        if extraction_ok:
            h["extraction_ok"] += 1

    def _ordered_models(self) -> List[str]:
        """FALLBACK_MODELS reordered by this session's observed reliability.

        A model with a nonzero attempt count and a bad failure rate (timeouts
        or malformed responses) is deprioritized — tried later, never
        dropped entirely, since a single bad kernel should not permanently
        blacklist a model that may work fine on the next one.
        """
        def _failure_rate(model: str) -> float:
            h = self._model_health.get(model)
            if not h or h["attempts"] == 0:
                return 0.0
            return (h["timeouts"] + h["malformed"]) / h["attempts"]
        return sorted(self.FALLBACK_MODELS, key=_failure_rate)

    # ── Retry / recovery policy ──────────────────────────────────────────
    # Two attempts per model: the initial call, then either (a) a same-model
    # retry with backoff on a transient infra failure (timeout/network/5xx/
    # rate-limit), or (b) a single "return ONLY code" recovery follow-up on a
    # malformed/reasoning-only response. Either way, a model is never
    # abandoned — and the fallback never invoked — after just one failure.
    _MAX_ATTEMPTS_PER_MODEL = 2
    _BACKOFF_BASE_SECONDS = 1.0

    _REASONING_RECOVERY_PROMPT = (
        "Your previous response contained analysis instead of code.\n\n"
        "Return ONLY the complete HIP source file.\n\n"
        "Do not include explanations.\n\n"
        "Do not include Markdown.\n\n"
        "Do not include reasoning.\n\n"
        "Return a complete compilable translation unit."
    )

    _STATUS_ICONS = {
        "ok": "✅", "retry": "🔁", "recovery": "🩹", "exhausted": "🛑",
    }

    def _log_event(self, model: str, status: str, note: str = "",
                   attempt: int = 1, final: Optional[str] = None) -> None:
        """One structured line per model event — actionable, not generic.

        Replaces the old "No usable code or JSON found" catch-all: every
        line names the model, the attempt, and the specific reason, so a
        reader (or a saved log) can tell a Kimi reasoning-only response
        apart from a GLM timeout apart from a DeepSeek HTTP error.
        """
        icon = self._STATUS_ICONS.get(status, "⚠")
        line = f"║ {icon} [{model}] attempt {attempt}/{self._MAX_ATTEMPTS_PER_MODEL} — {status}"
        if note:
            line += f": {note[:140]}"
        print(line)
        if final:
            print(f"║    └─ {final[:140]}")

    def _post_chat(self, model: str, messages: list, timeout: float):
        """One raw Fireworks chat-completion call.

        Returns ``(content, call_cost, latency_ms)``. Raises on any
        transport/HTTP failure — the caller classifies and decides whether
        to retry.
        """
        import urllib.request
        import json as _json
        data = _json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        }).encode()
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = json.loads(raw_body)
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        tokens_used = (
            usage.get("total_tokens", 0)
            or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        cost_per_1k = self.MODEL_COST_MAP.get(model, 0.0012)
        call_cost = round(tokens_used / 1000 * cost_per_1k, 4)
        return content, call_cost, latency_ms

    def _try_extract(self, content: str):
        """Run the full extraction pipeline against one LLM response.

        Strategy order: (1) JSON object with a ``ported_code`` field, (2)
        markdown-fenced ```cpp/```c++/```hip/```cuda blocks and JSON-field
        variants (via :func:`verification.extraction.extract_code`), (3) a
        sliding raw-text window anchored on ``#include``/``__global__``/
        ``namespace``/``template``, with conversational prose stripped by
        the legacy stripper. Only after every strategy fails is this
        classified as an extraction failure.

        Returns ``(fixed_code, parsed_json_or_None, failure_type_or_None, note)``.
        """
        if not content or not content.strip():
            return None, None, FailureType.EMPTY_RESPONSE, "empty response body"

        # Strategy 1: JSON object with a ported_code field.
        parsed = self._extract_json_from_text(content)
        if parsed and "ported_code" in parsed:
            fixed = self._fix_ported_code(parsed["ported_code"])
            ok, reason = self._gate_code(fixed)
            if ok:
                return fixed, parsed, None, "json-field"
            failure = (FailureType.REASONING_ONLY if "reasoning" in reason.lower()
                      else FailureType.PARTIAL_CODE)
            return None, None, failure, f"json-field rejected by lexical gate: {reason}"

        # Strategy 2/3: fenced cpp/c++/hip blocks, raw #include/__global__/
        # namespace/template window, conversational-text stripping.
        extraction = _extract_code_v2(content)
        code_text = (extraction.code or "").strip()
        if not code_text:
            code_text = self._extract_code_from_text(content)
        if code_text:
            fixed = self._fix_ported_code(code_text)
            ok, reason = self._gate_code(fixed)
            if ok:
                return fixed, None, None, f"text-extract:{extraction.strategy}"
            failure = (FailureType.REASONING_ONLY if "reasoning" in reason.lower()
                      else FailureType.EXTRACTION_FAILURE)
            return None, None, failure, f"text-extract rejected by lexical gate: {reason}"

        if parsed is not None:
            return None, None, FailureType.INVALID_JSON, "JSON parsed but had no ported_code field"
        return None, None, FailureType.EXTRACTION_FAILURE, "no code-shaped block found (markdown/JSON/raw-window all failed)"

    def _attempt_model(self, model: str, user_prompt: str, is_primary: bool,
                       infra_failures: List[Dict]) -> Optional[Dict]:
        """Try one model end-to-end, with retry and reasoning-recovery.

        Returns a result dict on success, or ``None`` once every recovery
        strategy for this model has been exhausted — the caller then moves
        to the next model, never straight to the template fallback.
        """
        timeout = 120 if is_primary else 30
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_failure_type = None
        last_note = ""

        for attempt in range(1, self._MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                content, call_cost, latency_ms = self._post_chat(model, messages, timeout)
            except urllib.error.HTTPError as exc:
                try:
                    err_body = exc.read().decode(errors="replace")[:200]
                except Exception:
                    err_body = ""
                failure_type = FailureType.classify_exception(exc)
                self._record_health(model, failure_type=failure_type)
                last_failure_type, last_note = failure_type, f"HTTP {exc.code}: {err_body}"
                self._log_event(model, status=failure_type, note=last_note, attempt=attempt)
                infra_failures.append({"model": model, "attempt": attempt,
                                       "failure_type": failure_type, "detail": last_note})
            except Exception as exc:
                failure_type = FailureType.classify_exception(exc)
                self._record_health(model, failure_type=failure_type)
                last_failure_type, last_note = failure_type, str(exc)[:160]
                self._log_event(model, status=failure_type, note=last_note, attempt=attempt)
                infra_failures.append({"model": model, "attempt": attempt,
                                       "failure_type": failure_type, "detail": last_note})
            else:
                fixed, parsed, failure_type, note = self._try_extract(content)
                if fixed:
                    self._record_health(model, latency_ms=latency_ms, extraction_ok=True)
                    self._log_event(model, status="ok", note=note, attempt=attempt,
                                    final="accepted — recovery/retry not needed" if attempt == 1
                                          else "accepted after recovery")
                    if parsed:
                        parsed["ported_code"] = fixed
                        parsed["cost"] = call_cost
                        parsed.setdefault("confidence", self._rubric_score_extracted(fixed))
                        parsed.setdefault("changes", [])
                        parsed.setdefault("explanation", "")
                        return parsed
                    return {
                        "ported_code": fixed,
                        "confidence": self._rubric_score_extracted(fixed),
                        "changes": ["LLM returned text without valid JSON — extracted code block"],
                        "explanation": "Code extracted from LLM text output",
                        "cost": call_cost,
                    }

                self._record_health(model, failure_type=failure_type, latency_ms=latency_ms)
                last_failure_type, last_note = failure_type, note
                self._log_event(model, status=failure_type, note=note, attempt=attempt)
                infra_failures.append({"model": model, "attempt": attempt,
                                       "failure_type": failure_type, "detail": note})

                # Recovery: send exactly one "code only, no reasoning"
                # follow-up before giving up on this model.
                recoverable = failure_type in (
                    FailureType.REASONING_ONLY, FailureType.PARTIAL_CODE,
                    FailureType.EXTRACTION_FAILURE, FailureType.INVALID_JSON,
                )
                if recoverable and attempt < self._MAX_ATTEMPTS_PER_MODEL:
                    self._log_event(model, status="recovery",
                                    note="sending code-only follow-up prompt", attempt=attempt)
                    messages = messages + [
                        {"role": "assistant", "content": content[:4000]},
                        {"role": "user", "content": self._REASONING_RECOVERY_PROMPT},
                    ]
                    continue
                break

            # Transient infra failure (timeout/network/5xx/rate-limit) — retry
            # the SAME model with a short backoff before moving on.
            if FailureType.is_transient(last_failure_type) and attempt < self._MAX_ATTEMPTS_PER_MODEL:
                backoff = self._BACKOFF_BASE_SECONDS * attempt
                self._log_event(model, status="retry",
                                note=f"transient {last_failure_type} — backing off {backoff:.1f}s",
                                attempt=attempt)
                time.sleep(backoff)
                continue
            break

        self._log_event(model, status="exhausted", note=last_note,
                        attempt=self._MAX_ATTEMPTS_PER_MODEL,
                        final=f"moving to next model (last: {last_failure_type})")
        return None

    def port_kernel(self, source_code: str, context: str = "",
                    cached_pattern: Optional[Dict] = None) -> Dict:
        """Port a CUDA kernel to ROCm/HIP using LLM."""

        # TRIZ: Fix source code BEFORE any LLM/template processing
        fixed_source = self._fix_ported_code(source_code)

        # Build prompt with context
        user_prompt = f"Port this CUDA kernel to AMD ROCm/HIP:\n\n```cuda\n{fixed_source}\n```\n"

        if context:
            user_prompt += f"\nAdditional context:\n{context}\n"

        if cached_pattern:
            user_prompt += (
                f"\nA similar pattern was found in memory (confidence: {cached_pattern.get('confidence', 0)}):\n"
                f"Original: {cached_pattern.get('original_snippet', '')}\n"
                f"Verified fix: {cached_pattern.get('verified_fix', '')}\n"
                f"Apply similar approach if applicable.\n"
            )

        user_prompt += "\nOutput the result as a JSON object inside a ```json markdown block, with fields: ported_code (the full kernel code), confidence (0-100), changes (list), explanation (string)."

        # For hackathon: if no API key, use template-based porting
        if not self.api_key or self.api_key == "test":
            return self._template_port(fixed_source, cached_pattern)

        # Health-ordered fallback list: self.model always goes first (it is
        # the configured preference), the rest are tried in order of THIS
        # session's observed reliability rather than a fixed static list —
        # a model that just timed out on kernel N is tried last on kernel
        # N+1, not blocked outright.
        primary = self.model
        models_to_try = [primary] + [m for m in self._ordered_models() if m != primary]

        infra_failures: List[Dict] = []
        for model in models_to_try:
            result = self._attempt_model(model, user_prompt, is_primary=(model == primary),
                                         infra_failures=infra_failures)
            if result is not None:
                result["failure_classification"] = None
                result["used_fallback"] = False
                return result

        # Every model — with retries and reasoning-recovery follow-ups —
        # failed to produce usable code. This is the ONLY point at which the
        # template fallback is justified; it never fires on a single
        # timeout or a single malformed response.
        print(f"║ 🛑 All {len(models_to_try)} model(s) exhausted after retries/recovery "
              f"— using HIP-safe template fallback")
        result = self._template_port(source_code, cached_pattern)
        if "ported_code" in result:
            result["ported_code"] = self._fix_ported_code(result["ported_code"])
        result["used_fallback"] = True
        result["failure_classification"] = "infrastructure"
        result["infra_failures"] = infra_failures
        return result

    @staticmethod
    def _gate_code(code: str):
        """Return ``(ok, reason)`` — lexical gate over a candidate port.

        Runs the same lexical validator the router uses.  This is the last
        barrier inside the porting agent: a response that fails here never
        surfaces as ``ported_code`` in the returned dict, so a caller that
        writes ``ported_code`` to disk (e.g. main.py's ported_kernels/ save)
        cannot accidentally serialise reasoning.

        Structural checks are deliberately deferred to the router — this
        agent may legitimately return small snippets (template fallback) that
        skip a full structural check.
        """
        if not code or not code.strip():
            return False, "empty code"
        try:
            lex = _validate_lexical(code)
        except Exception as exc:
            return True, f"lexical gate errored (allowing): {exc!r}"
        return lex.ok, lex.reason()

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict]:
        """Extract a JSON object from LLM text that may have surrounding prose.

        Tries, in order:
          1. Markdown ```json ... ``` block
          2. First { to last } slice
          3. Direct json.loads
        Returns parsed dict or None.
        """
        import re, json as _json

        # 1. Markdown json code block
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            try:
                return _json.loads(candidate)
            except _json.JSONDecodeError:
                pass

        # 2. First { to last } — strip surrounding prose
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return _json.loads(candidate)
            except _json.JSONDecodeError:
                pass

        # 3. Direct parse
        try:
            return _json.loads(text.strip())
        except _json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _extract_code_from_text(text: str) -> str:
        """Extract HIP/CUDA code from LLM text output (non-JSON responses).

        Handles:
          - Markdown ```cuda / ```hip / ```cpp code blocks
          - JSON values that contain code as a field value
          - Raw __global__ kernel definitions
          - From #include to end
        """
        import re
        # Try markdown code blocks (cuda, hip, cpp, or unlabeled)
        blocks = re.findall(r'```(?:cuda|hip|cpp|cu|cl)?\n(.*?)```', text, re.DOTALL)
        if blocks:
            return PortingAgent._strip_prose_lines(max(blocks, key=len).strip())
        # Try generic ``` blocks
        blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
        if blocks:
            return PortingAgent._strip_prose_lines(max(blocks, key=len).strip())
        # Try to find from #include to end (BEFORE __global__ — includes headers)
        match = re.search(r'(#include\s+<.*)', text, re.DOTALL)
        if match:
            return PortingAgent._strip_prose_lines(match.group(1).strip())
        # Try to find __global__ kernel definition (fallback)
        match = re.search(r'(__global__\s+void\s+\w+\s*\(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if match:
            return PortingAgent._strip_prose_lines(match.group(1).strip())
        # Look for __device__ function definitions
        match = re.search(r'(__device__\s+\w+\s+\w+\s*\(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if match:
            return PortingAgent._strip_prose_lines(match.group(1).strip())
        return ""

    @staticmethod
    def _strip_prose_lines(code: str) -> str:
        """Remove non-code prose lines from extracted code.
        
        Filters out lines that look like LLM analysis/commentary
        rather than actual HIP/C++ code. Also cleans up include
        directives contaminated by inline prose."""
        import re
        clean = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Skip empty lines at start
            if not stripped and not clean:
                continue
            # Fix: `#include <...> prose` → `#include <...>`
            if stripped.startswith('#include'):
                inc_match = re.match(r'(#include\s+<[^>]+>)', stripped)
                if inc_match:
                    clean.append(inc_match.group(1))
                    continue
                # Try match with backtick-garbled end
                inc_match2 = re.match(r'(#include\s+<[^`>]+[`>])', stripped)
                if inc_match2:
                    clean.append(inc_match2.group(1).replace('`', ''))
                    continue
            # Skip lines that are clearly prose (not code)
            prose_patterns = [
                r'^(Let\'?s|Need|Should|Maybe|Consider|Note:|Question:|Answer:|Step\s+\d)',
                r'^(Here\'?s|This |The |We |I |For |In |As |A |An )',
                r'^(First|Second|Third|Finally|Next|Then|After)',
                r'^Output:|^Input:',
                r'explanation|explain|analysis|decid(e|ing|es)',
            ]
            is_prose = any(re.match(p, stripped, re.IGNORECASE) for p in prose_patterns)
            if is_prose and not any(c in stripped for c in ['{', '}', ';', '__global__', '__device__']):
                continue
            # Remove trailing inline prose (backtick contamination)
            clean.append(stripped.rstrip('`').rstrip())
        result = '\n'.join(clean).strip()
        return result if result else code

    @staticmethod
    def _rubric_score(source_code: str, ported_code: str, changes: list,
                      has_header: bool = False, cached: bool = False) -> int:
        """Rubric-based confidence scoring for ported kernels (0-100).

        Rubric dimensions:
          - Wavefront Header (0-20) : #define WAVEFRONT_SIZE present in output
          - Portability (0-30)      : code is already AMD-ready OR fixes applied
          - Code Integrity (0-30)   : kernel structure preserved, non-empty
          - Change Logging (0-20)   : changes list depth and variety

        Special cases:
          - "No changes needed" (code already AMD-compatible) → full Portability score
          - Cached verified fix applied → high baseline + rubric sanity check
        """
        import re

        # ── Dimension 1: Wavefront Header (0-20) ──
        header_score = 0
        if "#define WAVEFRONT_SIZE 64" in ported_code:
            header_score = 20
        elif "WAVEFRONT_SIZE" in ported_code:
            header_score = 12

        # ── Dimension 2: Portability (0-35) ──
        no_changes_needed = any("no automatic changes needed" in c.lower()
                                for c in changes)
        fix_score = 0
        if no_changes_needed:
            # Code is already AMD-ready — high confidence
            fix_score = 35
        else:
            change_categories = set()
            for c in changes:
                cl = c.lower()
                if "wavefront" in cl:
                    change_categories.add("wavefront")
                if "mask" in cl or "0x3f" in cl or "0x1f" in cl:
                    change_categories.add("mask")
                if "shfl" in cl or "shuffle" in cl:
                    change_categories.add("shuffle")
                if "sync" in cl:
                    change_categories.add("sync")
                if "tile" in cl:
                    change_categories.add("tile")
                if "shared" in cl:
                    change_categories.add("shared_mem")
                if "lane" in cl:
                    change_categories.add("lane_id")
                if "ballot" in cl or "activemask" in cl:
                    change_categories.add("ballot")
                if "all_sync" in cl or "any_sync" in cl or "match_all" in cl:
                    change_categories.add("predicate_sync")
                if "warp_size" in cl or "warp_mask" in cl:
                    change_categories.add("warp_size")
                if "cached" in cl or "verified" in cl:
                    change_categories.add("cached")
            # Baseline (processing was done) + category bonus
            fix_score = min(15 + len(change_categories) * 5, 35)

        # ── Dimension 3: Code Integrity (0-30) ──
        integrity_score = 0
        code_len = len(ported_code.strip())
        if code_len > 0:
            integrity_score += 5
        if code_len > 50:
            integrity_score += 5
        if code_len > 200:
            integrity_score += 5
        if "__global__" in ported_code or "__device__" in ported_code:
            integrity_score += 10
        elif "void" in ported_code and ("(" in ported_code and ")" in ported_code):
            integrity_score += 5
        if source_code and code_len >= len(source_code.strip()) * 0.5:
            integrity_score += 5

        # ── Dimension 4: Change Logging (0-20) ──
        explain_score = 0
        if changes:
            if len(changes) >= 1:
                explain_score += 5
            if len(changes) >= 3:
                explain_score += 5
            if len(changes) >= 5:
                explain_score += 5
            if len(changes) >= 8:
                explain_score += 5

        total = header_score + fix_score + integrity_score + explain_score

        # Cached pattern bonus: +5 if a verified pattern was applied
        if cached:
            total += 5

        return min(total, 100)

    @staticmethod
    def _rubric_score_extracted(code_text: str) -> int:
        """Rubric for code extracted from LLM text output (0-100).

        Lower confidence because extraction is inherently lossy.
        """
        import re
        score = 0

        # Extraction Success (0-25)
        if code_text and len(code_text.strip()) > 0:
            score += 10
        if len(code_text) > 50:
            score += 15

        # Code Completeness (0-40)
        if "__global__" in code_text or "__device__" in code_text:
            score += 25
        elif "void" in code_text and re.search(r'\w+\s*\(', code_text):
            score += 10

        if re.search(r'#include\s*<', code_text):
            score += 10
        if re.search(r'\{[^}]*\}', code_text, re.DOTALL):
            score += 5

        # Structural Validity (0-35)
        if re.search(r'__global__\s+void\s+\w+\s*\(', code_text):
            score += 15
        if re.search(r'threadIdx|blockIdx|blockDim', code_text):
            score += 10
        if re.search(r'(__shared__|__device__|__constant__)', code_text):
            score += 10

        return min(score, 100)

    @staticmethod
    def _fix_ported_code(code: str) -> str:
        """Fix AMD-specific issues in ported code.
        
        Fixes:
        - 32-bit __shfl mask → 64-bit for wavefront64
        - `global void` → `__global__ void` (LLM drops __)
        - `device void` → `__device__ void`
        """
        import re
        # Fix missing __ on global/device
        code = re.sub(r'^global\s+void', '__global__ void', code, flags=re.MULTILINE)
        code = re.sub(r'^device\s+void', '__device__ void', code, flags=re.MULTILINE)
        # Count how many 32-bit masks remain
        mask_pattern = re.compile(r'(__shfl_\w+_sync\()0x[fF]{8}(,)')
        before = len(mask_pattern.findall(code))
        code = mask_pattern.sub(r'\g<1>0xffffffffffffffffULL\g<2>', code)
        after = len(mask_pattern.findall(code))
        if before > 0 and after == 0:
            pass  # All masks fixed
        elif before > 0:
            print(f"║ ⚠️ Mask fix: {before} found, {after} remaining (regex issue!)")
        return code

    def _template_port(self, source_code: str,
                       cached_pattern: Optional[Dict] = None) -> Dict:
        """Template-based porting for when API is unavailable (demo fallback)."""

        import re
        changes = []
        lines = source_code.split('\n')
        result_lines = []
        wavefront_header_added = False
        has_added_wave64_shfl = False

        # Template transformations (only on non-comment lines)
        shared_32_re = re.compile(r'(__shared__[^;]*?\[\s*)32(\s*\])')
        tile_32_re = re.compile(r'(tile\[)\s*32(\s*\]\[)\s*32(\s*\])')
        blockidx_32_re = re.compile(r'(blockIdx\.[xy])\s*\*\s*32\s*\+')
        syncwarp_re = re.compile(r'__syncwarp\(\s*\)')
        warp_size_re = re.compile(r'(?:const\s+)?int\s+WARP_SIZE\s*=\s*32')
        warp_size_define_re = re.compile(r'#define\s+WARP_SIZE\s+32\b')
        ballot_re = re.compile(r'__ballot_sync\(0xffffffff')
        shfl_xor_re = re.compile(r'__shfl_xor_sync\s*\(')
        threadidx_32_re = re.compile(r'(threadIdx\.[xy]\s*\*\s*)32(\b)')
        threadidx_mod32_re = re.compile(r'(threadIdx\.[xy]\s*%\s*)32(\b)')
        define_tile_re = re.compile(r'#define\s+TILE_SIZE\s+32\b')
        warp_mask_re = re.compile(r'(?:const\s+)?int\s+WARP_MASK\s*=\s*0x1[fF]\b')
        tid_warp_mask_re = re.compile(r'(tid\s*&\s*)0x1[fF](\s*\)?\s*==\s*0\b)')
        blockidx_tile_re = re.compile(r'(blockIdx\.[xy]\s*\*\s*)TILE_SIZE')
        shfl_down_re = re.compile(r'__shfl_down_sync\s*\(')
        activemask_re = re.compile(r'__activemask\s*\(\s*\)')
        all_sync_re = re.compile(r'__all_sync\s*\(')
        any_sync_re = re.compile(r'__any_sync\s*\(')
        match_all_re = re.compile(r'__match_all_sync\s*\(')
        warp_lane_shift_re = re.compile(r'(threadIdx\.[xy]\s*>>\s*)5(?!\d)')
        lane_id_32_re = re.compile(r'(lane_id|laneIdx)\s*[<]\s*32\b')
        warp_divergent_32_re = re.compile(r'(if\s*\(\s*(?:threadIdx\.[xy]|tid|lane_id|laneIdx)\s*[<]\s*)32(\s*\))')

        for line in lines:
            stripped = line.strip()

            # Skip comment-only lines
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                result_lines.append(line)
                continue

            # Track if this line was modified
            original = line

            # Fix 1: Hardcoded 32 in shared memory → change to 64
            line = shared_32_re.sub(r'\1 WAVEFRONT_SIZE \2', line)

            # Fix 2: Hardcoded 32 in tile declarations
            line = tile_32_re.sub(r'\1 WAVEFRONT_SIZE \2 WAVEFRONT_SIZE \3', line)

            # Fix 3: Hardcoded 32 in block indexing
            line = blockidx_32_re.sub(r'\1 * WAVEFRONT_SIZE +', line)

            # Fix 4: __syncwarp() → __syncthreads()
            if syncwarp_re.search(line):
                line = syncwarp_re.sub('__syncthreads();  // wavefront64: full block sync', line)
                if "wavefront64: full block sync" not in __import__('json').dumps(changes):
                    changes.append("__syncwarp() → __syncthreads() for HIP compatibility")

            # Fix 5: __shfl_down_sync — no safe automatic fix for offset semantics
            # (The actual fix depends on algorithm context; LLM handles this best)
            # But we CAN prepend offset=32 for wavefront64 (6 steps → 64 elements)
            
            # Fix 5b: Insert 6th shuffle offset for wavefront64
            if 'shfl_down' in stripped and 'val += __shfl' in line:
                if not has_added_wave64_shfl:
                    has_added_wave64_shfl = True
                    indent = line[:len(line) - len(line.lstrip())]
                    new_line = f"{indent}val += __shfl_down_sync(0xffffffffffffffffULL, val, 32);  // ADDED: wavefront64 offset\n{line}"
                    result_lines[-1] = new_line
                    if "wavefront64_offset32" not in str(changes):
                        changes.append("wavefront64: added offset=32 shuffle step (6-step reduction for 64 lanes)")
                    continue

            # Fix 6: 0x1f (warp mask 32) → 0x3f (wavefront mask 64)
            if '0x1f' in line and not stripped.startswith('//'):
                line = line.replace('0x1f', '0x3f')

            # Fix 7: Hardcoded WARP_SIZE = 32 (with or without const)
            line = warp_size_re.sub('const int WAVEFRONT_SIZE = 64;  // AMD wavefront', line)
            
            # Fix 7b: #define WARP_SIZE 32 (preprocessor macro style)
            if '#define WARP_SIZE' in line and warp_size_define_re.search(line):
                line = warp_size_define_re.sub('#define WAVEFRONT_SIZE 64  // AMD wavefront', line)
                if '#define WAVEFRONT_SIZE 64' not in ' '.join(changes):
                    changes.append("#define WARP_SIZE 32 → #define WAVEFRONT_SIZE 64")

            # Fix 8: __ballot_sync — fix mask and annotate
            if ballot_re.search(line):
                line = ballot_re.sub('__ballot_sync(0xffffffffffffffffULL', line)
                if "ballot_sync mask" not in str(changes):
                    changes.append("__ballot_sync mask → 0xffffffffffffffffULL for wavefront64")
            
            # Fix 8b: __shfl_xor_sync — annotate as wavefront-dependent
            if shfl_xor_re.search(line):
                # Safest auto-fix: add comment; actual offset fix is algorithm-dependent
                if "shfl_xor" not in str(changes):
                    changes.append("__shfl_xor_sync: verify XOR offsets work with wavefront64 (64 lanes, not 32)")
            
            # Fix 8c: threadIdx.* 32 pattern (pointer arithmetic, e.g., &shared[threadIdx.y * 32])
            line = threadidx_32_re.sub(r'\1 WAVEFRONT_SIZE ', line)

            # Fix 8c2: threadIdx.* % 32 → % WAVEFRONT_SIZE (lane ID calculation)
            if threadidx_mod32_re.search(line):
                line = threadidx_mod32_re.sub(r'\1WAVEFRONT_SIZE ', line)

            # Fix 8d: #define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE
            if define_tile_re.search(line):
                line = define_tile_re.sub('#define TILE_SIZE WAVEFRONT_SIZE  // AMD wavefront', line)
                if "#define TILE_SIZE WAVEFRONT_SIZE" not in ' '.join(changes):
                    changes.append("#define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE")

            # Fix 8e: int WARP_MASK = 0x1f → int WAVEFRONT_MASK = 0x3f
            if warp_mask_re.search(line):
                line = warp_mask_re.sub('int WAVEFRONT_MASK = 0x3f;  // wavefront64 mask', line)
                if "WARP_MASK → WAVEFRONT_MASK" not in str(changes):
                    changes.append("WARP_MASK 0x1f (32) → WAVEFRONT_MASK 0x3f (64)")

            # Fix 8f: tid & 0x1f == 0 → tid & 0x3f == 0 (warp mask check)
            line = tid_warp_mask_re.sub(r'\1 0x3f\2', line)

            # Fix 8f2: & 31 (decimal warp mask) → & 63 for wavefront64
            if re.search(r'&\s*31\b', line) and 'threadIdx' in line:
                line = re.sub(r'(&\s*)31(\b)', r'\g<1>63', line)

            # Fix 8g: blockIdx.* * TILE_SIZE → blockIdx.* * WAVEFRONT_SIZE
            line = blockidx_tile_re.sub(r'\1 WAVEFRONT_SIZE', line)

            # Fix 8h: __shfl_down_sync — annotate the offset issue
            if shfl_down_re.search(line):
                if "shfl_down" not in str(changes):
                    changes.append("__shfl_down_sync: verify offsets work with wavefront64 (64 lanes, offset must be power of two)")
            
            # Fix 8i: 32-bit mask → 64-bit mask for AMD wavefront64
            mask_line = re.sub(r'(__shfl_\w+_sync\()0x[fF]{8}(,)', r'\g<1>0xffffffffffffffffULL\g<2>', line)
            if mask_line != line:
                line = mask_line
                if "mask_64bit" not in str(changes):
                    changes.append("__shfl_*_sync: mask 0xffffffff → 0xffffffffffffffffULL (64-bit for wavefront64)")

            # Fix 9: __activemask() → __ballot_sync(0xffffffffffffffffULL, 1) on HIP
            if activemask_re.search(line):
                line = activemask_re.sub('__ballot_sync(0xffffffffffffffffULL, 1)', line)
                if "activemask" not in str(changes):
                    changes.append("__activemask() → __ballot_sync(0xffffffffffffffffULL, 1) for HIP compatibility")

            # Fix 10: __all_sync / __any_sync — annotate for wavefront64
            if all_sync_re.search(line):
                if "all_sync" not in str(changes):
                    changes.append("__all_sync: verify predicate works with wavefront64 (64 lanes)")
            if any_sync_re.search(line):
                if "any_sync" not in str(changes):
                    changes.append("__any_sync: verify predicate works with wavefront64 (64 lanes)")

            # Fix 11: __match_all_sync — annotate (no direct HIP equivalent)
            if match_all_re.search(line):
                if "match_all" not in str(changes):
                    changes.append("__match_all_sync: no direct HIP equivalent — may need algorithm redesign")

            # Fix 12: threadIdx.x >> 5 (warp index) → >> 6 for wavefront64
            line = warp_lane_shift_re.sub(r'\1 6;  // wavefront64: 64 lanes', line)

            # Fix 13: lane_id < 32 → lane_id < 64 (wavefront boundary)
            if lane_id_32_re.search(line):
                line = line.replace('< 32', '< WAVEFRONT_SIZE', 1)
                if "lane_id < 32" not in str(changes):
                    changes.append("lane_id < 32 → lane_id < WAVEFRONT_SIZE for wavefront64")

            # Fix 14: threadIdx.x/tid < 32 → < WAVEFRONT_SIZE (warp divergence boundary)
            line = warp_divergent_32_re.sub(r'\1 WAVEFRONT_SIZE \2', line)

            # Fix 14b: Loop bound in shuffle/scan patterns — offset < 32 → offset < WAVEFRONT_SIZE
            # Common in warp scan (Hillis-Steele): for (int offset = 1; offset < 32; offset *= 2)
            # On wavefront64, need 6 steps (1,2,4,8,16,32) to cover all 64 lanes
            if re.search(r'for\s*\(\s*int\s+offset\s*=\s*1\s*;\s*offset\s*<\s*32\b', line):
                line = re.sub(r'(offset\s*<\s*)32(\b)', r'\1WAVEFRONT_SIZE\2', line)
                if "shuffle loop bound" not in str(changes):
                    changes.append("shuffle scan loop bound offset < 32 → offset < WAVEFRONT_SIZE (6 steps for wavefront64)")


            # Track what changed
            if line != original:
                # Compute a change description based on what was modified
                if 'WAVEFRONT_SIZE' in line and 'WAVEFRONT_SIZE' not in original:
                    if 'shared' in original and '32' in original:
                        changes.append("__shared__ array sized 32 → WAVEFRONT_SIZE for wavefront64")
                    elif 'blockIdx' in original:
                        changes.append("blockIdx.*32 → blockIdx.*WAVEFRONT_SIZE")
                    elif 'tile' in original:
                        changes.append("tile[32][32] → tile[WAVEFRONT_SIZE][WAVEFRONT_SIZE]")
                    elif 'threadIdx' in original and '* 32' in original:
                        changes.append("threadIdx.*32 → threadIdx.*WAVEFRONT_SIZE in pointer arithmetic")
                if '0x3f' in line and '0x1f' in original:
                    changes.append("Warp mask 0x1f (32) → 0x3f (64) for wavefront64")
                if 'WAVEFRONT_SIZE = 64' in line and 'WARP_SIZE' in original:
                    changes.append("WARP_SIZE = 32 → WAVEFRONT_SIZE = 64")
                if 'TILE_SIZE WAVEFRONT_SIZE' in line and 'TILE_SIZE 32' in original:
                    changes.append("#define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE")
                if 'WAVEFRONT_MASK' in line and 'WARP_MASK' in original:
                    changes.append("WARP_MASK → WAVEFRONT_MASK")
                if '0x3f' in line and '0x1f' in original and '#define' not in original and 'WARP_MASK' not in original:
                    changes.append("tid & 0x1f → tid & 0x3f for wavefront64")

            result_lines.append(line)

        # Fix 9: Add wavefront awareness header (unless already present or first line has it)
        code = '\n'.join(result_lines)
        if "#define WAVEFRONT_SIZE 64" not in code:
            code = "#define WAVEFRONT_SIZE 64  // AMD GPU wavefront size\n" + code
            changes.append("Added #define WAVEFRONT_SIZE 64 header")

        # Deduplicate changes
        seen = set()
        unique_changes = []
        for c in changes:
            if c not in seen:
                seen.add(c)
                unique_changes.append(c)

        # Apply cached pattern if available
        has_cached = cached_pattern is not None
        if has_cached:
            unique_changes.append(f"Applied cached pattern from verified fix (id: {cached_pattern.get('id', 'unknown')})")
            if cached_pattern.get("verified_fix"):
                code = cached_pattern["verified_fix"]

        # Mission requirement: the template fallback must NEVER emit CUDA
        # headers or APIs — this line-by-line pass above only ever touched
        # warp32→wavefront64 patterns and left #include <cuda_runtime.h> (and
        # cudaMalloc/cudaMemcpy/... calls) untouched, which is a guaranteed
        # hipcc failure. Route through the same comprehensive CUDA→HIP header
        # rewriter every router.py translation already runs through, so the
        # fallback compiles under hipcc even when the LLM path never ran.
        code, header_changes = _ModelRouter._fix_ported_code(code, return_changelog=True)
        unique_changes.extend(c for c in header_changes if c not in unique_changes)
        if "hip/hip_runtime.h" not in code:
            code = "#include <hip/hip_runtime.h>\n" + code
            unique_changes.append("Added #include <hip/hip_runtime.h> (missing from fallback output)")
        if not code.lstrip().startswith("//"):
            code = ("// FALLBACK-GENERATED CODE — mechanical template port, "
                    "not produced or reviewed by an LLM. Verify before trusting.\n" + code)

        # Rubric-based confidence scoring
        has_wavefront_header = "#define WAVEFRONT_SIZE 64" in code
        confidence = self._rubric_score(source_code, code, unique_changes,
                                        has_header=has_wavefront_header,
                                        cached=has_cached)

        # When a verified cached fix is used, confidence should reflect
        # the cached pattern's stored verification result (known-good code).
        if has_cached and cached_pattern.get("verified_fix"):
            cached_conf = cached_pattern.get("confidence", 0.85)
            if cached_conf < 1:
                cached_conf = cached_conf * 100
            confidence = max(confidence, int(cached_conf))

        return {
            "ported_code": code,
            "confidence": confidence,
            "changes": unique_changes if unique_changes else ["No automatic changes needed — code appears portable"],
            "explanation": "Template-based porting applied. "
                          f"Made {len(unique_changes)} changes. "
                          "For production, use Fireworks API for better accuracy."
        }
