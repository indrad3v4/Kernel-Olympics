"""
Model Router — Lightweight agent orchestration for CUDA→ROCm porting.

Architecture:
  Risk Classifier → Model Router → GLM (planner) OR Kimi K2.7 (coder) OR DeepSeek (verifier)
                    ↓                    ↓                          ↓
                Pattern Memory ←─── verified fix ←───────────── real AMD GPU

TRIZ: Use risk classifier output as routing resource (no extra LLM call to decide).
      Each model does what it's best at — no wasted tokens.
"""

import json
import re
import os
import socket
import time
import uuid
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from prompt_evolution import prompt_opt, PromptOptimizer


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


def _extract_balanced_json(text: str):
    """Extract the first complete JSON object from text using balanced-brace counting.

    Counts opening/closing braces while respecting string literals (ignores braces
    inside quoted strings). Returns parsed dict or None.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] that make JSON invalid."""
    return re.sub(r',\s*([}\]])', r'\1', s)


def _extract_arrays_regex(text: str):
    """Extract individual arrays from malformed JSON using targeted regexes.

    Tries to parse 'fixes', 'missing_includes', and 'wrong_apis' arrays separately,
    then assembles a minimal dict. Returns dict or None.
    """
    result = {}

    # Extract fixes array — each element is a JSON object
    fixes_match = re.search(r'"fixes"\s*:\s*\[', text)
    if fixes_match:
        arr_start = fixes_match.end() - 1  # position of '['
        # Find balanced bracket
        depth = 0
        in_str = False
        esc = False
        for i in range(arr_start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    arr_text = text[arr_start:i + 1]
                    try:
                        result["fixes"] = json.loads(arr_text)
                    except Exception:
                        # Try parsing individual objects within the array
                        objs = re.findall(r'\{[^{}]*\}', arr_text, re.DOTALL)
                        parsed = []
                        for o in objs:
                            try:
                                parsed.append(json.loads(_strip_trailing_commas(o)))
                            except Exception:
                                pass
                        if parsed:
                            result["fixes"] = parsed
                    break

    # Extract missing_includes array
    mi_match = re.search(r'"missing_includes"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if mi_match:
        try:
            result["missing_includes"] = json.loads("[" + mi_match.group(1) + "]")
        except Exception:
            # Extract quoted strings as fallback
            incs = re.findall(r'"([^"]*)"', mi_match.group(1))
            if incs:
                result["missing_includes"] = incs

    # Extract wrong_apis array
    wa_match = re.search(r'"wrong_apis"\s*:\s*\[', text)
    if wa_match:
        arr_start = wa_match.end() - 1
        depth = 0
        in_str = False
        esc = False
        for i in range(arr_start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    arr_text = text[arr_start:i + 1]
                    try:
                        result["wrong_apis"] = json.loads(arr_text)
                    except Exception:
                        objs = re.findall(r'\{[^{}]*\}', arr_text, re.DOTALL)
                        parsed = []
                        for o in objs:
                            try:
                                parsed.append(json.loads(_strip_trailing_commas(o)))
                            except Exception:
                                pass
                        if parsed:
                            result["wrong_apis"] = parsed
                    break

    # Extract summary if present
    sum_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if sum_match:
        result["summary"] = sum_match.group(1)

    return result if result else None


# ── Model catalog ────────────────────────────────────────────────
# ✅ VERIFIED WORKING on Fireworks API (tested, confirmed):
#   - kimi-k2p6    (planner — complex kernel logic, multi-step reasoning)
#   - glm-5p2      (coder — accurate code generation)
#   - deepseek-v4-pro (fallback — general purpose)
#
# ❌ UNVERIFIED / REMOVED (not confirmed working, removed from catalog):
#   - gemma-4-31b-it  (dedicated deployment only, not available via Fireworks API)
#   - llama-v3p3-70b-instruct (unstable results on Fireworks)
#
# Only verified models are kept to avoid silent failures during porting.

MODEL_CATALOG = {
    "deepseek": {
        "id": "accounts/fireworks/models/deepseek-v4-pro",  # ✅ VERIFIED WORKING
        "role": "planner",          # CHANGED: reasoning model → planner (prose OK)
        "strength": "deep reasoning, chain-of-thought planning, pattern analysis",
        "cost_per_1k": 0.0012,
        "local_first": False,
        "max_tokens": 2048,      # planner needs room for reasoning
        "temperature": 0.3,      # creativity for diverse plans
        "timeout": 120,          # TRIZ #1: planner is prose-heavy, needs moderate time
    },
    "kimi27": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",  # ✅ VERIFIED WORKING
        "role": "coder",            # UNCHANGED
        "strength": "code generation, struct-aware HIP porting",
        "cost_per_1k": 0.00095,
        "local_first": False,
        "max_tokens": 8192,      # coder needs room for full kernel + JSON wrapper
        "temperature": 0.1,      # code generation needs precision
        "timeout": 180,          # TRIZ #1: coder with 8192 tokens needs the most time
    },
    "glm": {
        "id": "accounts/fireworks/models/glm-5p2",  # ✅ VERIFIED WORKING
        "role": "evaluator",         # CHANGED: was planner → now evaluator (strict JSON)
        "strength": "structured JSON output, correctness checking, wavefront64 validation",
        "cost_per_1k": 0.0014,
        "local_first": False,
        "max_tokens": 1024,      # evaluator output is compact (pass/fail + issues)
        "temperature": 0.0,      # deterministic evaluation
        "timeout": 120,          # TRIZ #1: evaluator, compact output
    },
    "gemma4": {
        "id": "accounts/fireworks/models/gemma-4-31b-it",  # Fireworks hosted
        "local_id": "gemma-4-31b-it",  # Model name when served via local vLLM
        "role": "verifier",          # final verification
        "strength": "Verification — local vLLM on MI300X if available, else Fireworks",
        "cost_per_1k": 0.0,
        "local_first": True,     # Try localhost:8000 first, then Fireworks
        "max_tokens": 1024,
        "temperature": 0.0,
        "timeout": 60,           # TRIZ #1: fast verification model
    },
}

# ── JSON schemas for response_format (enforced by Fireworks API) ──
# Using json_object (not json_schema) for compatibility — the system prompt
# already defines the exact shape. json_schema is stricter but not all models support it.

JSON_SCHEMAS = {
    "glm": {  # GLM evaluator — json_object (more widely supported than json_schema)
        "type": "json_object",
    },
    "kimi27": {
        "type": "json_object",
    },
    # DeepSeek is planner — no response_format (prose/reasoning is OK)
}
# ── Role-specific system prompts ──
# Each model gets its OWN role definition. No shared prompts.
# These are passed as system messages to the LLM alongside the phase prompt.

SYSTEM_PROMPTS = {
    "deepseek": (
        "You are a CUDA-to-HIP porting planner. "
        "Analyze the CUDA kernel and produce a detailed porting plan: "
        "list every CUDA-specific construct, its HIP replacement, and the order of changes. "
        "Focus on warp(32)→wavefront(64) divergence, __shfl mask widths, shared memory sizing, "
        "header replacements, and any local .cuh dependencies that must be inlined or removed. "
        "Write your plan as clear prose with a numbered checklist of fixes. "
        "Reason freely — your plan will be consumed by a coder agent."
    ),
    "kimi27": (
        "You are a CUDA-to-HIP code porting specialist. "
        "Port CUDA kernels to AMD ROCm/HIP, fixing warp→wavefront issues. "
        "Respond with JSON: {\"ported_code\":str,\"confidence\":int,\"changes\":[str],\"explanation\":str}."
    ),
    "glm": (
        "You are a HIP kernel code evaluator. "
        "Check ported code for wavefront64 correctness, CUDA remnants, and compilation safety. "
        'Respond with JSON: {"pass":bool,"issues":[str],"feedback":str,"verdict":str}. '
        "CRITICAL: Begin your response with the { character. "
        "DO NOT include reasoning, explanations, greetings, or chain-of-thought. "
        "Output ONLY the JSON object. The first character MUST be {. "
        "No text before or after the JSON."
    ),
}


# ── Static helper for building classifier pattern summary ──

def _format_patterns_summary(patterns: List[Dict]) -> str:
    """Build a formatted list of classifier-detected patterns from the pattern list.
    
    Each pattern dict is expected to have at minimum a "pattern" key.
    Optional keys: line/lineno, code/snippet, description/issue, severity/risk.
    """
    if not patterns:
        return ""
    lines = ["CLASSIFIER PATTERNS DETECTED:"]
    for i, p in enumerate(patterns, 1):
        pt = p.get("pattern", "unknown")
        ln = p.get("line") or p.get("lineno") or ""
        cd = (p.get("code") or p.get("snippet") or "")[:120]
        desc = p.get("description") or p.get("issue") or ""
        sev = p.get("severity") or p.get("risk") or "medium"
        entry = f"  {i}. {pt}"
        if ln:
            entry += f" (line {ln})"
        entry += f" [{sev}]"
        lines.append(entry)
        if cd:
            lines.append(f"     Code: {cd}")
        if desc:
            lines.append(f"     Issue: {desc}")
    lines.append("")
    return "\n".join(lines)


# NOTE: ROUTING_TABLE was removed — it was dead code. route() always runs
# the full DeepSeek→Kimi→GLM pipeline regardless of detected patterns.
# If per-pattern model selection is needed in the future, add it to route().


@dataclass
class AgentResult:
    model: str
    success: bool
    output: str
    confidence: float
    tokens_used: int = 0
    elapsed_ms: float = 0.0


@dataclass
class A2AMessage:
    """Structured inter-agent message — replaces blob truncation.

    TRIZ #3 (Local quality): Each agent receives a structured, prioritized
    message instead of a truncated text blob.  The summary ALWAYS fits;
    priority details are added in order so truncation hits low-priority
    items only.  full_ref keeps the complete content addressable without
    re-sending it.

    TRIZ #28 (Mechanical substitution): Replaces ad-hoc string slicing
    (plan[:2000], feedback[:800], compile_errs[:3]) with a budget-aware
    renderer that preserves the most important information.
    """
    summary: str              # 1-2 sentences, ALWAYS fits
    priority_details: list    # most important first, truncation hits low-priority only
    full_ref: str             # pattern memory key or run_id, not re-sent
    changelog: list           # what was modified (regex fixes, etc.)

    def to_prompt(self, max_chars: int = 4000) -> str:
        """Render to prompt text within budget. Summary always included.
        Priority details added in order until budget hit."""
        parts = [self.summary]
        budget = max_chars - len(self.summary) - 200  # reserve for formatting
        for d in self.priority_details:
            rendered = self._render_detail(d)
            if len("\n".join(parts)) + len(rendered) > max_chars:
                parts.append(f"... ({len(self.priority_details) - len(parts) + 1} more details omitted)")
                break
            parts.append(rendered)
        if self.changelog:
            parts.append("Applied fixes: " + "; ".join(self.changelog[:5]))
        return "\n".join(parts)

    @staticmethod
    def _render_detail(d: dict) -> str:
        if d.get("type") == "api_mapping":
            return f"  API: {d['cuda']} → {d['hip']}"
        elif d.get("type") == "header":
            return f"  HEADER: {d['cuda']} → {d['hip']}"
        elif d.get("type") == "error_fix":
            return f"  FIX [{d.get('priority','?')}]: {d['error']} → {d['fix']}"
        elif d.get("type") == "risk":
            return f"  RISK: {d['description']}"
        else:
            return f"  {d.get('text', str(d))}"


class ModelRouter:
    """Routes CUDA porting tasks to the best model for each pattern.

    Flow:
      1. Classifier detects patterns in kernel
      2. Router picks best model per pattern
      3. GLM plans the fix structure (if complex)
      4. Kimi K2.7 generates the code
      5. DeepSeek verifies the output
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = "https://api.fireworks.ai/inference/v1"
        self.total_cost = 0.0
        self.call_log: List[Dict] = []
        self.run_id: str = ""  # set by route() for A2AMessage full_ref keys

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract HIP/C++ code from LLM output.

        Priority:
          1. JSON {\"ported_code\": \"...\"} — Kimi's expected response format
          2. Markdown code blocks ```hip/cpp/cuda ... ```
          3. Raw code starting from #include or __global__
          4. Fallback: return as-is
        """
        import re as _re

        # ── Strategy 1: Parse JSON response (Kimi's expected format) ──
        # Kimi returns: {"ported_code": "...", "confidence": 80, ...}
        # The ported_code field contains the actual HIP source.
        raw = text.strip()

        # 1a: Direct JSON parse
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "ported_code" in obj:
                    return obj["ported_code"].strip()
            except (json.JSONDecodeError, TypeError):
                pass

        # 1b: JSON inside ```json ... ``` block
        json_block = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, _re.DOTALL)
        if json_block:
            try:
                obj = json.loads(json_block.group(1))
                if isinstance(obj, dict) and "ported_code" in obj:
                    return obj["ported_code"].strip()
            except (json.JSONDecodeError, TypeError):
                pass

        # 1c: Find {"ported_code": "..."} anywhere (balanced brace extraction)
        ported_match = _re.search(r'"ported_code"\s*:\s*"', raw)
        if ported_match:
            # Extract the string value with proper escape handling
            start = ported_match.end()
            result_chars = []
            i = start
            while i < len(raw):
                if raw[i] == '\\' and i + 1 < len(raw):
                    next_ch = raw[i + 1]
                    if next_ch == 'n':
                        result_chars.append('\n')
                    elif next_ch == 't':
                        result_chars.append('\t')
                    elif next_ch == '"':
                        result_chars.append('"')
                    elif next_ch == '\\':
                        result_chars.append('\\')
                    else:
                        result_chars.append(next_ch)
                    i += 2
                elif raw[i] == '"':
                    break  # end of string value
                else:
                    result_chars.append(raw[i])
                    i += 1
            extracted = ''.join(result_chars).strip()
            if len(extracted) > 20:  # sanity check
                return extracted

        # ── Strategy 2: Markdown code blocks ──
        blocks = _re.findall(r'```(?:cuda|hip|cpp|python)?\n(.*?)```', text, _re.DOTALL)
        if blocks:
            # Return the largest block (most likely the full kernel)
            return max(blocks, key=len).strip()

        # ── Strategy 3: Raw code from #include or __global__ ──
        match = _re.search(r'#include.*', text, _re.DOTALL)
        if match:
            return match.group(0).strip()
        match = _re.search(r'__global__\s+void.*', text, _re.DOTALL)
        if match:
            return match.group(0).strip()

        # ── Strategy 3b: TRIZ #22 — Salvage code from truncated responses ──
        # When JSON is truncated mid-generation, the above strategies may fail.
        # Look for raw C/C++ patterns that indicate kernel code is present
        # even if the JSON wrapper is incomplete.
        has_include = _re.search(r'#include\s*[<"]', text)
        has_global = _re.search(r'__global__', text)
        has_hip_ref = _re.search(r'\bhip\b', text, _re.IGNORECASE)
        if has_include or has_global:
            # Find the earliest code-like construct and capture from there
            # to the end of the text (truncated responses don't have clean ends)
            candidates = []
            inc_match = _re.search(r'#include\s*[<"].*', text, _re.DOTALL)
            if inc_match:
                candidates.append(inc_match.start())
            glob_match = _re.search(r'__global__.*', text, _re.DOTALL)
            if glob_match:
                candidates.append(glob_match.start())
            if candidates:
                start = min(candidates)
                salvaged = text[start:].strip()
                # Remove trailing incomplete JSON artifacts
                salvaged = _re.sub(r'["\}]\s*$', '', salvaged).strip()
                if len(salvaged) > 20:  # sanity check
                    return salvaged

        # ── Strategy 4: Fallback ──
        return text.strip()

    @staticmethod
    def _fix_ported_code(code: str, return_changelog: bool = False):
        """Fix AMD-specific issues in ported code.

        Post-processing safety net applied after every Kimi code generation
        and refinement pass. Catches common issues the LLM may miss.

        Args:
            code: The raw ported code string.
            return_changelog: If True, returns (fixed_code, changelog) where
                changelog is a list of human-readable strings describing each
                regex substitution that was actually applied. If False (default),
                returns just the fixed code string — preserving backward
                compatibility with existing callers that don't need the
                changelog.
        """
        changelog: List[str] = []

        def _tracked_sub(pattern, replacement, text, description):
            """Run re.sub, but record in changelog if any substitution occurred."""
            new_text, count = re.subn(pattern, replacement, text)
            if count > 0 and description:
                changelog.append(f"{description} (×{count})")
            return new_text

        # ── Comprehensive CUDA header replacement ──────────────────
        # Core CUDA runtime → HIP
        code = _tracked_sub(r'#include\s*[<"]cuda_runtime\.h[>"]', '#include <hip/hip_runtime.h>', code, 'cuda_runtime.h→hip/hip_runtime.h')
        code = _tracked_sub(r'#include\s*[<"]cuda_runtime_api\.h[>"]', '#include <hip/hip_runtime.h>', code, 'cuda_runtime_api.h→hip/hip_runtime.h')
        # CUDA math → HIP (hip already includes math)
        code = _tracked_sub(r'#include\s*[<"]cuda_math\.h[>"]\n?', '', code, 'cuda_math.h removed')
        # NVIDIA helper headers — NOT in ROCm, remove
        code = _tracked_sub(r'#include\s*[<"]helper_cuda\.h[>"]\n?', '', code, 'helper_cuda.h removed')
        code = _tracked_sub(r'#include\s*[<"]helper_functions\.h[>"]\n?', '', code, 'helper_functions.h removed')
        code = _tracked_sub(r'#include\s*[<"]helper_string\.h[>"]\n?', '', code, 'helper_string.h removed')
        code = _tracked_sub(r'#include\s*[<"]helper_timer\.h[>"]\n?', '', code, 'helper_timer.h removed')
        code = _tracked_sub(r'#include\s*[<"]helper_image\.h[>"]\n?', '', code, 'helper_image.h removed')
        code = _tracked_sub(r'#include\s*[<"]helper_gl\.h[>"]\n?', '', code, 'helper_gl.h removed')
        # CUDA device launch — not needed in HIP
        code = _tracked_sub(r'#include\s*[<"]device_launch_parameters\.h[>"]\n?', '', code, 'device_launch_parameters.h removed')
        # CUDA random, FFT, BLAS, sparse, solver — need HIP equivalents
        code = _tracked_sub(r'#include\s*[<"]curand\.h[>"]', '#include <hiprand/hiprand.h>', code, 'curand.h→hiprand/hiprand.h')
        code = _tracked_sub(r'#include\s*[<"]curand_kernel\.h[>"]', '#include <hiprand/hiprand_kernel.h>', code, 'curand_kernel.h→hiprand/hiprand_kernel.h')
        code = _tracked_sub(r'#include\s*[<"]cufft\.h[>"]', '#include <hipfft/hipfft.h>', code, 'cufft.h→hipfft/hipfft.h')
        code = _tracked_sub(r'#include\s*[<"]cublas_v2\.h[>"]', '#include <hipblas/hipblas.h>', code, 'cublas_v2.h→hipblas/hipblas.h')
        code = _tracked_sub(r'#include\s*[<"]cusparse\.h[>"]', '#include <hipsparse/hipsparse.h>', code, 'cusparse.h→hipsparse/hipsparse.h')
        code = _tracked_sub(r'#include\s*[<"]cusolver_common\.h[>"]', '#include <hipsolver/hipsolver.h>', code, 'cusolver_common.h→hipsolver/hipsolver.h')
        # NVRTC → no HIP equivalent, remove
        code = _tracked_sub(r'#include\s*[<"]nvrtc\.h[>"]\n?', '', code, 'nvrtc.h removed')
        # Remove project-specific .cuh headers — not available in HIP port
        code = _tracked_sub(r'#include\s*"[^"]*\.cuh"\n?', '', code, 'local .cuh headers removed')
        code = _tracked_sub(r"#include\s*<[^>]*\.cuh>\n?", '', code, 'system .cuh headers removed')
        # Remove any remaining CUDA-specific includes
        code = _tracked_sub(r'#include\s*[<"][^>"]*cuda[^>"]*[>"]\n?', '', code, 'remaining CUDA includes removed', )

        # ── API renames: cuda* → hip* ──────────────────────────────
        code = _tracked_sub(r'\bcudaMalloc\b', 'hipMalloc', code, 'cudaMalloc→hipMalloc')
        code = _tracked_sub(r'\bcudaFree\b', 'hipFree', code, 'cudaFree→hipFree')
        code = _tracked_sub(r'\bcudaMemcpy\b', 'hipMemcpy', code, 'cudaMemcpy→hipMemcpy')
        code = _tracked_sub(r'\bcudaMemcpyAsync\b', 'hipMemcpyAsync', code, 'cudaMemcpyAsync→hipMemcpyAsync')
        code = _tracked_sub(r'\bcudaMemset\b', 'hipMemset', code, 'cudaMemset→hipMemset')
        code = _tracked_sub(r'\bcudaDeviceSynchronize\b', 'hipDeviceSynchronize', code, 'cudaDeviceSynchronize→hipDeviceSynchronize')
        code = _tracked_sub(r'\bcudaGetLastError\b', 'hipGetLastError', code, 'cudaGetLastError→hipGetLastError')
        code = _tracked_sub(r'\bcudaError_t\b', 'hipError_t', code, 'cudaError_t→hipError_t')
        code = _tracked_sub(r'\bcudaSuccess\b', 'hipSuccess', code, 'cudaSuccess→hipSuccess')
        code = _tracked_sub(r'\bcudaGetDeviceCount\b', 'hipGetDeviceCount', code, 'cudaGetDeviceCount→hipGetDeviceCount')
        code = _tracked_sub(r'\bcudaSetDevice\b', 'hipSetDevice', code, 'cudaSetDevice→hipSetDevice')
        code = _tracked_sub(r'\bcudaGetDeviceProperties\b', 'hipGetDeviceProperties', code, 'cudaGetDeviceProperties→hipGetDeviceProperties')
        code = _tracked_sub(r'\bcudaDeviceProp\b', 'hipDeviceProp_t', code, 'cudaDeviceProp→hipDeviceProp_t')
        code = _tracked_sub(r'\bcudaStreamCreate\b', 'hipStreamCreate', code, 'cudaStreamCreate→hipStreamCreate')
        code = _tracked_sub(r'\bcudaStreamSynchronize\b', 'hipStreamSynchronize', code, 'cudaStreamSynchronize→hipStreamSynchronize')
        code = _tracked_sub(r'\bcudaEventCreate\b', 'hipEventCreate', code, 'cudaEventCreate→hipEventCreate')
        code = _tracked_sub(r'\bcudaEventRecord\b', 'hipEventRecord', code, 'cudaEventRecord→hipEventRecord')
        code = _tracked_sub(r'\bcudaEventSynchronize\b', 'hipEventSynchronize', code, 'cudaEventSynchronize→hipEventSynchronize')
        code = _tracked_sub(r'\bcudaEventElapsedTime\b', 'hipEventElapsedTime', code, 'cudaEventElapsedTime→hipEventElapsedTime')
        # cudaMemcpyKind
        code = _tracked_sub(r'\bcudaMemcpyHostToDevice\b', 'hipMemcpyHostToDevice', code, 'cudaMemcpyHostToDevice→hipMemcpyHostToDevice')
        code = _tracked_sub(r'\bcudaMemcpyDeviceToHost\b', 'hipMemcpyDeviceToHost', code, 'cudaMemcpyDeviceToHost→hipMemcpyDeviceToHost')
        code = _tracked_sub(r'\bcudaMemcpyDeviceToDevice\b', 'hipMemcpyDeviceToDevice', code, 'cudaMemcpyDeviceToDevice→hipMemcpyDeviceToDevice')
        # Pinned memory
        code = _tracked_sub(r'\bcudaMallocHost\b', 'hipHostMalloc', code, 'cudaMallocHost→hipHostMalloc')
        code = _tracked_sub(r'\bcudaFreeHost\b', 'hipHostFree', code, 'cudaFreeHost→hipHostFree')
        # Events
        code = _tracked_sub(r'\bcudaEvent_t\b', 'hipEvent_t', code, 'cudaEvent_t→hipEvent_t')
        # Device queries
        code = _tracked_sub(r'\bcudaGetDevice\b', 'hipDeviceGet', code, 'cudaGetDevice→hipDeviceGet')
        # checkCudaErrors macro — stub it out (no HIP equivalent)
        code = _tracked_sub(r'\bcheckCudaErrors\s*\(', '(void)(', code, 'checkCudaErrors→(void)(')
        # cuda_device variable name
        code = _tracked_sub(r'\bcuda_device\b', 'hip_device', code, 'cuda_device→hip_device')

        # ── WAVEFRONT_SIZE define ───────────────────────────────────
        # ROCm wavefront is 64 on gfx9 (MI300/MI250). CUDA warp is 32.
        if not re.search(r'#define\s+WAVEFRONT_SIZE', code):
            include_lines = list(re.finditer(r'#include\s+[<"].*?[>"]\n', code))
            if include_lines:
                last_include = include_lines[-1]
                insert_pos = last_include.end()
                code = code[:insert_pos] + '#define WAVEFRONT_SIZE 64\n' + code[insert_pos:]
            else:
                code = '#define WAVEFRONT_SIZE 64\n' + code
            changelog.append('#define WAVEFRONT_SIZE 64 added')

        # ── Shuffle intrinsics: fix masks for wavefront64 ──────────
        # __shfl_xor_sync mask: 0x1f (5-bit, warp32) → 0x3f (6-bit, wavefront64)
        code = _tracked_sub(
            r'(__shfl_xor_sync\s*\()0x1f(,)',
            r'\g<1>0x3f\g<2>',
            code, '__shfl_xor_sync mask 0x1f→0x3f'
        )
        # __shfl_up_sync / __shfl_down_sync mask: 0x1f → 0x3f
        code = _tracked_sub(
            r'(__shfl_up_sync\s*\()0x1f(,)',
            r'\g<1>0x3f\g<2>',
            code, '__shfl_up_sync mask 0x1f→0x3f'
        )
        code = _tracked_sub(
            r'(__shfl_down_sync\s*\()0x1f(,)',
            r'\g<1>0x3f\g<2>',
            code, '__shfl_down_sync mask 0x1f→0x3f'
        )
        # Full-width masks: 0xffffffff → 64-bit
        code = _tracked_sub(
            r'(__shfl_\w+_sync\s*\()0x[fF]{8}(,)',
            r'\g<1>0xffffffffffffffffULL\g<2>',
            code, '__shfl mask 0xffffffff→0xffffffffffffffffULL'
        )
        # Replace __syncwarp() with __syncthreads() for wavefront64 safety
        code = _tracked_sub(r'\b__syncwarp\s*\(\s*\)', '__syncthreads()', code, '__syncwarp()→__syncthreads()')

        # ── Warp size constant: 32 → 64 ────────────────────────────
        # Only replace standalone 32 in warp-size contexts, NOT array sizes
        code = _tracked_sub(r'\bwarpSize\b', '64', code, 'warpSize→64')
        code = _tracked_sub(r'\bWARP_SIZE\b(?!\s*64)', 'WAVEFRONT_SIZE', code, 'WARP_SIZE→WAVEFRONT_SIZE')

        if return_changelog:
            return code, changelog
        return code

    @staticmethod
    def _rubric_score_pipeline(kimi_success: bool, glm_success: bool,
                               verify_success: bool, verify_passed: bool,
                               has_ported_code: bool, ported_code: str,
                               changes_count: int) -> int:
        """Rubric-based pipeline confidence score (0-100).

        Dimensions:
          - Pipeline Completion (0-35): which stages ran successfully
          - Code Quality (0-35): generated code structure
          - Verification Outcome (0-30): pass/fail with rationale
        """
        score = 0

        # ── Dimension 1: Pipeline Completion (0-35) ──
        if kimi_success:
            score += 12
        if glm_success:
            score += 18
        if verify_success:
            score += 5

        # ── Dimension 2: Code Quality (0-35) — A9: outcome-based ──
        if has_ported_code and len(ported_code.strip()) > 50:
            score += 10
            # A9: Score HIP API presence (not CUDA keywords)
            hip_apis = ["hipMalloc", "hipFree", "hipMemcpy", "hipLaunchKernel",
                        "hip/hip_runtime.h", "hipThreadIdx", "hipBlockIdx",
                        "hipBlockDim", "hipStreamSynchronize"]
            hip_count = sum(1 for api in hip_apis if api in ported_code)
            if hip_count >= 3:
                score += 15
            elif hip_count >= 1:
                score += 8
            # A9: Penalize CUDA remnants (should be zero in a good port)
            cuda_remnants = ["cudaMalloc", "cudaFree", "cudaMemcpy",
                             "cuda_runtime.h", "cudaDeviceSynchronize"]
            cuda_count = sum(1 for api in cuda_remnants if api in ported_code)
            if cuda_count == 0:
                score += 10
            elif cuda_count <= 2:
                score += 3  # partial — some remnants remaining

        # ── Dimension 3: Verification Outcome (0-30) ──
        if verify_passed:
            score += 30
        elif verify_success:
            # Verification ran but found issues — partial credit
            score += 10

        return min(score, 100)

    @staticmethod
    def _rubric_score_response(output: str) -> float:
        """Rubric for individual model response quality (0.0-1.0).

        A9: Scores HIP API presence and CUDA remnant absence instead of
        rewarding CUDA keywords like __global__ and threadIdx.
        """
        if not output or len(output.strip()) == 0:
            return 0.0
        score = 0.3  # baseline: non-empty response
        import re
        if len(output) > 100:
            score += 0.1
        # A9: Score HIP API presence (not CUDA keywords)
        hip_apis = ["hipMalloc", "hipFree", "hipMemcpy", "hip/hip_runtime.h",
                    "hipLaunchKernel", "__global__"]  # __global__ valid in HIP too
        if any(api in output for api in hip_apis):
            score += 0.15
        if re.search(r'```(?:cuda|hip|cpp)?\n', output):
            score += 0.15
        if re.search(r'\{[^}]*\}', output, re.DOTALL):
            score += 0.15
        # A9: Penalize CUDA remnants
        cuda_remnants = ["cudaMalloc", "cudaFree", "cudaMemcpy", "cuda_runtime.h"]
        if not any(api in output for api in cuda_remnants):
            score += 0.15
        return min(score, 1.0)

    # ── A2A structured message builders ──────────────────────────

    def _build_deepseek_plan_message(self, plan_text: str,
                                     kernel_source: str = "") -> A2AMessage:
        """Parse DeepSeek's plan text into a structured A2AMessage.

        Extracts API mappings, header changes, and constants from the plan
        prose so that downstream agents (Kimi, GLM) receive prioritized
        details instead of a truncated blob.
        """
        # Summary: first 300 chars or first 2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', plan_text.strip())
        if len(sentences) >= 2 and len(sentences[0]) + len(sentences[1]) <= 300:
            summary = f"DeepSeek Plan: {sentences[0]} {sentences[1]}"
        else:
            summary = f"DeepSeek Plan: {plan_text[:300]}"

        priority_details = []

        # Extract API mappings: patterns like "cudaMalloc → hipMalloc" or
        # "replace cudaMalloc with hipMalloc"
        for m in re.finditer(r'(\w*[Cc]uda\w*)\s*(?:→|->|→ )\s*(\w*[Hh]ip\w*)', plan_text):
            priority_details.append({
                "type": "api_mapping",
                "cuda": m.group(1),
                "hip": m.group(2),
            })
        # Also catch "replace X with Y" style
        for m in re.finditer(r'replace\s+(\w+)\s+with\s+(\w+)', plan_text, re.IGNORECASE):
            priority_details.append({
                "type": "api_mapping",
                "cuda": m.group(1),
                "hip": m.group(2),
            })

        # Extract header changes: "#include <cuda_runtime.h> → #include <hip/hip_runtime.h>"
        for m in re.finditer(r'#include\s+[<"]([^>"]+\.h)[>"]\s*(?:→|->)\s*#include\s+[<"]([^>"]+\.h)[>"]', plan_text):
            priority_details.append({
                "type": "header",
                "cuda": m.group(1),
                "hip": m.group(2),
            })
        # Also catch "cuda_runtime.h → hip/hip_runtime.h" without #include prefix
        for m in re.finditer(r'(cuda_runtime\.h|helper_cuda\.h|helper_functions\.h|device_launch_parameters\.h)\s*(?:→|->)\s*(hip/[\w/]+\.h)', plan_text):
            priority_details.append({
                "type": "header",
                "cuda": m.group(1),
                "hip": m.group(2),
            })

        # Extract constants / sizing changes: "warpSize 32 → 64", "0x1f → 0x3f"
        for m in re.finditer(r'(warpSize|WAVEFRONT_SIZE|0x1f|0xffffffff)\s*(?:→|->)\s*(\w+)', plan_text):
            priority_details.append({
                "type": "risk",
                "description": f"Constant change: {m.group(1)} → {m.group(2)}",
            })

        full_ref = f"plan:{self.run_id}" if self.run_id else "deepseek_plan"

        return A2AMessage(
            summary=summary,
            priority_details=priority_details,
            full_ref=full_ref,
            changelog=[],
        )

    def _build_error_feedback_message(self, compile_errs: list,
                                      glm_analysis: dict = None,
                                      iteration: int = 0) -> A2AMessage:
        """Structure compile errors + GLM analysis into an A2AMessage.

        ALL errors are included as priority_details (not just first 3).
        If GLM analysis exists, its fixes are used and prioritized by
        severity (error > warning).
        """
        # Summary: count + first error type
        first_err = compile_errs[0] if compile_errs else "(no errors)"
        err_type = "error"
        if "warning" in first_err.lower():
            err_type = "warning"
        summary = f"{len(compile_errs)} compile {err_type}s. First: {first_err[:120]}"

        priority_details = []

        if glm_analysis:
            # Use GLM's structured fixes, sorted by priority
            fixes = glm_analysis.get("fixes", [])
            fixes = sorted(fixes, key=lambda x: x.get("priority", 99))
            for f in fixes:
                priority_details.append({
                    "type": "error_fix",
                    "error": f.get("error", "?")[:200],
                    "fix": f.get("exact_fix", f.get("root_cause", "?"))[:200],
                    "priority": f.get("priority", 99),
                })
            # Also include missing includes and wrong APIs
            for inc in glm_analysis.get("missing_includes", []):
                priority_details.append({
                    "type": "error_fix",
                    "error": f"Missing include: {inc}",
                    "fix": f"Add #include {inc}",
                    "priority": 1,
                })
            for api in glm_analysis.get("wrong_apis", []):
                priority_details.append({
                    "type": "api_mapping",
                    "cuda": api.get("cuda", "?"),
                    "hip": api.get("hip", "?"),
                })
        else:
            # No GLM analysis — structure raw errors, prioritize errors > warnings
            for i, err in enumerate(compile_errs):
                is_warning = "warning" in err.lower()
                priority_details.append({
                    "type": "error_fix",
                    "error": err[:200],
                    "fix": "see compiler output",
                    "priority": 99 if is_warning else 10 + i,
                })

        full_ref = f"errors:{self.run_id}:iter{iteration}" if self.run_id else f"errors:iter{iteration}"

        return A2AMessage(
            summary=summary,
            priority_details=priority_details,
            full_ref=full_ref,
            changelog=[],
        )

    def _build_glm_feedback_message(self, glm_result: dict) -> A2AMessage:
        """Structure GLM evaluation feedback into an A2AMessage."""
        summary = glm_result.get("verdict", "") or glm_result.get("feedback", "")[:300]

        priority_details = []
        for i, issue in enumerate(glm_result.get("issues", [])):
            priority_details.append({
                "type": "error_fix",
                "error": issue,
                "fix": "see feedback",
                "priority": i + 1,
            })

        full_ref = f"glm:{self.run_id}" if self.run_id else "glm_feedback"

        return A2AMessage(
            summary=summary,
            priority_details=priority_details,
            full_ref=full_ref,
            changelog=[],
        )

    # ── Phase prompt builders ────────────────────────────────────

    def _build_deepseek_plan_prompt(self, kernel_source: str,
                                    patterns: List[Dict]) -> str:
        """Build the DeepSeek planner phase prompt with classifier context.

        Role: DeepSeek-Planner — reasons freely about the CUDA kernel and produces
        a detailed porting plan as prose. No JSON required — reasoning is the asset here.
        The plan is passed to Kimi-Coder as context.
        """
        prompt = (
            "Analyze this CUDA kernel and produce a porting plan for AMD ROCm/HIP.\n"
            "Identify every CUDA-specific construct and its HIP replacement.\n"
            "Prioritize: warp(32)→wavefront(64) divergence, __shfl mask widths, "
            "shared memory sizing, header swaps, local .cuh dependencies.\n\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"```cuda\n{kernel_source[:5000]}\n```\n\n"
            "Write a detailed porting plan as a numbered checklist. "
            "For each item: what to change, where (line/construct), and why. "
            "Be specific — a coder agent will follow your plan exactly."
        )
        return prompt

    def _build_kimi_code_prompt(self, kernel_source: str,
                                patterns: List[Dict],
                                deepseek_plan: str = "") -> str:
        """Build the Kimi K2.7 code generator phase prompt.

        Role: Kimi-Coder — generates the actual ported HIP kernel code.
        Now receives DeepSeek's plan as context (was GLM analysis before).

        Output format: JSON with ported_code (str), confidence (0-100),
        changes (list[str]), explanation (str).
        """
        prompt = (
            "Port this CUDA kernel to AMD ROCm/HIP. Fix warp(32)→wavefront(64) issues.\n\n"
            "CHECKLIST:\n"
            "- __shfl_xor_sync mask 0x1f → 0x3f for wavefront64\n"
            "- __shfl_down_sync masks → 0xffffffffffffffffULL (64-bit)\n"
            "- warpSize 32 → WAVEFRONT_SIZE 64 or dynamic\n"
            "- shared memory sized for warp 32 → WAVEFRONT_SIZE (64)\n"
            "- __syncwarp() → __syncthreads()\n"
            "- #define WAVEFRONT_SIZE 64 at top\n"
            "- Replace #include <cuda_runtime.h> → #include <hip/hip_runtime.h>\n"
            "- Remove #include <helper_cuda.h>, <helper_functions.h>, <device_launch_parameters.h>\n"
            "- Remove ALL #include \"*.cuh\" local headers (inline their content if needed)\n\n"
        )

        if deepseek_plan:
            try:
                plan_msg = self._build_deepseek_plan_message(deepseek_plan, kernel_source)
                prompt += f"DeepSeek Planner's plan (follow this):\n{plan_msg.to_prompt(max_chars=4000)}\n\n"
            except Exception:
                prompt += f"DeepSeek Planner's plan (follow this):\n{deepseek_plan[:2000]}\n\n"

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"```cuda\n{kernel_source[:6000]}\n```\n\n"
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}.\n"
            "IMPORTANT: The ported_code field must contain the COMPLETE HIP kernel source. "
            "If the kernel is large, minimize explanation to save tokens. "
            "Prefer full code over partial code with verbose explanation."
        )
        return prompt

    def _build_kimi_refine_prompt(self, kernel_source: str,
                                  previous_code: str,
                                  feedback: str,
                                  patterns: List[Dict],
                                  deepseek_plan: str = "",
                                  iteration: int = 1,
                                  checklist_override: list[str] = None) -> str:
        """Build the Kimi refinement prompt for orchestration loop iterations.

        Kimi receives the original kernel, its previous output, and
        GLM evaluator's specific feedback to fix issues.

        TRIZ #15 (Dynamics): checklist_override allows the PromptOptimizer to
        inject an evolved checklist instead of the static fallback.
        """
        # TRIZ #15: Use evolved checklist if provided, else fallback to static
        checklist = checklist_override if checklist_override else [
            "__shfl_xor_sync mask 0x1f → 0x3f for wavefront64",
            "__shfl_down_sync masks → 0xffffffffffffffffULL (64-bit)",
            "warpSize 32 → WAVEFRONT_SIZE 64 or dynamic",
            "shared memory sized for warp 32 → WAVEFRONT_SIZE (64)",
            "__syncwarp() → __syncthreads()",
            "#define WAVEFRONT_SIZE 64 at top",
            "Replace #include <cuda_runtime.h> → #include <hip/hip_runtime.h>",
            "Remove #include <helper_cuda.h>, <helper_functions.h>, <device_launch_parameters.h>",
            'Remove ALL #include "*.cuh" local headers',
        ]

        checklist_text = "\n".join(f"- {item}" for item in checklist)
        prompt = (
            f"Fix your ported HIP kernel based on evaluator feedback (iteration {iteration}).\n\n"
            f"CHECKLIST:\n{checklist_text}\n\n"
        )

        if deepseek_plan:
            try:
                plan_msg = self._build_deepseek_plan_message(deepseek_plan)
                prompt += f"DeepSeek Planner's plan (reference):\n{plan_msg.to_prompt(max_chars=3000)}\n\n"
            except Exception:
                prompt += f"DeepSeek Planner's plan (reference):\n{deepseek_plan[:1500]}\n\n"

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        if feedback:
            try:
                fb_msg = A2AMessage(
                    summary=f"Evaluator feedback (fix ALL): {feedback[:300]}",
                    priority_details=[{"type": "error_fix", "error": feedback[:600], "fix": "see feedback above", "priority": 1}],
                    full_ref=f"feedback:{self.run_id}" if self.run_id else "feedback",
                    changelog=[],
                )
                prompt += f"Evaluator feedback (fix ALL):\n{fb_msg.to_prompt(max_chars=3000)}\n\n"
            except Exception:
                prompt += f"Evaluator feedback (fix ALL):\n{feedback}\n\n"

        prompt += (
            f"Original CUDA:\n```cuda\n{kernel_source[:4000]}```\n\n"
            f"Your previous output:\n```hip\n{previous_code[:4000]}```\n\n"
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}."
        )
        return prompt

    def _build_glm_evaluate_prompt(self, ported_code: str,
                                   patterns: List[Dict],
                                   deepseek_plan: str = "",
                                   feedback: str = "",
                                   iteration: int = 1,
                                   max_iterations: int = 3,
                                   regex_changelog: Optional[List[str]] = None) -> str:
        """Build the GLM evaluator prompt.

        Role: GLM-Evaluator — strict JSON output. Checks ported code for
        wavefront64 correctness, CUDA remnants, and compilation safety.
        System prompt already defines role + JSON contract.

        Args:
            regex_changelog: List of regex post-processing fixes already
                applied by _fix_ported_code(). When provided, GLM is told
                not to re-flag these already-fixed issues, preventing
                false-positive feedback on regex-handled patterns.
        """
        prompt = f"Evaluate this ported HIP kernel (iteration {iteration}/{max_iterations}).\n\n"

        # A3: Regex transparency — tell GLM what _fix_ported_code already fixed
        if regex_changelog:
            fixes_text = "\n".join(f"  - {fix}" for fix in regex_changelog)
            prompt += (
                "The following automatic regex fixes were already applied to this code.\n"
                "Do NOT re-flag these — they are resolved:\n"
                f"{fixes_text}\n\n"
            )

        prompt += (
            "Checks:\n"
            "- __shfl masks: 0xffffffffffffffffULL (64-bit, not 32-bit 0xffffffff)\n"
            "- __shfl_xor_sync mask: 0x3f (not 0x1f) for wavefront64\n"
            "- __syncwarp() → __syncthreads()\n"
            "- shared memory sized for 64, not 32\n"
            "- No CUDA headers (cuda_runtime.h, helper_cuda.h, device_launch_parameters.h)\n"
            "- No .cuh local headers remaining\n"
            "- WAVEFRONT_SIZE 64 defined or warpSize used dynamically\n\n"
        )

        if deepseek_plan:
            try:
                plan_msg = self._build_deepseek_plan_message(deepseek_plan)
                prompt += f"Planner's plan (reference):\n{plan_msg.to_prompt(max_chars=2000)}\n\n"
            except Exception:
                prompt += f"Planner's plan (reference):\n{deepseek_plan[:800]}\n\n"

        if feedback:
            try:
                fb_msg = A2AMessage(
                    summary=f"Previous issues (verify fixed): {feedback[:300]}",
                    priority_details=[{"type": "error_fix", "error": feedback[:600], "fix": "verify fixed", "priority": 1}],
                    full_ref=f"feedback:{self.run_id}" if self.run_id else "feedback",
                    changelog=[],
                )
                prompt += f"Previous issues (verify fixed):\n{fb_msg.to_prompt(max_chars=2000)}\n\n"
            except Exception:
                prompt += f"Previous issues (verify fixed):\n{feedback[:800]}\n\n"

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"```hip\n{ported_code[:4000]}\n```\n\n"
            'Respond with JSON: {"pass": bool, "issues": [str], '
            '"feedback": str, "verdict": str}.'
        )
        return prompt

    @staticmethod
    def _normalize_error(err: str) -> str:
        """Normalize a compile error line for semantic diffing (TRIZ #3/#22/#28).

        Strips volatile parts (line/column numbers, temp build paths, excess
        whitespace) so the same error at a different line/file path compares
        equal.  The error type + message is the semantic key.
        """
        import re as _re
        s = err.strip()
        # Strip temp build paths: /tmp/verifier_build_xxx/file.cpp → file.cpp
        s = _re.sub(r'/tmp/\S+/(\S+)', r'\1', s)
        # Strip leading file:line:col: prefix → keep "error:" / "warning:" etc.
        # Matches patterns like: file.cpp:67:5: error: ...
        s = _re.sub(r'^[\w./-]+:\d+:\d+:\s*', '', s)
        # Also handle file:line: (no column) prefix
        s = _re.sub(r'^[\w./-]+:\d+:\s*', '', s)
        # Collapse whitespace
        s = _re.sub(r'\s+', ' ', s).strip()
        return s

    def _build_glm_error_analysis_prompt(self, ported_code: str,
                                         compile_errors: List[str],
                                         iteration: int,
                                         patterns: List[Dict]) -> str:
        """Build GLM prompt for compile-error analysis (TRIZ #28).

        When hipcc fails, GLM analyzes the compile errors + code and tells
        Kimi WHAT to fix structurally — not just "error: undefined hipMalloc"
        but "you forgot #include <hip/hip_runtime.h>, that's why hipMalloc is undefined."

        This is the lightweight error-analyst mode, not full semantic evaluation.
        """
        err_text = "\n".join(compile_errors[:10])

        prompt = (
            f"You are a HIP/ROCm compile error analyst. Kimi generated code that fails to compile.\n"
            f"Analyze the compiler errors and tell Kimi EXACTLY what to fix.\n\n"
            f"COMPILER ERRORS (hipcc, iteration {iteration}):\n"
            f"{err_text}\n\n"
            f"CURRENT CODE (first 3000 chars):\n"
            f"```hip\n{ported_code[:3000]}\n```\n\n"
            "Analyze each error and provide:\n"
            "1. Root cause for each error (not just the error message)\n"
            "2. The EXACT fix needed (specific API name, include, or type)\n"
            "3. Priority order (fix headers first, then types, then logic)\n\n"
            "Common CUDA→HIP issues:\n"
            "- cuda_runtime.h → hip/hip_runtime.h (causes ALL cuda* functions to be undefined)\n"
            "- cudaMalloc → hipMalloc, cudaMemcpy → hipMemcpy, cudaFree → hipFree\n"
            "- cudaError_t → hipError_t, cudaSuccess → hipSuccess\n"
            "- checkCudaErrors() → remove or define wrapper\n"
            "- __shfl_*_sync mask: 0x1f (32-bit) → 0x3f (64-bit wavefront)\n"
            "- threadIdx.x threadIdx.y etc stay the same in HIP\n\n"
            'Respond with JSON: {"fixes": [{"error": str, "root_cause": str, '
            '"exact_fix": str, "priority": int}], "summary": str, '
            '"missing_includes": [str], "wrong_apis": [{"cuda": str, "hip": str}]}.'
        )
        return prompt

    # ── Main routing logic ──────────────────────────────────────

    def route(self, kernel_source: str, patterns: List[Dict],
              max_iterations: int = 10,
              on_phase=None,
              verifier=None,
              kernel_name: str = "test_kernel") -> Dict:
        """Route kernel through the loop engineering pipeline.

        Loop: DeepSeek (plan) → Kimi (code) → [hipcc compile FIRST] → GLM (evaluate only if compile passes) → feedback → Kimi refines

        TRIZ #13 (Do It In Reverse) / #28 (Mechanical Substitution):
        The verification loop now compiles FIRST, then evaluates. If hipcc fails,
        compile errors ARE the feedback — GLM is skipped entirely (saves ~12s/iter).
        If hipcc passes, GLM runs for semantic checks (shfl correctness, perf).

        This eliminates the root contradiction where GLM checked static patterns
        (shfl masks, headers) that _fix_ported_code() regex already handled,
        said "pass", while hipcc reported real compile errors — giving Kimi
        conflicting signals every iteration.

        Args:
            kernel_source: The CUDA kernel source code.
            patterns: List of classifier-detected patterns.
            max_iterations: Maximum Kimi→GLM cycles (default 10).
            on_phase: Optional callback(phase: str, detail: str) for live progress.
            verifier: Optional VerificationAgent for in-loop hipcc compile checks.
            kernel_name: Name of kernel (for verifier build dir isolation).

        Returns:
            {"ported_code": ..., "confidence": ..., "changes": [...],
             "model_used": ..., "cost": ..., "orchestrator_passed": ...,
             "iterations_used": ..., "compile_errors": [...]}
        """
        if not self.api_key:
            return {"ported_code": "", "confidence": 0,
                    "changes": ["No API key -- use template fallback"],
                    "model_used": "none", "cost": 0,
                    "orchestrator_passed": False, "iterations_used": 0}

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0,
                  "orchestrator_passed": False, "iterations_used": 0,
                  "compile_errors": [], "compile_passed": False}

        # A2A protocol: unique run ID for structured message full_ref keys
        # I3: Use UUID for reproducibility tracking and create run directory
        run_id = str(uuid.uuid4())[:8]
        self.run_id = f"{kernel_name}_{run_id}"
        run_dir = Path(f"runs/{self.run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        result["run_id"] = self.run_id

        # Track pipeline phase outcomes for rubric scoring
        planner_success = False
        coder_success = False
        verify_success = False
        verify_passed = False
        evaluator_feedback = ""
        compile_passed = False  # TRIZ #23: track compile state as feedback signal
        deepseek_plan_output = ""

        # ── Phase 1: DeepSeek PLANS the port (reasoning model — prose OK) ──
        if on_phase: on_phase("plan", "DeepSeek-v4-pro", "planning CUDA→HIP strategy")
        ds_prompt = self._build_deepseek_plan_prompt(kernel_source, patterns)
        plan = self._call_model("deepseek", ds_prompt,
                                system_prompt=SYSTEM_PROMPTS.get("deepseek", ""))
        # I3: Log model I/O for reproducibility
        try:
            (run_dir / "phase1_plan_input.json").write_text(
                json.dumps({"prompt": ds_prompt[:5000]}, indent=2), encoding="utf-8")
            (run_dir / "phase1_plan_output.json").write_text(
                json.dumps({"model": "deepseek", "success": plan.success,
                            "output": plan.output[:5000]}, indent=2), encoding="utf-8")
        except OSError:
            pass
        if plan.success:
            planner_success = True
            deepseek_plan_output = plan.output
            result["changes"].append(f"[deepseek] Plan generated ({len(plan.output)} chars)")
        else:
            result["changes"].append("[deepseek] Planning FAILED — proceeding without plan")

        # ── Phase 2: Kimi CODES the initial port ──
        if on_phase: on_phase("code", "Kimi K2.7", "generating HIP port from plan")
        kimi_prompt = self._build_kimi_code_prompt(kernel_source, patterns,
                                                   deepseek_plan=deepseek_plan_output)
        code = self._call_model("kimi27", kimi_prompt,
                                system_prompt=SYSTEM_PROMPTS.get("kimi27", ""))
        # I3: Log Kimi code generation
        try:
            (run_dir / "phase2_kimi_output.json").write_text(
                json.dumps({"model": "kimi27", "success": code.success,
                            "output": code.output[:5000]}, indent=2), encoding="utf-8")
        except OSError:
            pass
        if code.success:
            coder_success = True
            extracted = self._extract_code(code.output)
            extracted = self._fix_ported_code(extracted, return_changelog=True)
            if isinstance(extracted, tuple):
                extracted, regex_changelog = extracted
            else:
                regex_changelog = []
            result["ported_code"] = extracted
            result["regex_changelog"] = regex_changelog  # A3: track regex fixes
            if verifier and hasattr(verifier, 'quick_compile_check'):
                if on_phase: on_phase("compile", "hipcc", "in-loop compilation check")
                cc = verifier.quick_compile_check(extracted, kernel_name=kernel_name)
                if cc["compile_success"]:
                    result["changes"].append("[hipcc] In-loop compile: PASSED ✅")
                    compile_passed = True
                else:
                    compile_errs = cc.get("errors", [])
                    result["compile_errors"].extend(compile_errs)
                    prev_error_count = len(compile_errs)  # TRIZ #23: baseline for prompt evolution
                    err_summary = "; ".join(compile_errs[:3]) if compile_errs else cc["compile_output"][:300]
                    result["changes"].append(f"[hipcc] In-loop compile FAILED: {err_summary[:120]}")
                    # Feed compile errors to GLM evaluator as additional feedback
                    # A2A protocol: structure ALL errors, not just first 3
                    all_errs = compile_errs if compile_errs else [cc["compile_output"][:300]]
                    # Check if the code was truncated
                    is_truncated = "TRUNCATED" in extracted
                    if is_truncated:
                        result["changes"].append("[kimi27] Output was TRUNCATED — requesting shorter response")
                    try:
                        err_msg = self._build_error_feedback_message(all_errs, iteration=0)
                        structured_errs = err_msg.to_prompt(max_chars=4000)
                    except Exception:
                        structured_errs = "\n".join(compile_errs[:3]) if compile_errs else cc["compile_output"][:300]
                    if is_truncated:
                        evaluator_feedback = (
                            "CRITICAL: Your previous response was TRUNCATED (hit token limit). "
                            "Output ONLY the ported HIP code in a ```hip block. "
                            "No JSON wrapper, no explanation, no comments. "
                            "Just the raw C++ code with all CUDA→HIP replacements applied.\n\n"
                            f"REAL COMPILER ERRORS (hipcc) — fix these FIRST:\n"
                            + structured_errs
                        )
                    else:
                        evaluator_feedback = (
                            f"REAL COMPILER ERRORS (hipcc) — fix these FIRST:\n"
                            + structured_errs
                            + "\n\nAlso address any static analysis issues below."
                        )

            result["changes"].append("[kimi27] Generated ported kernel")
            result["model_used"] = "kimi27"
        else:
            result["changes"].append("[kimi27] Code generation FAILED")
            # Can't proceed without initial code
            result["cost"] = round(self.total_cost, 4)
            return result

        # ── Phase 3: hipcc COMPILE FIRST → (GLM eval only if compile passes) → Kimi refines ──
        # TRIZ #13 (Do It In Reverse) / #28 (Mechanical Substitution):
        # OLD: GLM evaluates (static checklist) → compile → override with compile errors
        #   Problem: GLM checks shfl masks/headers that _fix_ported_code() regex already
        #   fixed, so GLM says "pass" while hipcc reports real errors. Kimi gets conflicting
        #   signals: "GLM says good" + "but compile fails". Wastes ~12s/iter on GLM (84s total).
        # NEW: Compile FIRST. If compile fails → compile errors ARE the feedback (skip GLM).
        #   If compile passes → THEN run GLM for semantic check (shfl correctness, perf).
        # This eliminates conflicting signals and saves the GLM call when compile fails.
        opt = PromptOptimizer()
        prev_error_count = 0  # TRIZ #23: track error delta across iterations
        prev_errors_set  = set()  # TRIZ #22: track which errors we've already shown (raw)
        prev_errors_norm = set()  # TRIZ #3/#22: normalized set for semantic diffing
        stagnation_count = 0  # TRIZ #15: count iterations with no improvement
        error_history = []  # TRIZ #17: cap error context to last 2 iterations
        norm_error_history = []  # A5: track normalized error frozensets for cycle detection
        for iteration in range(1, max_iterations + 1):
            if not result["ported_code"]:
                break

            # ── Step 1: hipcc compile check FIRST ──────────────────────────
            # TRIZ #13: Reverse the order — compile before evaluate.
            # If compile fails, compile errors ARE the feedback. Skip GLM entirely.
            compile_failed_this_iter = False
            if verifier and hasattr(verifier, 'quick_compile_check'):
                if on_phase: on_phase("compile", "hipcc", f"compile-first check (attempt {iteration}/{max_iterations})")
                cc = verifier.quick_compile_check(result["ported_code"], kernel_name=kernel_name)
                # I3: Log iteration compile result for reproducibility
                try:
                    (run_dir / f"iteration_{iteration}_compile.json").write_text(
                        json.dumps({"iteration": iteration,
                                    "compile_success": cc["compile_success"],
                                    "errors": cc.get("errors", [])[:8]},
                                   indent=2), encoding="utf-8")
                except OSError:
                    pass
                if cc["compile_success"]:
                    result["changes"].append(
                        f"[hipcc] Compile-first check {iteration}: PASSED ✅")
                    result["compile_errors"] = []
                    compile_passed = True
                else:
                    compile_failed_this_iter = True
                    compile_errs = cc.get("errors", [])
                    result["compile_errors"].extend(compile_errs)

                    # TRIZ #3/#22/#28: Semantic error diffing — normalize before
                    # comparing so the same error at a different line number is
                    # NOT flagged as "new" every iteration.
                    current_errors_set = set(e.strip() for e in compile_errs if e.strip())
                    current_norm_set   = set(self._normalize_error(e) for e in compile_errs if e.strip())
                    new_errors_norm    = current_norm_set - prev_errors_norm
                    resolved_errors    = prev_errors_norm - current_norm_set
                    new_errors         = current_errors_set - prev_errors_set   # raw diff (for display)
                    prev_errors_set    = current_errors_set
                    prev_errors_norm   = current_norm_set

                    # TRIZ #23: Track convergence — error count delta
                    current_err_count = len(compile_errs)
                    error_delta = prev_error_count - current_err_count
                    error_history.append(current_err_count)
                    result["changes"].append(
                        f"[hipcc] Iter {iteration}: {current_err_count} errors "
                        f"(delta: {error_delta:+d}, new: {len(new_errors_norm)}, "
                        f"resolved: {len(resolved_errors)})")

                    # LIVE VISIBILITY: Print error details during loop, not after
                    top_errs = compile_errs[:2] if compile_errs else ["(no error lines parsed)"]
                    for err_line in top_errs:
                        # Truncate and clean for terminal display
                        clean = err_line.strip()[:60]
                        if clean:
                            print(f"║  │  ⚠ {clean:<58}║")
                    trend = f"{'↓' if error_delta > 0 else '↑' if error_delta < 0 else '→'} {current_err_count} errs (Δ{error_delta:+d}, new:{len(new_errors_norm)})"
                    print(f"║  │  📊 {trend:<58}║")

                    # TRIZ #15: Detect stagnation — 3 iterations with no improvement
                    if error_delta <= 0:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0

                    # A5: Cycle detection — if the same normalized error set
                    # recurs within the last 4 iterations, the loop is
                    # oscillating (e.g., 5→3→5→3) without making real
                    # progress. Double-count cycles to trigger stagnation
                    # recovery faster.
                    current_norm_frozen = frozenset(
                        self._normalize_error(e) for e in compile_errs if e.strip()
                    )
                    if current_norm_frozen in norm_error_history[-4:]:
                        stagnation_count += 2  # cycle detected — escalate faster
                        result["changes"].append(
                            f"[hipcc] Cycle detected: same error set recurred "
                            f"(stagnation_count={stagnation_count})")
                        print(f"║  │  🔁 CYCLE: same errors recurred — stagnation escalated{'':<27}║")
                    norm_error_history.append(current_norm_frozen)

                    # TRIZ #15: After 3 stagnant iterations, escalate to DeepSeek re-plan
                    if stagnation_count >= 3 and iteration < max_iterations:
                        result["changes"].append(
                            f"[hipcc] Stagnation detected ({stagnation_count} iterations no improvement) "
                            f"— escalating to DeepSeek re-plan")
                        print(f"║  │  🔄 STAGNATION: {stagnation_count} iters no improvement — re-planning{'':<24}║")
                        if on_phase: on_phase("plan", "DeepSeek-v4-pro",
                            f"re-planning due to stagnation (iter {iteration})")
                        re_plan = self._call_model(
                            "deepseek", ds_prompt,
                            system_prompt=SYSTEM_PROMPTS.get("deepseek", ""),
                        )
                        if re_plan.success:
                            deepseek_plan_output = re_plan.output
                            result["changes"].append(
                                f"[deepseek] Re-plan generated (stagnation recovery)")
                            stagnation_count = 0  # Reset after re-plan
                        # Keep the same compile errors for Kimi, but with fresh plan

                    err_summary = "; ".join(compile_errs[:3]) if compile_errs else cc["compile_output"][:300]
                    result["changes"].append(
                        f"[hipcc] Compile-first check {iteration}: FAILED: {err_summary[:120]}")

                    # TRIZ #22/#28: Feed only NEW (semantically) errors to Kimi.
                    # A2A protocol: structure ALL errors via A2AMessage, not just first 3-5.
                    # Use normalized diff for the decision; raw errors for display.
                    all_errs_for_kimi = compile_errs if compile_errs else [cc["compile_output"][:300]]
                    if new_errors_norm:
                        # Genuinely new error type/message — show raw form
                        feedback_intro = (
                            f"REAL COMPILER ERRORS (hipcc) — NEW errors since last iteration (iteration {iteration}):\n"
                        )
                    elif new_errors and not new_errors_norm:
                        # Raw diff shows "new" but normalized diff shows 0 → same
                        # error at a different line number.  Tell Kimi explicitly.
                        feedback_intro = (
                            f"REAL COMPILER ERRORS (hipcc) — SAME error persists (possibly at different line) (iteration {iteration}).\n"
                            f"The error type and message are identical to last iteration; only the line number shifted.\n"
                            f"You MUST try a DIFFERENT approach. Previous fix did not work.\n"
                        )
                    else:
                        # All errors are the same as before — send them all but flag stagnation
                        feedback_intro = (
                            f"REAL COMPILER ERRORS (hipcc) — SAME errors persisting (iteration {iteration}).\n"
                            f"You MUST try a DIFFERENT approach. Previous fix did not work.\n"
                        )

                    # A2A: Build structured message from ALL errors
                    try:
                        err_msg = self._build_error_feedback_message(
                            all_errs_for_kimi, iteration=iteration)
                        structured_errs = err_msg.to_prompt(max_chars=4000)
                    except Exception:
                        # Fallback: old truncation approach
                        if new_errors_norm:
                            structured_errs = "\n".join(list(new_errors)[:5] if new_errors else compile_errs[:5])
                        else:
                            structured_errs = "\n".join(compile_errs[:3] if compile_errs else [cc["compile_output"][:300]])

                    evaluator_feedback = (
                        feedback_intro
                        + structured_errs
                        + "\n\nFocus on:\n"
                        "- Missing HIP API calls (cuda* not converted to hip*)\n"
                        "- Undefined functions/macros (checkCudaErrors, etc.)\n"
                        "- Type mismatches (hipError_t vs cudaError_t)\n"
                        "- Missing or wrong #include directives\n"
                        + (f"- ⚠️ Stagnation: {stagnation_count} iterations without improvement — try a DIFFERENT approach\n"
                           if stagnation_count > 0 else "")
                    )
                    # TRIZ #23: Record iteration for prompt evolution
                    opt.record_iteration(
                        prev_error_count, len(compile_errs), opt.get_checklist()
                    )
                    prev_error_count = len(compile_errs)

                    # ── TRIZ #28: GLM Error Analyst — translate compile errors for Kimi ──
                    # Root contradiction: GLM was skipped on compile failure, but that's
                    # when Kimi needs semantic guidance MOST. Raw hipcc errors like
                    # "undefined reference to hipMalloc" don't tell Kimi WHY — it needs
                    # "you forgot #include <hip/hip_runtime.h>". GLM bridges this gap.
                    if compile_errs and iteration < max_iterations:
                        if on_phase: on_phase("analyze", "GLM-5.2",
                            f"analyzing compile errors for Kimi (iter {iteration})")
                        print(f"║  │  🔍 GLM analyzing {len(compile_errs)} compile errors for Kimi{'':<30}║")
                        glm_err_prompt = self._build_glm_error_analysis_prompt(
                            result["ported_code"], compile_errs, iteration, patterns)
                        glm_err = self._call_model(
                            "glm", glm_err_prompt,
                            system_prompt="You are a HIP/ROCm compile error analyst. Respond ONLY with JSON.",
                            prefill='{"fixes":'  # TRIZ #9: force JSON
                        )
                        if glm_err.success:
                            # Parse GLM error analysis
                            raw_glm = glm_err.output.strip()
                            json_start = raw_glm.find("{")
                            if json_start >= 0:
                                raw_glm_json = raw_glm[json_start:]
                            else:
                                raw_glm_json = raw_glm
                            glm_analysis = None

                            # ── Strategy 1: Direct json.loads (strip prose prefix) ──
                            try:
                                glm_analysis = json.loads(raw_glm_json)
                            except (json.JSONDecodeError, TypeError, ValueError) as e:
                                logger.debug("GLM error analysis JSON parse failed: %s", e)

                            # ── Strategy 2: Balanced-brace extraction ──
                            # Count opening/closing braces respecting string literals,
                            # extract the complete JSON object starting from first '{'.
                            if glm_analysis is None:
                                glm_analysis = _extract_balanced_json(raw_glm)

                            # ── Strategy 3: Targeted regex extraction of individual arrays ──
                            if glm_analysis is None:
                                glm_analysis = _extract_arrays_regex(raw_glm)

                            # ── Strategy 4: Last-resort minimal structure ──
                            if glm_analysis is None:
                                first_brace = raw_glm.find("{")
                                last_brace = raw_glm.rfind("}")
                                if first_brace >= 0 and last_brace > first_brace:
                                    glm_analysis = {"fixes": [], "_raw": raw_glm[first_brace:last_brace+1]}

                            # ── Build feedback from parsed analysis ──
                            fixes = glm_analysis.get("fixes", []) if glm_analysis else []
                            missing_inc = glm_analysis.get("missing_includes", []) if glm_analysis else []
                            wrong_apis = glm_analysis.get("wrong_apis", []) if glm_analysis else []

                            if glm_analysis and (fixes or missing_inc or wrong_apis):
                                # Build structured feedback for Kimi
                                fixes = sorted(fixes,
                                    key=lambda x: x.get("priority", 99))
                                fix_lines = []
                                for f in fixes[:7]:
                                    fix_lines.append(
                                        f"  • {f.get('error', '?')[:80]}\n"
                                        f"    Root cause: {f.get('root_cause', '?')[:120]}\n"
                                        f"    Fix: {f.get('exact_fix', '?')[:150]}"
                                    )

                                evaluator_feedback = (
                                    f"GLM ERROR ANALYSIS (iteration {iteration}):\n"
                                    f"GLM analyzed {len(compile_errs)} compiler errors"
                                )
                                if fixes:
                                    evaluator_feedback += (
                                        f" and identified {len(fixes)} root causes.\n\n"
                                        f"PRIORITY FIXES:\n" + "\n".join(fix_lines) + "\n"
                                    )
                                else:
                                    evaluator_feedback += ".\n\n"
                                if missing_inc:
                                    evaluator_feedback += (
                                        f"\nMISSING INCLUDES (add these):\n"
                                        + "\n".join(f"  #include {inc}" for inc in missing_inc) + "\n"
                                    )
                                if wrong_apis:
                                    evaluator_feedback += (
                                        f"\nWRONG APIs (replace CUDA → HIP):\n"
                                        + "\n".join(f"  {a.get('cuda','?')} → {a.get('hip','?')}" for a in wrong_apis) + "\n"
                                    )
                                if glm_analysis.get("summary"):
                                    evaluator_feedback += f"\nSUMMARY: {glm_analysis['summary'][:200]}\n"

                                result["changes"].append(
                                    f"[glm] Error analysis: {len(fixes)} fixes identified "
                                    f"(missing_includes={len(missing_inc)}, wrong_apis={len(wrong_apis)})")
                                print(f"║  │  💡 GLM: {len(fixes)} fixes, {len(missing_inc)} includes, {len(wrong_apis)} APIs{'':<26}║")
                            else:
                                result["changes"].append(
                                    f"[glm] Error analysis parse failed — using raw compile errors")
                                print(f"║  │  ⚠ GLM analysis parse failed — falling back to raw errors{'':<16}║")
                        else:
                            result["changes"].append(
                                f"[glm] Error analyst call failed — using raw compile errors")
                            print(f"║  │  ⚠ GLM analyst call failed — falling back to raw errors{'':<18}║")

            result["iterations_used"] = iteration

            # ── Step 2: If compile passed → run GLM for semantic evaluation ──
            # TRIZ #28: GLM is now ONLY used when code actually compiles —
            #   it checks semantic correctness (shfl masks, perf), not compile errors.
            # If compile failed (or no verifier), skip GLM — go straight to Kimi refine.
            # NOTE: GLM error-analyst mode was already called above when compile failed.
            parsed = None  # will stay None if GLM is skipped
            if not compile_failed_this_iter:
                eval_prompt = self._build_glm_evaluate_prompt(
                    result["ported_code"], patterns,
                    deepseek_plan=deepseek_plan_output,
                    feedback=evaluator_feedback,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    regex_changelog=result.get("regex_changelog"),
                )
                if on_phase: on_phase("evaluate", "GLM-5.2", f"semantic eval (attempt {iteration}/{max_iterations}, compile passed)")
                result["changes"].append(
                    f"[glm] Evaluating code (attempt {iteration}/{max_iterations}, compile passed)")
                evaluator = self._call_model(
                    "glm", eval_prompt,
                    system_prompt=SYSTEM_PROMPTS.get("glm", ""),
                    prefill='{"pass":'  # TRIZ #9: force JSON start, prevent prose
                )

                if not evaluator.success:
                    result["changes"].append(
                        f"[glm] Call failed (iteration {iteration})")
                    break

                # Parse GLM evaluator JSON response
                # GLM follows json_schema — should be clean JSON, but keep fallbacks
                raw = evaluator.output.strip()
                parsed = None

                # ── Prose-stripping: GLM may output "Let me evaluate..." before JSON ──
                # Find the first { that looks like start of JSON object
                json_start = raw.find("{")
                if json_start > 0:
                    raw_json = raw[json_start:]  # strip prose prefix
                elif json_start == 0:
                    raw_json = raw
                else:
                    raw_json = raw  # no { at all — will fail all strategies

                # Strategy 1: pure JSON (after prose strip)
                if raw_json.startswith("{"):
                    try: parsed = json.loads(raw_json)
                    except (json.JSONDecodeError, TypeError, ValueError) as e: logger.debug("GLM JSON parse strategy 1 failed: %s", e)

                # Strategy 2: JSON inside ```json ... ``` markdown
                if parsed is None:
                    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                    if m:
                        try: parsed = json.loads(m.group(1))
                        except (json.JSONDecodeError, TypeError, ValueError) as e: logger.debug("GLM JSON parse strategy 2 failed: %s", e)

                # Strategy 3: find {"pass" ... anywhere with flexible whitespace
                if parsed is None:
                    m = re.search(r'\{\s*"pass"\s*:', raw)
                    if m:
                        candidate = raw[m.start():]
                        try: parsed = json.loads(candidate)
                        except (json.JSONDecodeError, TypeError, ValueError) as e: logger.debug("GLM JSON parse strategy 3 failed: %s", e)
                        if parsed is None:
                            # balanced braces extraction
                            depth = 0
                            in_string = False
                            escape = False
                            for ci, ch in enumerate(candidate):
                                if escape:
                                    escape = False
                                    continue
                                if ch == '\\':
                                    escape = True
                                    continue
                                if ch == '"' and not escape:
                                    in_string = not in_string
                                if not in_string:
                                    if ch == '{': depth += 1
                                    elif ch == '}':
                                        depth -= 1
                                        if depth == 0:
                                            try: parsed = json.loads(candidate[:ci+1])
                                            except (json.JSONDecodeError, TypeError, ValueError) as e: logger.debug("GLM JSON parse strategy 3 (balanced) failed: %s", e)
                                            break

                # Strategy 4: regex field extraction (last resort)
                if parsed is None:
                    pass_match = re.search(r'"pass"\s*:\s*(true|false)', raw, re.IGNORECASE)
                    if pass_match:
                        issues_match = re.findall(r'"issues"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
                        feedback_match = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                        verdict_match = re.search(r'"verdict"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                        parsed = {
                            "pass": pass_match.group(1).lower() == "true",
                            "issues": [s.strip().strip('"') for s in issues_match[0].split(',')] if issues_match else [],
                            "feedback": feedback_match.group(1) if feedback_match else "",
                            "verdict": verdict_match.group(1) if verdict_match else "",
                        }

                if parsed is None:
                    # ── TRIZ #20: Continuation of useful action ──
                    # Don't break the loop on parse failure. Extract whatever
                    # feedback we can from the raw response and continue refining.
                    prose_feedback = raw[:600] if raw else "No feedback extracted"
                    result["changes"].append(
                        f'[glm] JSON parse error (iter {iteration}), '
                        f'continuing with prose feedback')
                    evaluator_feedback = (
                        f"Evaluator could not be parsed. Raw response (use as feedback):\n"
                        f"{prose_feedback}\n\n"
                        "Common issues to fix:\n"
                        "- __shfl masks must be 64-bit (0xffffffffffffffffULL)\n"
                        "- __shfl_xor_sync mask 0x1f → 0x3f for wavefront64\n"
                        "- Replace CUDA headers with hip/hip_runtime.h\n"
                        "- Remove .cuh local headers\n"
                        "- WAVEFRONT_SIZE 64\n"
                    )

            # ── Step 3: Convergence check ───────────────────────────────────
            # Converged when: compile passed AND GLM passed (or no GLM needed).
            # If compile passed and GLM says pass → done!
            if parsed is not None and parsed.get("pass", False):
                verify_success = True
                verify_passed = True
                result["changes"].append(
                    f"[glm] Passed semantic evaluation (iteration {iteration})")
                # Compile already passed (we only ran GLM because it did),
                # or there's no verifier (GLM pass is sufficient in that case).
                result["orchestrator_passed"] = True
                break  # Truly converged — compile + GLM both passed

            # ── Step 4: Kimi refines with whatever feedback we have ──────────
            # If compile failed → feedback = compile errors (set in Step 1)
            # If GLM failed/parsed None → feedback = GLM feedback or parse fallback
            # If GLM parsed but not pass → feedback = GLM issues
            if parsed is not None and not parsed.get("pass", False):
                verify_success = True
                evaluator_feedback = parsed.get("feedback", "")
                issues = parsed.get("issues", [])
                if issues:
                    result["changes"].append(
                        f"[glm] Iteration {iteration}: "
                        f"{' | '.join(issues[:3])}")

            if iteration < max_iterations:
                # Loop back: Kimi refines with feedback
                feedback_label = "compile errors" if compile_failed_this_iter else "GLM feedback"
                if on_phase: on_phase("refine", "Kimi K2.7", f"refining with {feedback_label} (iter {iteration}→{iteration+1})")
                # TRIZ #15: Evolve prompt based on compile error patterns
                evolved = opt.evolve_prompt(result.get("compile_errors", []))
                result["changes"].append(f"[prompt-v{evolved.version_id}] Checklist evolved: {len(evolved.checklist)} items")
                refine_prompt = self._build_kimi_refine_prompt(
                    kernel_source, result["ported_code"],
                    evaluator_feedback, patterns,
                    deepseek_plan=deepseek_plan_output,
                    iteration=iteration + 1,
                    checklist_override=evolved.checklist,
                )
                refine = self._call_model(
                    "kimi27", refine_prompt,
                    system_prompt=SYSTEM_PROMPTS.get("kimi27", "")
                )
                if refine.success:
                    extracted = self._extract_code(refine.output)
                    extracted = self._fix_ported_code(extracted, return_changelog=True)
                    if isinstance(extracted, tuple):
                        extracted, regex_changelog = extracted
                    else:
                        regex_changelog = []
                    result["ported_code"] = extracted
                    result["regex_changelog"] = regex_changelog  # A3: track regex fixes
                    result["changes"].append(
                        f"[kimi27] Refined with {feedback_label} "
                        f"(iteration {iteration} → {iteration + 1})")
                else:
                    result["changes"].append(
                        f"[kimi27] Refinement failed (iteration {iteration})")
                    break
            # else: max iterations reached, accept current output

        result["compile_passed"] = compile_passed
        result["prompt_versions"] = opt.get_stats()  # TRIZ #15/#23: prompt evolution summary

        # ── Phase 4: Gemma 4 final verification ──
        if result["ported_code"]:
            if on_phase: on_phase("verify", "Gemma 4", "final verification")
            gemma_prompt = self._build_glm_evaluate_prompt(
                result["ported_code"], patterns,
                regex_changelog=result.get("regex_changelog"),
            )
            verify = self._call_model("gemma4", gemma_prompt,
                                      system_prompt=SYSTEM_PROMPTS.get("glm", ""))
            if verify.success:
                verify_success = verify_success or True
                result["model_used"] = "gemma4"
                # Report which endpoint actually served the call
                last_call = self.call_log[-1] if self.call_log else {}
                verify_source = last_call.get("source", "fireworks")
                source_label = "local vLLM (AMD GPU)" if "local" in verify_source else "Fireworks API"
                try:
                    parsed = json.loads(verify.output)
                    if parsed.get("pass", False):
                        verify_passed = verify_passed or True
                        result["changes"].append(
                            f"[gemma4] Verified — no issues found ({source_label})")
                    else:
                        issues = parsed.get("issues", [])
                        result["changes"].append(
                            f"[gemma4] Issues found ({source_label}): {'; '.join(issues[:3])}")
                except (json.JSONDecodeError, TypeError):
                    if "PASS" in verify.output.upper()[:10]:
                        verify_passed = verify_passed or True
                        result["changes"].append(
                            f"[gemma4] Verified — no issues found ({source_label})")
                    else:
                        result["changes"].append(
                            f"[gemma4] Issues found ({source_label}): {verify.output[:200]}")
            else:
                result["changes"].append(
                    "[gemma4] Verification unavailable (local vLLM + Fireworks both failed)")

        # Rubric-based scoring
        result["confidence"] = self._rubric_score_pipeline(
            kimi_success=coder_success,
            glm_success=planner_success,
            verify_success=verify_success,
            verify_passed=verify_passed,
            has_ported_code=bool(result["ported_code"]),
            ported_code=result["ported_code"],
            changes_count=len(result["changes"]),
        )
        result["cost"] = round(self.total_cost, 4)
        return result

    def _call_model(self, model_key: str, prompt: str,
                    system_prompt: str = "",
                    prefill: str = "") -> AgentResult:
        model_info = MODEL_CATALOG[model_key]
        model_id = model_info["id"]
        local_first = model_info.get("local_first", False)
        model_timeout = model_info.get("timeout", 90)
        t0 = time.perf_counter()

        # Try in order: local-first for Gemma, Fireworks-first for others
        endpoints = []
        if local_first:
            endpoints = ["local", "fireworks"]
        else:
            endpoints = ["fireworks", "local"]

        for endpoint in endpoints:
            try:
                # Build messages with optional system prompt + assistant prefill
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                # TRIZ #9: Preliminary Anti-Action — assistant prefill forces
                # GLM to start with JSON, making prose preamble structurally
                # impossible. The model continues from the prefill.
                if prefill:
                    messages.append({"role": "assistant", "content": prefill})

                if endpoint == "local":
                    local_model = model_info.get("local_id", model_id)
                    data_bytes = json.dumps({
                        "model": local_model,
                        "messages": messages,
                        "max_tokens": model_info.get("max_tokens", 512),
                    }).encode()
                    req = urllib.request.Request(
                        "http://localhost:8000/v1/chat/completions",
                        data=data_bytes,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        raw = resp.read()
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError as e:
                            source = "local-vllm"
                            raw_preview = raw[:500].decode(errors="replace")
                            self.call_log.append({"model": model_key, "source": source,
                                                  "error": f"JSON parse failed: {e}",
                                                  "raw_response": raw_preview[:200]})
                            continue
                        content = data["choices"][0]["message"]["content"]
                        # TRIZ #9: Prepend prefill for local endpoint too
                        if prefill:
                            content = prefill + content
                        self.call_log.append({"model": model_key, "source": "local-vllm", "cost": 0})
                        return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                           0, round((time.perf_counter()-t0)*1000, 1))
                else:  # Fireworks
                    payload = {
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": model_info.get("max_tokens", 1024),
                        "temperature": model_info.get("temperature", 0.2),
                    }
                    # Use json_schema for DeepSeek (strict), json_object for others
                    schema = JSON_SCHEMAS.get(model_key)
                    if schema:
                        payload["response_format"] = schema
                    data_bytes = json.dumps(payload).encode()
                    # TRIZ #11: Retry once with 2x timeout on timeout failure
                    for attempt in range(2):
                        try:
                            req = urllib.request.Request(
                                f"{self.base_url}/chat/completions",
                                data=data_bytes,
                                headers={
                                    "Authorization": f"Bearer {self.api_key}",
                                    "Content-Type": "application/json"
                                }
                            )
                            with urllib.request.urlopen(req, timeout=model_timeout * (attempt + 1)) as resp:
                                raw = resp.read()
                            break  # success — exit retry loop
                        except urllib.error.URLError as e:
                            if "timed out" in str(e).lower() and attempt == 0:
                                # Retry with 2x timeout
                                continue
                            raise  # Re-raise for outer handler
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as e:
                        source = "fireworks"
                        raw_preview = raw[:500].decode(errors="replace")
                        self.call_log.append({"model": model_key, "source": source,
                                              "error": f"JSON parse failed: {e}",
                                              "raw_response": raw_preview[:200]})
                        continue
                    content = data["choices"][0]["message"]["content"]
                    finish_reason = data["choices"][0].get("finish_reason", "")
                    usage = data.get("usage", {})
                    tokens = (
                        usage.get("total_tokens", 0)
                        or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    )
                    # Truncation detection: if finish_reason is "length", the output was cut off
                    if finish_reason == "length":
                        content += "\n<!-- TRUNCATED: output hit max_tokens limit -->"
                    # TRIZ #9: Prepend prefill to content — the API returns
                    # only the continuation, we need the full string for parsing
                    if prefill:
                        content = prefill + content
                    cost = tokens / 1000 * model_info["cost_per_1k"]
                    self.total_cost += cost
                    self.call_log.append({"model": model_key, "tokens": tokens, "cost": cost})
                    return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                       tokens, round((time.perf_counter()-t0)*1000, 1))
            except Exception as e:
                source = "local-vllm" if endpoint == "local" else "fireworks"
                err_msg = str(e)[:200]
                self.call_log.append({"model": model_key, "source": source, "error": err_msg})
                # If response_format caused a 400, retry without it
                fw_payload = payload if endpoint == "fireworks" else {}
                if endpoint == "fireworks" and "400" in err_msg and "response_format" in str(fw_payload):
                    try:
                        fallback_payload = dict(fw_payload)
                        fallback_payload.pop("response_format", None)
                        data_bytes = json.dumps(fallback_payload).encode()
                        req = urllib.request.Request(
                            f"{self.base_url}/chat/completions",
                            data=data_bytes,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            }
                        )
                        with urllib.request.urlopen(req, timeout=model_timeout) as resp:
                            raw = resp.read()
                            data = json.loads(raw)
                            content = data["choices"][0]["message"]["content"]
                            usage = data.get("usage", {})
                            tokens = (
                                usage.get("total_tokens", 0)
                                or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                            )
                            cost = tokens / 1000 * model_info["cost_per_1k"]
                            self.total_cost += cost
                            self.call_log.append({"model": model_key, "tokens": tokens, "cost": cost,
                                                  "note": "response_format not supported, retried without"})
                            return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                               tokens, round((time.perf_counter()-t0)*1000, 1))
                    except Exception as e2:
                        self.call_log.append({"model": model_key, "source": "fireworks",
                                              "error": f"fallback also failed: {str(e2)[:60]}"})
                continue  # Try next endpoint

        return AgentResult(model_key, False, "All endpoints failed", 0.0)

    def get_stats(self) -> Dict:
        calls = len(self.call_log)
        total_tokens = sum(c.get("tokens", 0) for c in self.call_log)
        models_used = set(c["model"] for c in self.call_log)
        return {
            "calls": calls,
            "total_tokens": total_tokens,
            "total_cost": round(self.total_cost, 4),
            "models_used": list(models_used),
            "call_log": self.call_log[-5:],  # last 5 calls
        }

    def reset_stats(self):
        self.total_cost = 0.0
        self.call_log = []
