"""
Model Router — Lightweight agent orchestration for CUDA→ROCm porting.

Architecture:
  Risk Classifier → Model Router → DeepSeek (planner) OR GLM-5.2 (coder) OR Kimi K2.7 (evaluator)
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
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from prompt_evolution import prompt_opt, PromptOptimizer
from debug_session import DebugSession
from verification.spec_parser import auto_generate_spec as _auto_gen_spec
from verification.structural import (
    validate_structure as _validate_structure,
    ValidationResult as _StructuralResult,
)
from verification.lexical import validate_lexical as _validate_lexical
from verification.extraction import extract_code as _extract_code_v2


# ── Per-iteration state ─────────────────────────────────────────────────────
#
# Before the structural gate was added, an iteration had exactly two outcomes:
# hipcc-passed or hipcc-failed. Every reader downstream assumed the compile
# branch had run, so `compile_errs`, `cc`, `error_origins`, `linker_only` and
# friends were only ever bound inside the compile-fail path.
#
# The structural gate introduced a THIRD outcome — rejected before hipcc — that
# raises `compile_failed_this_iter` without ever entering the compile branch.
# Any downstream reader outside the compile-fail sub-block (e.g. the informed
# re-plan at ~router.py:3347) then crashed with UnboundLocalError on the very
# variables it had previously relied on.
#
# This dataclass is the single source of truth for an iteration's outcome.
# Every field has a safe default, so no reader can hit an unbound name again.
# The old local names (compile_errs, linker_only, ...) are still mirrored in
# the loop body for backward compatibility with the ~200 existing references —
# but the *authoritative* value is on `IterationState`, and every branch is now
# required to fill it before any reader downstream.
@dataclass
class IterationState:
    """Outcome of one refinement iteration.

    ``gate`` records which stage produced this iteration's verdict:

        "structural" — code was rejected before hipcc ran (text-level defect)
        "compile"    — hipcc ran; ``compile_success`` says whether it passed
        "run"        — hipcc passed; binary ran; ``run_crashed`` says whether
        "skipped"    — no verifier attached; iteration is a no-op refine

    The unified ``compile_failed`` property is what the loop's control flow
    keys on: it is True whenever the iteration did NOT reach green, regardless
    of which gate stopped it. This preserves the pre-existing semantics of
    ``compile_failed_this_iter`` without requiring any reader to know about the
    new structural gate.
    """
    iteration: int
    gate: str = "skipped"
    # Structural gate
    structural_ok: bool = True
    structural_reject: bool = False
    structural_errors: List[str] = field(default_factory=list)
    structural_missing: List[str] = field(default_factory=list)
    # Compile gate
    compile_ran: bool = False
    compile_success: bool = False
    compile_errs: List[str] = field(default_factory=list)
    error_origins: List = field(default_factory=list)
    linker_only: bool = False
    all_harness_origin: bool = False
    # Run gate
    run_crashed: bool = False
    # LLM analyses attached to this iteration
    glm_analysis: Optional[dict] = None
    replanned: bool = False
    # Stage contract: records which repair mode was active when entering
    # the refine/repair phase.  None = no repair needed this iteration.
    repair_mode: Optional[str] = None  # None | "lexical" | "structural" | "compiler"

    @property
    def compile_failed(self) -> bool:
        """True whenever this iteration did not reach a green compile+run.

        Structural reject is treated as a failure equivalent to a compile
        error — the code is not ready and the loop must refine — but it is
        **not** a compile error, so compile-specific recovery paths (DeepSeek
        informed re-plan, GLM error-analyst) must gate on
        ``compile_ran and not compile_success`` instead of on this property.
        """
        return self.structural_reject or (self.compile_ran and not self.compile_success)


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
    "glm": {
        "id": "accounts/fireworks/models/glm-5p2",  # ✅ VERIFIED WORKING
        "role": "coder",            # CHANGED: was evaluator → coder
        "strength": "code generation, struct-aware HIP porting, structured output",
        "cost_per_1k": 0.0014,
        "local_first": False,
        "max_tokens": 16384,     # coder needs room for full kernel + JSON wrapper
        "temperature": 0.1,      # code generation needs precision
        "timeout": 180,          # coder with large tokens needs the most time
    },
    "kimi27": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",  # ✅ VERIFIED WORKING
        "role": "evaluator",         # CHANGED: was coder → evaluator
        "strength": "structured JSON output, correctness checking, wavefront64 validation, error analysis",
        "cost_per_1k": 0.00095,
        "local_first": False,
        "max_tokens": 1024,      # evaluator output is compact (pass/fail + issues)
        "temperature": 0.0,      # deterministic evaluation
        "timeout": 120,          # evaluator, compact output
    },
    "gemma4": {
        "id": "accounts/fireworks/models/gemma-4-31b-it",  # Fireworks hosted
        "fallback_id": "accounts/fireworks/models/deepseek-v4-pro",  # unavailable on this account
        "local_id": "gemma-4-31b-it",  # Model name when served via local vLLM
        "role": "verifier",          # final verification
        "strength": "Verification — local vLLM on MI300X if available, else Fireworks (falls back to DeepSeek v4 Pro)",
        "cost_per_1k": 0.0,
        "local_first": True,     # Try localhost:8000 first, then Fireworks, then fallback
        "max_tokens": 1024,
        "temperature": 0.0,
        "timeout": 60,           # fast verification model
    },
}

# ── JSON schemas for response_format (enforced by Fireworks API) ──
# Using json_object (not json_schema) for compatibility — the system prompt
# already defines the exact shape. json_schema is stricter but not all models support it.

JSON_SCHEMAS = {
    "glm": {  # GLM coder — json_object for structured code output
        "type": "json_object",
    },
    "kimi27": {  # Kimi evaluator — json_object
        "type": "json_object",
    },
    # DeepSeek is planner — no response_format (prose/reasoning is OK)
}
# ── P0: hard wall-clock budget for one route() call ──
# 5 iterations x ~3 min = the 15-minute run we are trying to kill. max_iterations
# bounds the iteration COUNT, not wall time: a single Kimi call that times out at
# 180s and retries at 2x burns ~9 minutes on its own. This bounds wall time.
#
# Implemented as a monotonic deadline rather than signal.SIGALRM (which the audit
# prompt suggested) for two reasons: SIGALRM does not exist on Windows, where this
# suite is developed, and it only fires on the main thread. A deadline is portable,
# testable without spawning real timers, and — because _call_model clamps each
# request timeout to the remaining budget — it actually bounds the blocking socket
# reads that consume nearly all of the wall time. A signal could not do the latter.
MAX_PIPELINE_SECONDS = int(os.environ.get("MAX_PIPELINE_SECONDS", "300"))

# Floor for a clamped LLM timeout. Below this a request cannot realistically
# round-trip, so we fail fast instead of issuing a request doomed to time out.
MIN_LLM_TIMEOUT_SECONDS = 5

# ── P0: phase budgets ──
# Clamping each call to the REMAINING budget is not enough. Each model's own
# timeout is a large fraction of the whole pipeline: deepseek 120s and kimi27
# 180s against a 180s budget. Nothing stopped the planner from eating the entire
# run before the coder had written a line.
#
# The 2026-07-10 run is exactly that: DeepSeek planned for 86.9s, Kimi coded for
# 93.5s — together 180.4s, 100% of the budget — and the loop reached its deadline
# holding code that had never compiled, let alone been refined. It could not have
# converged from there, at any max_iterations.
#
# The planner is the phase to squeeze. Its output is advisory: Kimi's prompt
# handles deepseek_plan="" and route() already logs "proceeding without plan".
# Without the coder there is no kernel at all. So the planner gets a slice, and
# the coder plus one compile get a reservation the planner may not touch.
PLAN_BUDGET_FRACTION = 0.20      # planner may use at most 20% of the pipeline budget
CODE_RESERVE_FRACTION = 0.55     # of the budget, held back for the coder
COMPILE_RESERVE_SECONDS = 25     # clock kept for hipcc; a compile cannot be interrupted

# ── P0: per-stage budget caps (max, not target) ──
# Each phase gets a hard ceiling so no single stage can exhaust the pipeline
# before later phases have had their fair shot. These are floors for remaining
# budget, not allocations — a phase receives min(CAP, what is left after
# protected reserves). The sum of strict-phase caps (PLAN + CODEGEN = 100)
# plus COMPILE_RESERVE (25) plus REPAIR_RESERVE (60) is 185, which is 5s over
# the 180s pipeline budget — but these are MAXES, not targets, so the actual
# pipeline fits as long as the planner and coder together run in ≤ 90s, which
# the observed 22s + 90s trace already satisfies.
PLAN_CAP = 30              # planner: at most 30s
CODEGEN_CAP = 70           # coder: at most 70s for the initial HIP port
COMPILE_RESERVE = 25       # same as COMPILE_RESERVE_SECONDS — always protected
REPAIR_RESERVE = 60        # protected until first compile failure, then released for repair cycles
VERIFY_CAP = 15            # final verification: at most 15s per call

# Output-token ceiling for a planner that was handed a hipified draft. It is asked
# for a delta checklist — "__shfl_up_sync width 32→64 at L78", not a porting essay —
# and tokens emitted are what the phase's wall time is actually made of. The 2048
# default let a 38.2s plan restate a translation the regex had already finished.
PLAN_DELTA_MAX_TOKENS = 640

# ── P0: stagnation thresholds ──
# Each stagnant iteration costs a Kimi refine (~180s) + a GLM analysis (~30s) +
# a DeepSeek re-plan (~80s). Three of them do not fit in a 3-minute demo, so the
# loop must give up while there is still budget to return a best attempt.
#
# STAGNATION_ABORT_THRESHOLD was effectively 3-with-5-replans, which needed ~6
# iterations to fire and so never did before the user hit Ctrl+C.
STAGNATION_ABORT_THRESHOLD = 2   # consecutive iterations with no error reduction
MAX_REPLANS = 2                  # total DeepSeek re-plans allowed per route() call


class PipelineTimeoutError(RuntimeError):
    """Raised when route() exceeds its wall-clock budget."""


class PortMode(str, Enum):
    """Describes how a CUDA source should be ported.

    WHOLE_PROGRAM — the source is a complete, self-contained program with its
    own ``main()`` AND all of ``main()``'s dependencies survive in the port.
    The coder must reproduce everything, including the host driver.

    DEVICE_SUBSET — the source self-contains ``main()`` but one or more
    user-defined helpers called by ``main()`` cannot be resolved (missing
    local headers, undefined symbols). The coder should port only the
    ``__global__`` / ``__device__`` functions and drop the host driver.
    """
    WHOLE_PROGRAM = "WHOLE_PROGRAM"
    DEVICE_SUBSET = "DEVICE_SUBSET"


class Deadline:
    """Monotonic wall-clock budget shared by route() and _call_model.

    `budget_s <= 0` means "no limit" — every check passes and remaining() is None,
    so callers fall back to their configured per-model timeouts.
    """

    def __init__(self, budget_s: float):
        self.budget_s = budget_s
        self._t0 = time.monotonic()

    @property
    def unlimited(self) -> bool:
        return self.budget_s <= 0

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def remaining(self) -> Optional[float]:
        """Seconds left, or None when unlimited."""
        if self.unlimited:
            return None
        return self.budget_s - self.elapsed()

    def expired(self) -> bool:
        return (not self.unlimited) and self.remaining() <= 0

    def exhausted(self) -> bool:
        """True when too little budget remains to complete another LLM call.

        Distinct from expired(): with 3s left and a 5s floor, the clock has not
        run out but every request we could issue is already doomed. Treating that
        as "still running" makes a timeout look like a model failure.
        """
        if self.unlimited:
            return False
        return self.remaining() < MIN_LLM_TIMEOUT_SECONDS

    def clamp_timeout(self, timeout_s: float) -> float:
        """Shrink a per-request timeout so it cannot outlive the budget."""
        rem = self.remaining()
        if rem is None:
            return timeout_s
        return max(0.0, min(timeout_s, rem))

    def has_at_least(self, seconds: float) -> bool:
        """True when `seconds` of budget remain (always true when unlimited)."""
        if self.unlimited:
            return True
        return self.remaining() >= seconds

    def phase_cap(self, fraction: float, reserve_s: float = 0.0) -> Optional[float]:
        """Seconds a single phase may spend, or None when unlimited.

        Two independent limits, whichever is tighter:
          - `fraction` of the TOTAL budget — a share, so a fast phase cannot
            hoard the clock just because it happened to run first;
          - whatever remains after holding back `reserve_s` for later phases.

        Returns a value that may be <= 0, meaning "this phase cannot run at all".
        Callers decide whether to skip it or fail; the distinction matters, since
        skipping the planner is fine and skipping the coder is not.
        """
        if self.unlimited:
            return None
        share = self.budget_s * fraction
        after_reserve = self.remaining() - reserve_s
        return min(share, after_reserve)


# ── Prompt versioning ──
# Every system prompt string below carries "[prompt vX.Y.Z]" so a run's output
# can be correlated with the exact prompt text that produced it. Bump on ANY
# prompt edit and add an entry to prompts/CHANGELOG.md + data/prompt_changelog.json:
#   MAJOR — behavioral change (new instructions, removed constraints)
#   MINOR — context additions (new examples, expanded edge cases)
#   PATCH — wording, typo, formatting fixes
PROMPT_VERSION = "v4.0.0"

# ── Role-specific system prompts ──
# Each model gets its OWN role definition. No shared prompts.
# These are passed as system messages to the LLM alongside the phase prompt.

SYSTEM_PROMPTS = {
    "deepseek": (
        f"You are a CUDA-to-HIP porting planner. [prompt {PROMPT_VERSION}] "
        "Analyze the CUDA kernel and produce a detailed porting plan: "
        "list every CUDA-specific construct, its HIP replacement, and the order of changes. "
        "Focus on warp(32)→wavefront(64) divergence, __shfl mask widths, shared memory sizing, "
        "header replacements, and any local .cuh dependencies that must be inlined or removed. "
        "Write your plan as clear prose with a numbered checklist of fixes. "
        "Reason freely — your plan will be consumed by a coder agent. "
        "EXCEPTION: when you are shown a HIP draft that has already been mechanically "
        "translated, plan ONLY the warp→wavefront semantics that remain. Do not restate "
        "header swaps or API renames that the draft already applies, and keep the plan "
        "to a terse checklist — the coder needs the delta, not a tutorial."
    ),
    "glm": (
        f"You are a CUDA-to-HIP code porting specialist. [prompt {PROMPT_VERSION}] "
        "Port CUDA kernels to AMD ROCm/HIP, fixing warp→wavefront issues. "
        "Respond with JSON: {\"ported_code\":str,\"confidence\":int,\"changes\":[str],\"explanation\":str}."
    ),
    "kimi27": (
        f"You are a HIP kernel code evaluator. [prompt {PROMPT_VERSION}] "
        "Check ported code for wavefront64 correctness, CUDA remnants, and compilation safety. "
        'Respond with JSON: {"pass":bool,"issues":[str],"feedback":str,"verdict":str}. '
        "CRITICAL: Begin your response with the { character. "
        "DO NOT include reasoning, explanations, greetings, or chain-of-thought. "
        "Output ONLY the JSON object. The first character MUST be {. "
        "No text before or after the JSON."
    ),
    # Used by the in-loop evaluator error-analysis call.
    "kimi_error_analyst": (
        f"You are a HIP/ROCm compile error analyst. [prompt {PROMPT_VERSION}] "
        "Respond ONLY with JSON."
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
# the full DeepSeek→GLM→Kimi pipeline regardless of detected patterns.
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
      3. DeepSeek plans the fix structure (if complex)
      4. GLM-5.2 generates the code
      5. Kimi K2.7 evaluates the output
      6. DeepSeek verifies the output
    """

    def __init__(self, api_key: str = "", debug: Optional[bool] = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = "https://api.fireworks.ai/inference/v1"
        self.total_cost = 0.0
        self.call_log: List[Dict] = []
        self.run_id: str = ""  # set by route() for A2AMessage full_ref keys
        # Replaced by route() with a real budget. Unlimited by default so a
        # bare _call_model() outside the pipeline keeps its per-model timeout.
        self._deadline: Deadline = Deadline(0)
        # Phase 11: Debug Mode. `self.debug` is ALWAYS a session object — a null
        # one when Debug Mode is off — so no call site branches on it. route()
        # replaces it with a live session when enabled.
        self._debug_requested = debug
        self.debug = DebugSession.disabled()
        # Names the stage that the next _call_model() belongs to, so a raw model
        # response lands in 02_planning/ vs 03_translation/ vs 10_evaluation/
        # without _call_model needing to know the pipeline's shape.
        self._debug_stage: str = ""
        # True when route() created the session and must therefore finalize it.
        self._owns_debug_session: bool = True
        # ── P0: stage-budget tracking ──
        # _repair_released is set True after the first compile failure, freeing
        # REPAIR_RESERVE seconds for refine/retry cycles in the iteration loop.
        self._repair_released: bool = False
        # Adaptive-stop tracking: consecutive iterations producing the same
        # normalized first-error signature trigger an early abort instead of
        # burning budget on an unstuck loop.
        self._last_error_sig: str = ""
        self._same_error_count: int = 0
        # Semantic Translation Repair Engine: a deterministic strategy→outcome
        # cache shared across every kernel this router handles in a session, so a
        # repair learned on one file is tried first on the next (Phase 9). Also
        # tracks which (patched) code we have already deterministically repaired,
        # so a compile-fail iteration does not re-run the engine on code it just
        # produced and get an unchanged result.
        self._semantic_repair_cache: Dict[str, str] = {}
        self._semantic_repaired_hashes: set = set()

    def _finalize_debug(self, result: Dict):
        """Write the summary — but only for a session route() itself created.

        A caller that supplied the session (main.py, which still has an
        authoritative verify() compile to record) finalizes it themselves. This
        method is the single place that rule is enforced.
        """
        if not self._owns_debug_session:
            return None
        return self.debug.finalize(result)

    def _attempt_semantic_repair(self, cuda_source: str, hip_source: str,
                                 compile_errs: List[str], verifier,
                                 kernel_name: str, iteration: int):
        """Deterministic pre-LLM repair: compiler diagnostics → minimal patches.

        Runs the Semantic Translation Repair Engine over the current compile
        errors. The engine recovers semantic information the translation dropped
        (a macro, a ``__device__`` helper, a struct) from the ORIGINAL CUDA
        source and restores it with the smallest additive edit — never
        regenerating the file, never calling a model.

        Returns ``(patched_code, compile_check)`` when the repair strictly
        reduces the error count (or compiles clean), else ``None``. The single
        confirming ``quick_compile_check`` here is what guarantees the engine can
        never hand back code that compiles worse than what it was given.
        """
        if not verifier or not hasattr(verifier, "quick_compile_check"):
            return None
        if not (cuda_source and hip_source and compile_errs):
            return None
        # Skip code we have already repaired once: if the last engine pass left
        # this exact text, re-running it would only reproduce the same result.
        code_hash = hash(hip_source)
        if code_hash in self._semantic_repaired_hashes:
            return None
        try:
            from verification.semantic_repair import SemanticRepairEngine
        except Exception as exc:  # never let a repair import take down the loop
            logger.debug("semantic repair unavailable: %s", exc)
            return None

        try:
            engine = SemanticRepairEngine(
                cuda_source, hip_source,
                debug_session=self.debug, cache=self._semantic_repair_cache)
            # Additive high-confidence pass (instant, no compile): restores every
            # symbol the CUDA source can account for in one shot.
            dry = engine.repair(compile_errs)
        except Exception as exc:
            logger.debug("semantic repair engine raised: %s", exc)
            return None

        self._semantic_repaired_hashes.add(code_hash)
        if not dry.changed:
            return None

        # Confirm the patched code actually compiles better before adopting it.
        with self.debug.stage("hipcc"):
            cc = verifier.quick_compile_check(dry.patched_code, kernel_name=kernel_name)
        before = len(compile_errs)
        after = len(cc.get("errors", []))
        improved = cc.get("compile_success") or after < before
        self.debug.transition(
            "SEMANTIC_REPAIR",
            reason=(f"restored {len(dry.accepted_patches)} symbol(s) from CUDA source; "
                    f"errors {before}→{after}" + ("" if improved else " (rejected — no improvement)")),
            validation_result=improved, iteration=iteration)
        if not improved:
            return None
        self._semantic_repaired_hashes.add(hash(dry.patched_code))
        return dry.patched_code, cc

    # Definition-shaped markers for the injected NVIDIA helper shims. Any one of
    # them present means the block is already in the file. Checked instead of the
    # `// _verifier_helper_shims` comment alone, which _extract_code can strip.
    _HELPER_SHIM_SENTINELS = (
        "_verifier_helper_shims",
        "struct StopWatchInterface {",
        "static inline int findCudaDevice",
    )

    # A markdown fence line, with or without a language tag (```cpp, ``` c++, ```HIP).
    _FENCE_LINE = re.compile(r'^[ \t]*```[^\n]*$', re.MULTILINE)

    # The truncation marker in either spelling — the legacy HTML form is a parse
    # bomb at C++ file scope and must never reach hipcc.
    _TRUNCATION_MARKER = re.compile(
        r'^[ \t]*(?:<!--[ \t]*TRUNCATED[^\n]*?-->|//[ \t]*TRUNCATED[^\n]*)$', re.MULTILINE)

    @classmethod
    def _sanitize_extracted(cls, code: str) -> str:
        """Strip artifacts that are text, not C++, from an extracted port.

        _extract_code's fallbacks capture from a code-like anchor to the END of the
        model's response, so a closing ``` fence, trailing prose, or an appended
        truncation marker ride along into the file hipcc compiles. None of those are
        defects a model can fix — it never wrote them — so every refinement iteration
        reproduces the same errors and the budget drains (Δ+0, new:0).

        Everything from the first stray fence onward is dropped: a fence at file
        scope means the code ended there and prose began.
        """
        if not code:
            return code
        code = cls._TRUNCATION_MARKER.sub("", code)
        m = cls._FENCE_LINE.search(code)
        if m:
            code = code[:m.start()]
        return code.strip()

    @classmethod
    def _extract_code(cls, text: str) -> str:
        """Extract HIP/C++ code from LLM output.

        Priority:
          1. JSON {\"ported_code\": \"...\"} — Kimi's expected response format
          2. Markdown code blocks ```hip/cpp/cuda ... ```
          3. Raw code starting from #include or __global__
          4. Fallback: return as-is

        Every return runs through _sanitize_extracted: strategies 3 and 3b anchor on
        a code token and then take everything to the end of the response, so a
        closing fence and the model's trailing explanation come with it.
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
                return cls._sanitize_extracted(extracted)

        # ── Strategy 2: Markdown code blocks ──
        # The language tag is whatever the model felt like typing: cpp, c++, C++,
        # hip, HIP, cu. Pinning an allow-list meant `” ```c++ ”` fell through to
        # Strategy 3, which swallowed the closing fence and the prose after it.
        blocks = _re.findall(r'```[ \t]*[A-Za-z0-9_+#+-]*[ \t]*\r?\n(.*?)```', text, _re.DOTALL)
        if blocks:
            # Return the largest block (most likely the full kernel)
            return cls._sanitize_extracted(max(blocks, key=len))

        # ── Strategy 3: Raw code from #include or __global__ ──
        match = _re.search(r'#include.*', text, _re.DOTALL)
        if match:
            return cls._sanitize_extracted(match.group(0))
        match = _re.search(r'__global__\s+void.*', text, _re.DOTALL)
        if match:
            return cls._sanitize_extracted(match.group(0))

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
                    return cls._sanitize_extracted(salvaged)

        # ── Strategy 4: Fallback ──
        return cls._sanitize_extracted(text)

    @staticmethod
    def _hipify_source(cuda_source: str):
        """Mechanically translate CUDA → HIP before any LLM sees the kernel.

        Returns (hipified_source, changelog).

        This is the same deterministic transform `_fix_ported_code` already runs
        on Kimi's OUTPUT, pointed at the INPUT instead. Nothing new is invented:
        that table is idempotent (verified in tests), so a hipified source may
        pass through it again after Kimi without drift.

        Why this matters: on nvidia_shfl_scan.cu (15k chars, 419 lines) it
        rewrites 53 cuda* API calls, 2 CUDA includes, 2 NVIDIA helper headers and
        38 checkCudaErrors call sites — in ~19ms. Those are transforms the coder
        was spending ~90s of LLM time reproducing by hand, one token at a time.

        What it CANNOT do is warp(32)→wavefront(64) semantics: __shfl width
        arguments, shared memory sized blockDim/32, mask literals whose meaning
        (not spelling) changes. That is the part worth an LLM, and it is exactly
        what the SIGSEGV in the 2026-07-09 run was.
        """
        out = ModelRouter._fix_ported_code(cuda_source, return_changelog=True)
        if isinstance(out, tuple):
            return out
        return out, []

    @staticmethod
    def _residual_cuda_symbols(source: str) -> List[str]:
        """CUDA identifiers the mechanical pass could not translate.

        A non-empty list means hipify alone cannot be trusted — something in the
        source has no deterministic HIP spelling, so the fast path must not claim
        a port. Used as a cheap pre-check before spending a compile on it.
        """
        return sorted(set(re.findall(r'\bcuda[A-Z]\w*', source)
                          + re.findall(r'\bcu(?:Blas|Rand|Fft|Sparse|Solver)\w*', source)))

    # Warp-level primitives whose MEANING changes on a 64-lane wavefront. A regex
    # can rename them; it cannot re-derive the lane arithmetic around them.
    _WAVEFRONT_SENSITIVE = re.compile(
        r'__shfl\w*|__syncwarp|__ballot\w*|__activemask|__any_sync|__all_sync'
        r'|__match_\w+_sync|__reduce_\w+_sync|\bwarpSize\b'
    )

    @classmethod
    def _needs_wavefront_semantics(cls, source: str, patterns: Optional[List[Dict]] = None) -> bool:
        """True when a mechanical port cannot possibly be correct.

        The fast path costs a real hipcc compile (~20s of a 180s budget). Spending
        it on a kernel that uses __shfl or warpSize buys nothing: the translation
        will compile and then SIGSEGV, because shared memory sized blockDim/32 and
        a width=32 shuffle are wrong on wavefront64 no matter how the symbols are
        spelled. That is the 2026-07-09 crash, and it is statically visible.

        Checks the source directly (robust when the classifier found nothing) and
        the classifier's findings (catches spellings the regex misses).
        """
        if cls._WAVEFRONT_SENSITIVE.search(source):
            return True
        for p in (patterns or []):
            name = str(p.get("pattern", "")).lower()
            if "shfl" in name or "warp" in name or "ballot" in name or "syncwarp" in name:
                return True
        return False

    @staticmethod
    def _compute_adaptive_max_tokens(source: str, model_key: str = "glm") -> int:
        """Scale the coder's output budget to the kernel it must emit.

        glm (coder) asks for 16384 tokens on every call. A 2k-char kernel cannot use
        them, but the request still reserves capacity and the model still drifts
        toward filling the space. Budget for the source round-tripping back out,
        plus the JSON wrapper and escaping overhead, then clamp.

        ~4 chars/token is the usual English/code ratio; x2 covers the response
        restating the kernel, and the floor keeps small kernels from truncating.
        """
        ceiling = MODEL_CATALOG.get(model_key, {}).get("max_tokens", 16384)
        estimated = int((len(source) / 4) * 2) + 512  # round-trip + JSON wrapper
        return max(2048, min(ceiling, estimated))

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

        # ── NVIDIA helper_cuda.h / helper_functions.h compat shims ──
        # Those headers are stripped above (no HIP equivalent ships in
        # ROCm), but full NVIDIA sample programs (not bare kernel snippets)
        # often call symbols they used to provide. checkCudaErrors has a
        # simple textual replacement above; these don't, because call sites
        # vary (assigned to a variable, passed a pointer-to-pointer, etc.) —
        # define tiny compatible stand-ins instead of guessing every call
        # site. See docs/fix-plan-self-contained-programs.md, Bug 2.
        # The guard must key on the shim's DEFINITIONS, not on its marker comment.
        # _extract_code's raw-code fallback starts at the first `#include`, which
        # discards every line above it — including `// _verifier_helper_shims`. The
        # struct and functions below it survive, so a marker-only guard re-injected
        # the whole block on the next pass and hipcc reported a redefinition of
        # StopWatchInterface and findCudaDevice. These two sentinels are
        # definition-shaped: the original NVIDIA sample CALLS both symbols but
        # defines neither, so they cannot be confused with legitimate use.
        if (not any(s in code for s in ModelRouter._HELPER_SHIM_SENTINELS)
                and ModelRouter._is_self_contained(code)):
            # The shim is injected BEFORE the code's first #include (so its
            # symbols are declared before any use), which means it CANNOT
            # rely on the program's own includes — it must bring everything
            # it references. hipSetDevice lives in hip/hip_runtime.h; omitting
            # it here made every self-contained port fail with 'use of
            # undeclared identifier' at hipSetDevice's exact column (56),
            # re-injected on every refine iteration — unfixable by the LLM
            # loop because the corruption post-dates the model's output.
            # (No #pragma once: this lands in the main file, not a header.)
            shim = (
                '\n// _verifier_helper_shims — stand-ins for NVIDIA helper_cuda.h /\n'
                '// helper_functions.h symbols (no HIP/ROCm equivalent ships).\n'
                '#include <hip/hip_runtime.h>\n'
                '#include <chrono>\n'
                '#include <cstdlib>\n'
                # EXIT_WAIVED is a helper_string.h macro, not a std one. The NVIDIA
                # samples exit with it when hardware lacks a required feature, and
                # stripping helper_*.h left it undeclared in every restored driver.
                '#ifndef EXIT_WAIVED\n'
                '#define EXIT_WAIVED 2\n'
                '#endif\n'
                'struct StopWatchInterface { std::chrono::steady_clock::time_point start; double elapsed_ms = 0; };\n'
                'static inline int findCudaDevice(int, const char **) { (void)hipSetDevice(0); return 0; }\n'
                'static inline void sdkCreateTimer(StopWatchInterface **t) { *t = new StopWatchInterface(); }\n'
                'static inline void sdkStartTimer(StopWatchInterface **t) { (*t)->start = std::chrono::steady_clock::now(); }\n'
                'static inline void sdkStopTimer(StopWatchInterface **t) {\n'
                '    auto _now = std::chrono::steady_clock::now();\n'
                '    (*t)->elapsed_ms = std::chrono::duration<double, std::milli>(_now - (*t)->start).count();\n'
                '}\n'
                'static inline float sdkGetTimerValue(StopWatchInterface **t) { return (float)(*t)->elapsed_ms; }\n'
                'static inline void sdkDeleteTimer(StopWatchInterface **t) { delete *t; *t = nullptr; }\n'
                'static inline void getLastCudaError(const char *) {}\n\n'
            )
            include_lines = list(re.finditer(r'#include\s+[<"].*?[>"]\n', code))
            if include_lines:
                insert_pos = include_lines[0].start()
                code = code[:insert_pos] + shim + code[insert_pos:]
            else:
                code = shim + code
            changelog.append('NVIDIA helper_cuda/helper_functions compat shims added '
                              '(findCudaDevice, sdk* timers, StopWatchInterface, getLastCudaError)')

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

        # ── CUDA vector type substitutions ──────────────────────────
        # CUDA-specific types (uint4, float4, uchar4, int4, double4) have
        # NO direct HIP equivalents. Replace with base-type pointer or
        # struct definition based on context.
        # uint4 → keep as struct if used in __shared__, else base pointer
        vector_types = [
            (r'\buint4\b', 'unsigned int'),  # 4 × uint32
            (r'\buint2\b', 'unsigned int'),  # 2 × uint32
            (r'\buint3\b', 'unsigned int'),  # 3 × uint32
            (r'\buchar4\b', 'unsigned char'), # 4 × uint8
            (r'\buchar2\b', 'unsigned char'), # 2 × uint8
            (r'\bint4\b', 'int'),             # 4 × int32
            (r'\bint2\b', 'int'),             # 2 × int32
            (r'\bint3\b', 'int'),             # 3 × int32
            (r'\bfloat4\b', 'float'),         # 4 × float32
            (r'\bfloat2\b', 'float'),         # 2 × float32
            (r'\bdouble4\b', 'double'),       # 4 × float64
            (r'\bdouble2\b', 'double'),       # 2 × float64
            (r'\blong4\b', 'long'),           # 4 × int64
            (r'\blong2\b', 'long'),           # 2 × int64
        ]
        for pattern, base_type in vector_types:
            # In __shared__ declarations: "__shared__ uint4 var[N]" → use base_type array
            code = _tracked_sub(
                rf'(__shared__\s+){pattern}(\s+\w+)',
                rf'\g<1>{base_type}\g<2>',
                code, f'{base_type}4→{base_type} (shared)'
            )
            # In variable declarations outside shared: "uint4 var" → base_type
            code = _tracked_sub(
                rf'(?<![.\w]){pattern}(?=[\s*;,()\[\]])',
                base_type,
                code, f'vector type {base_type}4'
            )

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
        # (?<!#define ) — a macro NAME must stay an identifier: Kimi
        # plausibly emits '#define warpSize 64', and the unguarded
        # substitution turned it into '#define 64 64' → hipcc 'macro name
        # must be an identifier'. Uses in macro BODIES / code still convert.
        code = _tracked_sub(r'(?<!#define )\bwarpSize\b', '64', code, 'warpSize→64')
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

    def _postprocess_port(self, model_output: str, kernel_source: str,
                          iteration: int = 0, generation: Optional[int] = None,
                          model: str = "kimi27", tokens: int = 0,
                          latency_ms: float = 0.0,
                          port_mode: Optional[str] = None):
        """Turn a raw coder response into the code the compiler will actually see.

        Returns ``(code, regex_changelog, main_restored, structural)``.

        Phase 11: every generation the loop consumes passes through here, so this
        is where Debug Mode captures the full decision chain for one generation —
        raw response, extraction report, lexical verdict, structural verdict,
        symbol diff, and static-analysis findings. Logging happens after each
        decision is made, so an artifact always reflects the value the pipeline
        actually acted on rather than a re-derivation.

        Every path that accepts model output — the initial port, a refine, and the
        refine's retry — must apply the same steps in the same order, or the
        pipeline compiles something different from what it reasoned about.

        ORDER IS LOAD-BEARING: restore main() BEFORE the mechanical pass, not after.
        _fix_ported_code injects the NVIDIA helper shims (findCudaDevice, sdk*Timer,
        StopWatchInterface) only when it sees a self-contained program. If main() is
        restored afterwards, the file is not self-contained at fix time, the shims
        are never injected, and the driver we just reattached calls findCudaDevice
        into a void. Restoring first also means the driver is hipified by that same
        pass — it is never hipified in isolation, which is what used to duplicate
        the shim block and #define WAVEFRONT_SIZE.

        The structural report is computed against the FINAL code (after main-restore
        and the mechanical pass) so it reflects exactly what hipcc will see. It is
        the single choke point that gates every generation the loop consumes; wiring
        it here means the initial Kimi call, the refine, and the refine-retry all
        pay the ~1ms check before a 60s hipcc call fires on a truncated brace.

        Extraction and lexical validation run FIRST — before any regex fix or the
        structural check.  A response that is pure reasoning ("Let's search memory
        more concretely.  I think...") has balanced braces (zero of each) and would
        pass the structural gate, then compile-fail with ``error: unknown type name
        'local'`` — a symptom of raw LLM text having been written to disk.  The
        lexical gate catches that class and folds its errors into the structural
        result, so the existing "structural reject → refine, no compile, no repair"
        path fires without duplicating the plumbing.
        """
        # Prefer the provider-agnostic extractor.  It handles JSON fields, fenced
        # blocks, and raw-code windows, and NEVER returns markdown fences or
        # trailing prose as part of the code.  On extraction failure we still
        # try the legacy fallback so a partially-recognized response is not
        # silently discarded — the lexical gate below decides whether either
        # candidate is actually source code.
        #
        # The stage timer wraps the parse itself, not the logging of it — that is
        # what makes "parser latency" in metrics.json a measurement rather than
        # an aspiration.
        with self.debug.stage("extraction"):
            extraction = _extract_code_v2(model_output)
            if extraction.ok:
                code = extraction.code
            else:
                code = self._extract_code(model_output)

        # Debug Mode: persist the generation and the extraction decision before
        # any repair regex mutates the text. `gen` ties every later artifact for
        # this generation together.
        gen = generation
        if self.debug.enabled:
            from debug_session import discarded_text as _discarded
            gen = self.debug.log_generation(
                raw_response=model_output,
                extracted_code=code or "",
                discarded=_discarded(model_output, code or ""),
                iteration=iteration, generation=generation, model=model,
                tokens=tokens, latency_ms=latency_ms,
                success=bool(code and code.strip()),
                fallback_extractor_used=not extraction.ok,
            )
            self.debug.log_extraction(extraction, generation=gen, iteration=iteration)
            self.debug.transition(
                "CODE_EXTRACTED",
                reason=f"strategy={extraction.strategy}",
                validation_result=extraction.ok, iteration=iteration, generation=gen)

        # Why a restore was declined matters as much as when one happened: a silent
        # no-op here looks identical to "the coder kept its main()".
        blockers: List[str] = []
        if (self._is_self_contained(kernel_source)
                and not self._is_self_contained(code)):
            original_main = self._extract_main(kernel_source)
            if original_main:
                blockers = self._unsatisfied_main_calls(
                    original_main, code, kernel_source)

        code, restored = self._ensure_main_preserved(code, kernel_source, port_mode=port_mode)
        fixed = self._fix_ported_code(code, return_changelog=True)
        if isinstance(fixed, tuple):
            code, changelog = fixed
        else:
            code, changelog = fixed, []
        if restored:
            changelog.append("main() restored from the original CUDA source")
        elif blockers:
            changelog.append(
                "main() NOT restored — the original driver calls "
                + ", ".join(blockers)
                + ", which this port does not define; reattaching it would create an "
                  "undefined symbol instead of fixing one")

        # Lexical gate FIRST — pure reasoning, markdown, and role tags never
        # reach hipcc.  A failure here is folded into the structural result so
        # the existing structural-reject → refine path fires unchanged.
        with self.debug.stage("lexical_validation"):
            try:
                lexical = _validate_lexical(code)
            except Exception as _lx_exc:  # a bug in the gate must not kill the loop
                logger.debug("lexical validation errored: %s", _lx_exc)
                lexical = None

        # Structural gate against the post-fix code, not the raw extract: hipcc will
        # see the fixed version, so that is what must stand up. A failure here is a
        # hard reject upstream (see the loop's pre-compile check), so keep this
        # scoped strictly to defects that cannot occur in valid C++.
        with self.debug.stage("structural_validation"):
            try:
                structural = _validate_structure(kernel_source, code)
            except Exception as _sv_exc:  # never let the gate itself take down the loop
                logger.debug("structural validation errored: %s", _sv_exc)
                structural = _StructuralResult(ok=True, warnings=["structural check errored"])

        # Semantic gate — catches the pathologies the structural checker misses:
        # prose at file scope, ghost kernel launches, and executable statements
        # outside function bodies.
        with self.debug.stage("semantic_validation"):
            try:
                semantic = self._reject_structurally_invalid(code, kernel_source)
            except Exception as _sv_exc:  # never let the gate itself take down the loop
                logger.debug("semantic validation errored: %s", _sv_exc)
                semantic = _StructuralResult(ok=True, warnings=["semantic check errored"])

        # Debug Mode: record BOTH verdicts as they were computed, before the
        # merge below folds the lexical failure into the structural result. A
        # reader must be able to tell "the braces were fine, the text was prose"
        # from "the braces were broken" — the merged object cannot say that.
        if self.debug.enabled:
            self.debug.log_lexical(lexical, generation=gen, iteration=iteration, code=code)
            self.debug.transition(
                "LEXICAL_VALIDATION",
                reason=(lexical.reason() if lexical is not None else "validator errored"),
                validation_result=(lexical.ok if lexical is not None else None),
                iteration=iteration, generation=gen)
            self.debug.log_structural(structural, generation=gen, iteration=iteration,
                                      cuda_source=kernel_source, hip_source=code)
            self.debug.transition(
                "STRUCTURAL_VALIDATION", reason=structural.reason(),
                validation_result=structural.ok, iteration=iteration, generation=gen)
            self.debug.log_symbols(kernel_source, code, generation=gen, iteration=iteration)
            self.debug.log_structural(semantic, generation=gen, iteration=iteration,
                                      cuda_source=kernel_source, hip_source=code)
            self.debug.transition(
                "SEMANTIC_VALIDATION", reason=semantic.reason(),
                validation_result=semantic.ok, iteration=iteration, generation=gen)
            # Static analysis runs on the exact text hipcc is about to see, and
            # is persisted whether or not a compile follows. When the compile is
            # skipped by a gate, these findings are the only pre-compile
            # evidence that survives.
            self.debug.log_static_analysis(code, generation=gen, iteration=iteration)
            if lexical is not None and not lexical.ok:
                self.debug.count("lexical_rejects")
            if not structural.ok:
                self.debug.count("structural_rejects")
            if not semantic.ok:
                self.debug.count("semantic_rejects")

        if lexical is not None and not lexical.ok:
            merged_errors = ["[lexical] " + e for e in lexical.errors]
            merged_errors.extend(structural.errors)
            structural = _StructuralResult(
                ok=False,
                errors=merged_errors,
                warnings=list(structural.warnings) + [
                    "lexical: " + s for s in lexical.prose_line_samples[:2]
                ],
                missing_symbols=list(structural.missing_symbols),
            )
            changelog.append(
                "[lexical] rejected: " + "; ".join(lexical.errors)[:120])

        if not semantic.ok:
            merged_errors = list(structural.errors) + [
                "[semantic] " + e for e in semantic.errors[:10]
            ]
            structural = _StructuralResult(
                ok=False,
                errors=merged_errors,
                warnings=list(structural.warnings) + list(semantic.warnings),
                missing_symbols=list(structural.missing_symbols),
            )
            changelog.append(
                "[semantic] rejected: " + "; ".join(semantic.errors)[:120])

        return code, changelog, restored, structural

    @staticmethod
    def _reject_structurally_invalid(ported_code: str, original_source: str) -> "_StructuralResult":
        """Semantic gate against three pathologies the structural checker misses.

        1. **Prose at file scope** — evaluator-feedback notes leaked into code,
           detected by signal phrases (The, This, We, Note, But, So, However, ...)
           or ``[A-Z][a-z]+.*` `` patterns at brace depth 0.

        2. **Ghost kernel launches** — ``identifier<<<`` without a matching
           ``__global__ void identifier(`` definition anywhere in the file.

        3. **Executable statements outside function bodies** — lines with ``=``,
           ``;``, ``<<<``, ``->``, ``::`` at file scope that are not includes,
           defines, namespace, template declarations, function signatures,
           forward declarations, comments, or blank lines.
        """
        warnings: List[str] = []
        lines = ported_code.splitlines()

        # ── Pass 1: collect all __global__ kernel definitions ──
        kernel_defs = set()
        for m in re.finditer(r'__global__\s+\w+\s+(\w+)\s*\(', ported_code):
            kernel_defs.add(m.group(1))

        # ── Pass 2: find all kernel launches (identifier<<<) and verify they
        #            have a matching definition (Check #2: ghost kernels) ──
        for m in re.finditer(r'(\w+)\s*<<<', ported_code):
            name = m.group(1)
            # Skip common keywords / types that look like a kernel launch
            if name in ('if', 'for', 'while', 'switch', 'template', 'sizeof',
                        'const', 'constexpr', 'static', 'extern', 'void',
                        'int', 'float', 'double', 'char', 'bool', 'class',
                        'struct', 'enum', 'union', 'auto', 'return',
                        'namespace', 'using', 'typedef', 'typename',
                        'public', 'private', 'protected', 'virtual',
                        'override', 'final', 'export', 'import', 'include',
                        'alignas', 'alignof', 'decltype', 'noexcept',
                        'static_cast', 'dynamic_cast', 'const_cast',
                        'reinterpret_cast', 'hipLaunchKernelGGL',
                        'hipLaunchCooperativeKernel'):
                continue
            if name not in kernel_defs:
                warnings.append(
                    f"ghost kernel launch '{name}': called via <<<>>> "
                    f"but no __global__ definition found")

        # ── Pass 3: check each line at file scope ──
        brace_depth = 0
        in_block_comment = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            lineno = i + 1

            if not stripped:
                continue

            # Handle block comments spanning multiple lines
            if '/*' in stripped and '*/' not in stripped:
                in_block_comment = True
                continue
            if in_block_comment:
                if '*/' in stripped:
                    in_block_comment = False
                continue

            # Strip comments from the line for analysis
            code_part = re.sub(r'//.*$', '', stripped).strip()
            code_part = re.sub(r'/\*.*?\*/', '', code_part).strip()
            if not code_part:
                continue

            # Only check lines at file scope (brace depth == 0)
            if brace_depth == 0:
                # ── Check #1: Prose at file scope ──
                if re.match(
                    r'^(?:[A-Z][a-z]+.*`|(?:The|This|We|Note|But|So|However|'
                    r'Also|Therefore|Thus|Hence)\b)', code_part
                ):
                    warnings.append(
                        f"line {lineno}: natural-language prose leaked at "
                        f"file scope: {code_part[:100]!r}")
                    continue

                # When the line opens a brace at depth 0 (e.g. single-line
                # kernel body), only check the portion before the first {.
                # Content inside the function body will be at depth >= 1.
                before_brace = code_part.split('{', 1)[0].strip()
                check_part = before_brace if before_brace else code_part

                # ── Check #3: Executable statements outside function bodies ──
                # Allow: preprocessor directives, comments, valid file-scope C++
                # declarations, braces, and blank lines.
                if re.match(r'^\s*(?:#|//|\*|/\*)', stripped):
                    pass  # preprocessor / comment
                elif re.match(
                    r'^(?:__global__|__device__|template|namespace|using'
                    r'|typedef|struct|class|enum|union'
                    r'|constexpr|const|static|extern|inline|virtual|friend'
                    r'|typename|public|private|protected)\b', code_part
                ):
                    pass  # valid file-scope declaration keyword
                elif re.match(r'^[A-Za-z_]\w*\s*\(', check_part):
                    pass  # function declaration / signature
                elif stripped in ('{', '}', '};'):
                    pass  # bare braces
                elif re.match(r'^\}', stripped):
                    pass  # closing brace with optional label
                elif re.match(r'^[A-Za-z_]\w*\s*::\s*~?\w+\s*\(', check_part):
                    pass  # method definition outside class (e.g. MyClass::foo())
                elif re.match(
                    r'^(?:#\s*define|#\s*include|#\s*if|#\s*ifdef|#\s*ifndef'
                    r'|#\s*else|#\s*elif|#\s*endif|#\s*pragma|#\s*error'
                    r'|#\s*warning|#\s*line)', stripped
                ):
                    pass  # preprocessor (already caught above but be explicit)
                else:
                    # Reject if it contains executable-statement markers
                    has_semicolon = ';' in check_part
                    has_kernel_launch = '<<<' in check_part
                    # '=' not preceded by <, >, !, or = (avoid ==, !=, <=, >=)
                    has_assign = bool(re.search(r'(?<![-<>=!])=(?!=)', check_part))
                    has_arrow = '->' in check_part
                    has_scope = ('::' in check_part
                                 and '::' not in code_part.split('(')[0].strip().rstrip('*&'))

                    if has_semicolon and (has_kernel_launch or has_assign or has_arrow or has_scope):
                        warnings.append(
                            f"line {lineno}: executable statement outside "
                            f"function body: {check_part[:100]!r}")

            # Update brace depth for the next line
            brace_depth += stripped.count('{') - stripped.count('}')

        ok = len(warnings) == 0
        return _StructuralResult(
            ok=ok, errors=list(warnings), warnings=[], missing_symbols=[])

    @staticmethod
    def _is_self_contained(source: str) -> bool:
        """True when *source* defines its own ``int main(`` — a complete,
        runnable program, not a bare kernel snippet.

        Mirrors the check verifier.py's ``_generate_harness`` uses to decide
        whether to wrap a driver around the ported code — however, the spec's
        ``port_mode`` key is the authoritative source for that decision now.
        This function is retained for prompt-truncation decisions (whether to
        show the full source to the coder/planner so ``main()`` is not silently
        truncated out of their context window) and for *incremental* compile-time
        fallbacks (e.g. the regex check at line 239 of verifier.py).

        A self-contained program's own ``main()`` can sit past a 5000-character
        truncation boundary — truncating that out from under the coder means it
        never saw (let alone could reproduce) the driver that runs the kernels.
        See docs/fix-plan-self-contained-programs.md, Bug 1.
        """
        return bool(re.search(r'^\s*int\s+main\s*\(', source, re.MULTILINE))

    @staticmethod
    def _strip_to_kernel_only(source: str) -> str:
        """Extract only ``__global__`` and ``__device__`` function bodies.

        Strips: copyright comments, ``#include`` lines, ``#define`` outside
        kernels, host-only functions, ``int main()``, and all other non-kernel
        code.  Result starts with the first ``__global__`` / ``__device__``
        function so the coder cannot see (and therefore cannot reproduce)
        copyright banners, host helpers, or the host driver.

        TRIZ #10 (Preliminary Action) + #22 (Throwing Away): instead of telling
        the coder not to reproduce host code, simply never show it.
        """
        lines = source.splitlines()
        extracted = []
        in_kernel = False
        brace_depth = 0

        for line in lines:
            stripped = line.strip()
            # Detect start of __global__ or __device__ function
            if (stripped.startswith('__global__')
                    or stripped.startswith('__device__')) and not in_kernel:
                in_kernel = True
                extracted.append(line)
                brace_depth += line.count('{') - line.count('}')
                # Single-line balanced kernel (e.g. __global__ void k() { x; })
                # closes brace on same line — immediately reset.
                if brace_depth <= 0:
                    in_kernel = False
                    extracted.append('')  # blank separator
                continue

            if in_kernel:
                extracted.append(line)
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0 and len(extracted) > 1:
                    in_kernel = False
                    # Add a blank line between kernels
                    extracted.append('')

        return '\n'.join(extracted)

    @staticmethod
    def _compute_port_mode(kernel_source: str) -> PortMode:
        """Determine whether the coder should port the whole program or just the device subset.

        Returns ``DEVICE_SUBSET`` when the source is self-contained (has its own
        ``int main(``) but one or more user-defined helpers that ``main()`` calls
        are not satisfiable (missing local headers, unresolved symbols). In that
        case the coder is told to port only the ``__global__`` / ``__device__``
        functions and to drop the host driver (which will be replaced by a
        synthesized harness).

        Returns ``WHOLE_PROGRAM`` otherwise — the coder must reproduce everything,
        including the driver, or the existing harness path handles it.
        """
        if not ModelRouter._is_self_contained(kernel_source):
            return PortMode.WHOLE_PROGRAM
        main_text = ModelRouter._extract_main(kernel_source)
        if not main_text:
            return PortMode.WHOLE_PROGRAM
        # A self-contained program whose driver depends on code that cannot be
        # reproduced in HIP must be ported as DEVICE_SUBSET: keep only the
        # __global__/__device__ kernels and let the synthesized harness drive
        # them. Two deterministic signals of an unportable driver:
        #   1. an unvendored quoted local header (e.g. shfl_integral_image.cuh)
        #      whose functions can never be defined here; and
        #   2. an NVIDIA CUDA-SDK helper header (helper_cuda.h / helper_functions.h
        #      / helper_timer.h …) — these supply host-only utilities like
        #      findCudaDevice / checkCudaErrors / the sdk*Timer family that have
        #      no HIP equivalent and cannot be reproduced by the coder.
        #
        # BUG FIX: the previous implementation called
        # ``_unsatisfied_main_calls(main_text, kernel_source, kernel_source)`` —
        # comparing the source against ITSELF, so ``available`` always equalled
        # ``original_funcs`` and the dropped-symbol set was always empty. This
        # function could therefore NEVER return DEVICE_SUBSET: every full NVIDIA
        # sample was ported WHOLE_PROGRAM, told to reproduce main(), and dragged
        # in unportable SDK host code that no refinement iteration could compile
        # (the nvidia_shfl_scan TIMEOUT). Programs with neither signal below stay
        # WHOLE_PROGRAM, exactly as before.
        if ModelRouter._unresolved_local_headers(kernel_source):
            return PortMode.DEVICE_SUBSET
        if ModelRouter._uses_cuda_sdk_helpers(kernel_source):
            return PortMode.DEVICE_SUBSET
        return PortMode.WHOLE_PROGRAM

    # NVIDIA CUDA-Samples helper headers: host-only utilities (findCudaDevice,
    # checkCudaErrors, StopWatchInterface / sdk*Timer, …) that ship only with the
    # CUDA Samples repo and have no ROCm/HIP equivalent. A self-contained program
    # that includes one cannot be reproduced whole — only its device kernels port.
    _CUDA_SDK_HELPER_HEADERS = re.compile(
        r'#\s*include\s*[<"]\s*helper_\w+\.(?:h|hpp|cuh)\s*[>"]')

    @staticmethod
    def _uses_cuda_sdk_helpers(source: str) -> bool:
        """True when *source* includes an NVIDIA CUDA-SDK ``helper_*`` header."""
        return bool(ModelRouter._CUDA_SDK_HELPER_HEADERS.search(source))

    @staticmethod
    def _spec_port_mode(kernel_name: str, verifier=None) -> Optional["PortMode"]:
        """Return the hand-tagged ``port_mode`` from *kernel_name*'s JSON spec.

        Function-level analysis alone cannot always tell that a driver is
        unportable, so a human may tag ``port_mode`` in the spec. That tag is
        the authoritative override (see test_port_mode's design note and the
        route() comment). Returns None when there is no spec, no verifier to
        load it, or the spec does not name a recognized mode.
        """
        if verifier is None or not hasattr(verifier, "load_spec"):
            return None
        try:
            spec = verifier.load_spec(kernel_name)
        except Exception:
            return None
        if not spec:
            return None
        raw = spec.get("port_mode")
        for mode in PortMode:
            if raw == mode.value:
                return mode
        return None

    @staticmethod
    def _extract_main(source: str) -> str:
        """Return the full text of *source*'s ``int main(...)`` definition, or "".

        Brace-matched rather than regex-terminated: a driver's body contains
        nested blocks and string literals with braces in them, so "everything
        up to the next ``}`` at column 0" silently truncates mid-function on
        any program that indents its closing brace.

        Skips a ``main`` that is only declared (``int main(int, char**);``) —
        there is nothing to preserve there.
        """
        m = re.search(r'^[ \t]*int[ \t]+main[ \t]*\(', source, re.MULTILINE)
        if not m:
            return ""
        open_brace = source.find("{", m.end())
        if open_brace < 0:
            return ""
        # A declaration ends at ';' before any body opens.
        semi = source.find(";", m.end())
        if 0 <= semi < open_brace:
            return ""

        depth = 0
        in_string = in_char = in_line_comment = in_block_comment = False
        escape = False
        for i in range(open_brace, len(source)):
            ch = source[i]
            nxt = source[i + 1] if i + 1 < len(source) else ""
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                continue
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                continue
            if in_string or in_char:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif (ch == '"' and in_string) or (ch == "'" and in_char):
                    in_string = in_char = False
                continue
            if ch == "/" and nxt == "/":
                in_line_comment = True
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "'":
                in_char = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[m.start():i + 1]
        return ""  # unbalanced — refuse to hand back a truncated driver

    @classmethod
    def _ensure_main_preserved(cls, ported_code: str, original_source: str,
                               port_mode: Optional[str] = None):
        """Re-append the driver when the coder dropped a self-contained program's main().

        Returns ``(code, restored: bool)``.

        The 2026-07-09 run's entire 180s budget went to this: the CUDA source was a
        complete program, the coder returned only the kernels, and hipcc failed with
        ``ld.lld: error: undefined symbol: main``. Every agent downstream then
        reasoned about a *compile* error that was really a missing-driver error, and
        three LLM phases (plan, analyse, re-plan) could not fix what none of them
        had been told.

        The driver is not regenerated — it is lifted verbatim from the original CUDA
        source, so no model is in the loop and the cost is a regex.

        It is appended RAW, still speaking CUDA. The caller (_postprocess_port) runs
        _fix_ported_code over the combined file immediately afterwards, which both
        hipifies the driver and — because the file now has a main() — injects the
        NVIDIA helper shims the driver needs. Hipifying the driver here instead would
        either duplicate that shim preamble or, if stripped, leave findCudaDevice and
        the sdk*Timer family undefined.

        Only fires when the original was self-contained and the port is not: a port
        that kept its main() is left untouched, and a bare kernel snippet has no
        driver to restore.

        When ``port_mode`` is ``\"DEVICE_SUBSET\"`` the restore is skipped — the
        coder was instructed to drop the host driver intentionally (its dependencies
        cannot be resolved), and a synthesized harness replaces it.
        """
        if not ported_code or not original_source:
            return ported_code, False
        # DEVICE_SUBSET: the coder was told to drop the driver intentionally.
        # Reattaching it would re-introduce the very unresolved symbols that
        # triggered DEVICE_SUBSET mode in the first place.
        if port_mode == PortMode.DEVICE_SUBSET.value:
            return ported_code, False
        if not cls._is_self_contained(original_source):
            return ported_code, False
        if cls._is_self_contained(ported_code):
            return ported_code, False

        original_main = cls._extract_main(original_source)
        if not original_main:
            # Unbalanced or undetectable — refuse to append a truncated driver.
            return ported_code, False

        # Never trade a compile problem for a link problem. If the driver calls a
        # helper the port does not define, reattaching it invents an undefined
        # symbol that no refinement can resolve without deleting the driver again.
        if cls._unsatisfied_main_calls(original_main, ported_code, original_source):
            return ported_code, False

        return (ported_code.rstrip() + "\n\n"
                "// ── main() restored from the original CUDA source ──\n"
                "// The port dropped the driver of a self-contained program. This is the\n"
                "// original main(), verbatim — no model rewrote it. The caller's\n"
                "// mechanical CUDA→HIP pass translates it and injects the helper shims\n"
                "// it needs, now that this file has a main() again.\n"
                + original_main + "\n"), True

    # A function DEFINITION at file scope: `... name(args) {`. Deliberately not a
    # declaration (`...;`) — a prototype defines no symbol for the linker.
    _FUNC_DEF = re.compile(
        r'^[ \t]*(?:(?:static|inline|extern|__global__|__device__|__host__|'
        r'template\s*<[^>]*>)\s+)*'
        r'[A-Za-z_][A-Za-z0-9_:<>,\t \*&]*?\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?\{',
        re.MULTILINE)

    # An identifier immediately followed by `(` — a call, a definition, or a cast.
    _CALL_SITE = re.compile(r'\b([A-Za-z_]\w*)\s*\(')

    # Control-flow keywords that look like calls to the regex above.
    _NOT_CALLS = frozenset({
        "if", "for", "while", "switch", "return", "sizeof", "catch",
        "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
    })

    @classmethod
    def _defined_functions(cls, code: str) -> set:
        """Names of functions *defined* (not merely declared) at file scope in code."""
        return {m.group(1) for m in cls._FUNC_DEF.finditer(code)} - cls._NOT_CALLS

    @classmethod
    def _unsatisfied_main_calls(cls, main_text: str, ported_code: str,
                                original_source: str) -> List[str]:
        """User-defined helpers that *main* calls but the port does not define.

        Restoring a driver verbatim is only safe when the driver's own dependencies
        survive in the port. `nvidia_shfl_scan.cu` is the counter-example: its main()
        calls ``shuffle_integral_image_test()``, whose body needs
        ``shfl_integral_image.cuh`` — a header this repo never vendored. The coder is
        told, correctly, to drop that whole code path. Reattaching a main() that calls
        it anyway converts a compile problem into an unfixable link problem, and every
        refinement iteration then fights the restore.

        Only names *defined in the original* are considered. Runtime and library
        symbols (printf, exit, hipMalloc, the injected shims) are, by construction,
        not function definitions in the original .cu, so they are never reported.
        """
        original_funcs = cls._defined_functions(original_source) - {"main"}
        if not original_funcs:
            return []
        called = {m.group(1) for m in cls._CALL_SITE.finditer(main_text)} - cls._NOT_CALLS
        available = cls._defined_functions(ported_code)
        return sorted((called & original_funcs) - available)

    # A diagnostic emitted by the linker, not the compiler. `clang++: error: linker
    # command failed` always accompanies the real `ld.lld: undefined symbol: X` line,
    # so both spellings must count or the set is never "linker-only".
    _LINKER_ERROR = re.compile(
        r'undefined (?:symbol|reference)|linker command failed|^\s*ld(?:\.lld)?\s*:')

    @classmethod
    def _is_linker_only(cls, compile_errors: List[str],
                        error_origins: Optional[List[str]] = None) -> bool:
        """True when every hipcc diagnostic is a link failure, not a compile error.

        ``undefined symbol: main`` is not something the coder can debug from the error
        text: nothing in the file it wrote is syntactically wrong. Sending it to the
        error analyst buys a confident, plausible, useless root cause, and sending it
        to the planner buys a fresh strategy for a problem that is not strategic. Both
        cost ~38s of a 180s budget. The fix is a regex (`_ensure_main_preserved`), so
        the models are skipped entirely.

        Note this is deliberately NOT `all(o == "link" for o in error_origins)`:
        verifier._classify_error_origin tags only the ``undefined symbol: main`` line
        "link" and leaves its inseparable ``clang++: ... linker command failed`` mate
        "unknown", so that predicate is false on every real link failure. Origins are
        used as corroboration; the text is what decides.
        """
        errs = [e for e in compile_errors if e.strip()]
        if not errs:
            return False
        if not all(cls._LINKER_ERROR.search(e) for e in errs):
            return False
        # An origins list that names a non-link compile error contradicts the text.
        return not any(o in ("ported_code", "harness") for o in (error_origins or []))

    @classmethod
    def _is_missing_main_error(cls, compile_errors: List[str],
                               error_origins: Optional[List[str]] = None) -> bool:
        """True when the linker is specifically saying ``main`` is undefined."""
        if "link" in (error_origins or []):
            return True
        return any(re.search(r'undefined (?:symbol|reference to)[\s\S]{0,10}\bmain\b', e)
                   for e in compile_errors)

    @staticmethod
    def _unresolved_local_headers(source: str) -> List[str]:
        """Return quoted local ``.cuh``/``.h`` includes in *source* that don't
        exist anywhere under ``sample_kernels/`` in this repo.

        A full NVIDIA sample may depend on a project-specific header (e.g.
        ``shfl_integral_image.cuh``) that was never vendored into this repo.
        ``_fix_ported_code`` strips the ``#include`` line, but any function
        whose implementation lives only in that header remains an undefined
        symbol — Kimi cannot port code it cannot see, and inventing a stub
        would silently change program behavior. See Bug 3.
        """
        repo_sample_dir = Path(__file__).resolve().parent.parent / "sample_kernels"
        missing = []
        for m in re.finditer(r'#include\s*"([^"]+\.(?:cuh|h|hpp))"', source):
            fname = m.group(1)
            if not list(repo_sample_dir.rglob(Path(fname).name)):
                missing.append(fname)
        return missing

    def _build_deepseek_plan_prompt(self, kernel_source: str,
                                    patterns: List[Dict],
                                    hipified_source: str = "") -> str:
        """Build the DeepSeek planner phase prompt with classifier context.

        Role: DeepSeek-Planner — reasons freely about the CUDA kernel and produces
        a detailed porting plan as prose. No JSON required — reasoning is the asset here.
        The plan is passed to Kimi-Coder as context.

        hipified_source: the mechanically translated draft (see _hipify_source). When
        supplied, the planner is asked for the DELTA only. Planning the header swaps
        and cuda*→hip* renames that a regex already performed is work whose output is
        discarded: the coder is editing the draft, not the original. On 2026-07-09 the
        planner spent 38.2s — 21% of the budget — re-deriving a translation that had
        finished in 19ms, and the one thing it needed to say (the __shfl_up_sync width)
        was a single line.
        """
        if hipified_source:
            # No CUDA embed at all. The draft IS the translation of it, and the
            # planner's job here is not translation.
            prompt = (
                "A CUDA kernel has already been mechanically translated to HIP: headers "
                "swapped, cuda*→hip* renamed, checkCudaErrors replaced, WAVEFRONT_SIZE "
                "defined. That work is DONE and correct. Do not plan it again.\n\n"
                "Plan ONLY the warp(32)→wavefront(64) semantics a regex cannot do:\n"
                + self._WAVEFRONT_CHECKLIST + "\n"
            )
            pattern_summary = _format_patterns_summary(patterns)
            if pattern_summary:
                prompt += pattern_summary + "\n"
            prompt += (
                f"HIP DRAFT (already translated — plan the edits to THIS):\n"
                f"```hip\n{hipified_source[:4000]}\n```\n\n"
                "Reply with a short numbered checklist: one line per edit, naming the "
                "construct and the exact replacement. If a checklist item does not apply "
                "to this kernel, omit it. Do not restate what is already done. "
                "Be brief — the coder needs the delta, not a tutorial."
            )
            return prompt

        prompt = (
            "Analyze this CUDA kernel and produce a porting plan for AMD ROCm/HIP.\n"
            "Identify every CUDA-specific construct and its HIP replacement.\n"
            "Prioritize: warp(32)→wavefront(64) divergence, __shfl mask widths, "
            "shared memory sizing, header swaps, local .cuh dependencies.\n\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        # Bug 1: a self-contained program's own main() can sit past a fixed
        # character budget — truncating here means the planner (and, via its
        # plan, Kimi) never learns the program has a driver to preserve.
        source_for_prompt = kernel_source if self._is_self_contained(kernel_source) else kernel_source[:5000]
        prompt += (
            f"```cuda\n{source_for_prompt}\n```\n\n"
            "Write a detailed porting plan as a numbered checklist. "
            "For each item: what to change, where (line/construct), and why. "
            "Be specific — a coder agent will follow your plan exactly."
        )
        return prompt

    def _build_deepseek_replan_prompt(self, kernel_source: str,
                                      patterns: List[Dict],
                                      failed_code: str,
                                      compile_errors: List[str],
                                      previous_plan: str) -> str:
        """Build a stagnation-recovery re-plan prompt (Bug 7).

        The original stagnation escalation re-sent the exact same prompt as
        the initial plan (no compile errors, no failed code, no indication
        the first plan didn't work), so DeepSeek — at temperature 0.3, but
        starting from byte-identical input — produced a substantially
        identical plan. This prompt instead shows DeepSeek what it tried and
        why it failed, and explicitly asks for a DIFFERENT strategy.
        """
        err_text = "\n".join(compile_errors[:8]) if compile_errors else "(no specific errors captured)"
        prompt = (
            "Your previous porting plan for this CUDA kernel produced code that "
            "still fails to compile after multiple refinement attempts. The same "
            "or similar errors keep recurring — the current strategy is not working.\n\n"
            f"YOUR PREVIOUS PLAN:\n{previous_plan[:2000]}\n\n"
            f"CODE PRODUCED FROM THAT PLAN (most recent attempt):\n"
            f"```hip\n{failed_code[:2000]}\n```\n\n"
            f"COMPILER ERRORS THAT KEEP RECURRING:\n{err_text}\n\n"
            "Produce a DIFFERENT porting strategy — do not repeat the previous "
            "approach. Consider: a different algorithm structure for the "
            "warp/wavefront logic, different HIP APIs than previously chosen, "
            "or a simpler approach that avoids the construct that keeps failing.\n\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        # Bug 1: see _build_deepseek_plan_prompt — don't truncate a
        # self-contained program's source out from under its own main().
        source_for_prompt = kernel_source if self._is_self_contained(kernel_source) else kernel_source[:5000]
        prompt += (
            f"ORIGINAL CUDA KERNEL:\n```cuda\n{source_for_prompt}\n```\n\n"
            "Write a detailed porting plan as a numbered checklist, explicitly "
            "different from the previous plan above. For each item: what to "
            "change, where, and why. Be specific — a coder agent will follow "
            "your plan exactly."
        )
        return prompt

    # Semantics the mechanical pass cannot do: meaning changes, not spellings.
    # These stay on the coder's checklist even when a hipified draft is supplied.
    _WAVEFRONT_CHECKLIST = (
        "- __shfl_*_sync: the `width` argument and mask semantics change on "
        "wavefront64 (masks → 0x3f / 0xffffffffffffffffULL; width must not stay 32)\n"
        "- shared memory sized blockDim/32 → blockDim/64 (this is the usual SIGSEGV)\n"
        "- warpSize or a hardcoded 32 → WAVEFRONT_SIZE (64)\n"
        "- __syncwarp() → __syncthreads()\n"
    )

    # Spellings the mechanical pass already handles. Only listed when no
    # preprocessed draft was supplied.
    _MECHANICAL_CHECKLIST = (
        "- #define WAVEFRONT_SIZE 64 at top\n"
        "- Replace #include <cuda_runtime.h> → #include <hip/hip_runtime.h>\n"
        "- Remove #include <helper_cuda.h>, <helper_functions.h>, <device_launch_parameters.h>\n"
        "- Remove ALL #include \"*.cuh\" local headers (inline their content if needed)\n"
        "- checkCudaErrors(x) → (void)(x); findCudaDevice/sdkCreateTimer/sdkStartTimer/"
        "sdkStopTimer/sdkGetTimerValue/sdkDeleteTimer/StopWatchInterface/getLastCudaError "
        "have no HIP equivalent — replace with hipSetDevice/std::chrono or remove\n"
    )

    def _build_kimi_code_prompt(self, kernel_source: str,
                                patterns: List[Dict],
                                deepseek_plan: str = "",
                                preprocessed_source: str = "",
                                port_mode: Optional[str] = None) -> str:
        """Build the Kimi K2.7 code generator phase prompt.

        Role: Kimi-Coder — generates the actual ported HIP kernel code.
        Now receives DeepSeek's plan as context (was GLM analysis before).

        preprocessed_source: a mechanically hipified draft (see _hipify_source).
        When supplied, Kimi EDITS it rather than re-porting from scratch: the
        53 API renames and 38 checkCudaErrors sites are already done, so the
        checklist narrows to wavefront semantics and the response is a diff-sized
        edit rather than a 419-line rewrite. Output tokens are the coder's real
        latency, and a rewrite is also what reintroduced compile errors on the
        2026-07-09 run.

        Output format: JSON with ported_code (str), confidence (0-100),
        changes (list[str]), explanation (str).
        """
        if preprocessed_source:
            prompt = (
                "A CUDA kernel has ALREADY been mechanically translated to HIP "
                "(headers, cuda*→hip* API renames, checkCudaErrors, WAVEFRONT_SIZE). "
                "Your job is the part a regex cannot do: warp(32)→wavefront(64) "
                "SEMANTICS.\n\n"
                "EDIT the HIP draft below. Do NOT re-port from scratch and do not "
                "restructure code that is already correct — change only what the "
                "checklist calls out.\n\n"
                "CHECKLIST (semantics only — the mechanical work is done):\n"
                + self._WAVEFRONT_CHECKLIST + "\n"
            )
        else:
            prompt = (
                "Port this CUDA kernel to AMD ROCm/HIP. Fix warp(32)→wavefront(64) issues.\n\n"
                "CHECKLIST:\n"
                + self._WAVEFRONT_CHECKLIST
                + self._MECHANICAL_CHECKLIST + "\n"
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

        # Bug 1: a self-contained program (has its own main()) must be shown
        # in FULL — truncating a fixed-length embed cut off main() itself for
        # any source past ~6000 chars, so Kimi ported only the kernels and
        # never saw (let alone could reproduce) the driver that runs them.
        # V2: TRIZ #10/#22 initially tried _strip_to_kernel_only to prevent
        # host-code reproduction, but this broke plan/pattern line-reference
        # alignment (DeepSeek says "fix line 253" but stripped source has
        # only 60 lines). Now we show the FULL source so plans stay valid
        # and use an explicit instruction to restrict output to kernel-only.
        self_contained = self._is_self_contained(kernel_source)
        source_for_prompt = kernel_source if self_contained else kernel_source[:6000]
        # DEVICE_SUBSET mode: output kernel functions only — explicit instruction.
        if port_mode == PortMode.DEVICE_SUBSET.value:
            if preprocessed_source:
                draft = (preprocessed_source if self_contained
                         else preprocessed_source[:6000])
                excerpt = kernel_source[:2000]
                elided = "\n... (original elided — the draft below is its translation)" \
                    if len(kernel_source) > len(excerpt) else ""
                prompt += (
                    f"ORIGINAL CUDA (reference only — do not port this again):\n"
                    f"```cuda\n{excerpt}{elided}\n```\n\n"
                    f"HIP DRAFT TO EDIT (mechanically translated; return this file with "
                    f"the checklist applied):\n```hip\n{draft}\n```\n\n"
                )
            else:
                prompt += f"```cuda\n{source_for_prompt}\n```\n\n"
            prompt += (
                "\u26a0\ufe0f DEVICE-SUBSET MODE: Output ONLY the __global__ "
                "and/or __device__ kernel function(s). Do NOT include host code, "
                "helper functions, main(), or a test harness. "
                "The system automatically wraps your kernel in a proper "
                "HIP compilation unit.\n\n"
            )
        else:
            if preprocessed_source:
                draft = (preprocessed_source if self_contained
                         else preprocessed_source[:6000])
                excerpt = kernel_source[:2000]
                elided = "\n... (original elided — the draft below is its translation)" \
                    if len(kernel_source) > len(excerpt) else ""
                prompt += (
                    f"ORIGINAL CUDA (reference only — do not port this again):\n"
                    f"```cuda\n{excerpt}{elided}\n```\n\n"
                    f"HIP DRAFT TO EDIT (mechanically translated; return this file with "
                    f"the checklist applied):\n```hip\n{draft}\n```\n\n"
                )
            else:
                prompt += f"```cuda\n{source_for_prompt}\n```\n\n"

        # Bug 3: a full sample may depend on a project-specific local header
        # this repo never vendored (e.g. shfl_integral_image.cuh). Kimi can't
        # port a function it can't see, and inventing a stub would silently
        # change behavior — tell it to drop the dependent code path instead.
        missing_headers = self._unresolved_local_headers(kernel_source)
        if missing_headers:
            prompt += (
                f"NOTE: This source includes {', '.join(missing_headers)}, which is NOT "
                "present in this repository and cannot be ported. Any function whose "
                "implementation lives only in that header — and any test/code path that "
                "calls it — must be DROPPED from your port. Do not invent a substitute "
                "implementation. Port only the self-contained portion of this program "
                "that does not depend on it.\n\n"
            )

        prompt += (
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}.\n"
            "IMPORTANT: The ported_code field must contain the COMPLETE HIP kernel source. "
            "If the kernel is large, minimize explanation to save tokens. "
            "Prefer full code over partial code with verbose explanation."
        )
        # DEVICE_SUBSET is a *refinement* of self_contained: the source has a
        # main() but its host driver is unportable, so we deliberately drop it.
        # Guard the "keep main()" instruction with `not DEVICE_SUBSET` — otherwise
        # it fires alongside the DEVICE_SUBSET "drop main()" instruction above and
        # the coder receives directly contradictory orders, producing the whole
        # unportable program *and* a stray harness (the nvidia_shfl_scan failure).
        if self_contained and port_mode != PortMode.DEVICE_SUBSET.value:
            prompt += (
                "\nCRITICAL: The CUDA source above has its own main() function. Your "
                "ported_code MUST also include a complete main() with identical logic, "
                "ported to HIP APIs. Do NOT strip main() — it drives the full test."
            )
            if missing_headers:
                # The two instructions above and below collide on exactly one line of
                # the driver: main() calls the code path we just told the coder to
                # drop. Left unresolved, the coder either keeps a call to a function
                # it did not define (link error) or drops main() (also a link error).
                prompt += (
                    "\nRESOLVING THE CONFLICT: main() itself calls into the dropped "
                    "code path. Reproduce main() exactly as above, EXCEPT omit the "
                    "call to any function you dropped, omit the variables that hold "
                    "only its result, and simplify any expression that used them. "
                    "Everything else in main() stays. Do not stub the dropped "
                    "function, and do not delete main()."
                )
        elif port_mode == PortMode.DEVICE_SUBSET.value:
            prompt += (
                "\nDEVICE-SUBSET MODE: The CUDA source above has its own main() "
                "but one or more host dependencies cannot be resolved (missing "
                "local headers, undefined host symbols). Port ONLY the "
                "__global__ / __device__ functions — drop the host driver "
                "(main() and all host helper functions called from main()). "
                "The harness authoring system will synthesize a test harness "
                "on its own. Do NOT include main() or any host-only code."
            )
            if missing_headers:
                prompt += (
                    "\nNOTE: The missing headers listed above are known and "
                    "expected — they are exactly why device-subset mode was "
                    "selected. Drop any code path that depends on them."
                )
        return prompt

    def _build_kimi_refine_prompt(self, kernel_source: str,
                                  previous_code: str,
                                  feedback: str,
                                  patterns: List[Dict],
                                  deepseek_plan: str = "",
                                  iteration: int = 1,
                                  checklist_override: list[str] = None,
                                  stagnation_count: int = 0,
                                  regex_changelog: Optional[List[str]] = None,
                                  frozen_base_code: str = "",
                                  preprocessed_source: str = "",
                                  structural_report: Optional[Dict] = None,
                                  port_mode: Optional[str] = None) -> str:
        """Build the Kimi refinement prompt for orchestration loop iterations.

        Kimi receives the original kernel, its previous output, and
        GLM evaluator's specific feedback to fix issues.

        TRIZ #15 (Dynamics): checklist_override allows the PromptOptimizer to
        inject an evolved checklist instead of the static fallback.

        A2A v2 Channels:
          Channel 1 (regex_changelog): deterministic fixes already applied.
          Channel 3 (stagnation): iteration memory — how many failures so far.

        frozen_base_code: P1 two-layer fix. When non-empty, this kernel already
        compiles and only crashes at runtime. Kimi is told to patch it, not to
        rewrite it — a full rewrite is what reintroduced compile errors on the
        2026-07-09 run.

        preprocessed_source: the mechanically hipified draft. NOT embedded here —
        `previous_code` is already the working HIP copy, and re-sending the draft
        would grow the prompt this parameter exists to shrink. It is used to state
        that the mechanical pass is done, and to name any cuda* symbols that have
        crept back into `previous_code` (a regression the loop otherwise rediscovers
        through hipcc, one iteration at a time).
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

        # Bug 1: previous_code[:4000] was independently fatal for a
        # self-contained program — the refine loop handed Kimi a chopped
        # copy of its OWN prior output and asked for complete code back, so
        # it could never converge on anything written past char 4000. Judge
        # self-containment from the original kernel_source (ground truth),
        # not previous_code, since previous_code may itself be missing
        # main() precisely because of the bug this fix addresses.
        # V2: _strip_to_kernel_only broke plan/pattern line-reference alignment.
        # Show full source and use explicit instruction instead.
        self_contained = self._is_self_contained(kernel_source)
        source_for_prompt = kernel_source if self_contained else kernel_source[:4000]
        previous_for_prompt = previous_code if self_contained else previous_code[:4000]
        prompt += (
            f"Original CUDA:\n```cuda\n{source_for_prompt}```\n\n"
            f"Your previous output:\n```hip\n{previous_for_prompt}```\n\n"
        )
        if port_mode == PortMode.DEVICE_SUBSET.value:
            prompt += (
                "\u26a0\ufe0f DEVICE-SUBSET MODE: Output ONLY the __global__ "
                "and/or __device__ kernel function(s). Do NOT include host code, "
                "helper functions, main(), or a test harness. "
                "The system automatically wraps your kernel in a proper "
                "HIP compilation unit.\n\n"
            )
        # ── A2A v2 Channel 3: Iteration memory ──
        if stagnation_count >= 1:
            prompt += (
                f"\n⚠️ ITERATION MEMORY: This is iteration {iteration}. "
                f"There have been {stagnation_count} consecutive iterations without "
                f"compile improvement (stagnation threshold: {STAGNATION_ABORT_THRESHOLD}).\n"
                "Your previous output(s) produced the SAME compiler errors. "
                "Do NOT output the same code again. Try a completely different "
                "approach — different API calls, different structure, different headers.\n\n"
            )
        # ── A2A v2 Channel 1: Deterministic fixes already applied ──
        if regex_changelog:
            changelog_text = "\n".join(f"  ✅ {c}" for c in regex_changelog[:10])
            prompt += (
                f"\nAUTO-FIXED ITEMS (already applied — do NOT redo):\n{changelog_text}\n\n"
            )

        # ── Structural report from the pre-hipcc gate ──
        # These are text-level defects the compiler will otherwise reproduce every
        # iteration (unbalanced braces, dropped helpers, duplicated bodies). Naming
        # them here means the coder acts on the shape of its own output, not on a
        # parser error whose root cause is that shape.
        if structural_report:
            struct_lines = []
            if structural_report.get("errors"):
                struct_lines.append("STRUCTURAL ERRORS (blocked hipcc — fix these FIRST):")
                struct_lines.extend(f"  ❌ {e}" for e in structural_report["errors"])
            if structural_report.get("missing_symbols"):
                struct_lines.append(
                    "SYMBOLS DROPPED from the original CUDA source (restore them):")
                struct_lines.append(
                    "  " + ", ".join(structural_report["missing_symbols"][:12]))
            if structural_report.get("warnings"):
                struct_lines.append("STRUCTURAL WARNINGS:")
                struct_lines.extend(f"  ⚠ {w}" for w in structural_report["warnings"][:6])
            if struct_lines:
                prompt += "\n" + "\n".join(struct_lines) + "\n\n"

        # ── Mechanical pass already applied (see _hipify_source) ──
        # Cheap to say, expensive to rediscover: without it each refine spends
        # output tokens re-deriving cuda*→hip* renames that a regex did in 19ms.
        if preprocessed_source:
            prompt += (
                "\nMECHANICAL PASS ALREADY APPLIED: headers, cuda*→hip* API renames, "
                "checkCudaErrors and WAVEFRONT_SIZE are done deterministically. Do not "
                "redo them and do not reintroduce CUDA spellings. Spend your edit on "
                "wavefront64 semantics.\n"
            )
            crept_back = self._residual_cuda_symbols(previous_code)
            if crept_back:
                prompt += (
                    f"⚠️ Your previous output reintroduced CUDA symbols that the "
                    f"mechanical pass had removed: {', '.join(crept_back[:8])}. "
                    f"Remove them.\n"
                )
            prompt += "\n"
        # ── P1 two-layer SIGSEGV fix: Layer 1 is immutable ──
        # The base kernel below already passed hipcc. The only defect is at runtime.
        # Rewriting it trades a crashing binary for one that does not build at all.
        if frozen_base_code:
            base_for_prompt = (frozen_base_code if self_contained
                               else frozen_base_code[:4000])
            prompt += (
                "\n[IMPORTANT] The base kernel below COMPILES CLEANLY and crashes only "
                "at runtime. It is a frozen baseline (Layer 1).\n"
                "Apply TARGETED fixes on top of it (Layer 2). Do NOT rewrite, "
                "restructure, or re-port the base kernel — change only the specific "
                "lines responsible for the runtime fault (typically __shfl width/mask "
                "semantics on wavefront64, shared memory sized blockDim/32 instead of "
                "blockDim/64, or out-of-bounds indexing).\n"
                "If your output does not compile, it will be DISCARDED and this base "
                "kernel returned instead — a compiling kernel that crashes beats a "
                "kernel that does not build.\n"
                f"BASE KERNEL (Layer 1, compiles):\n```hip\n{base_for_prompt}```\n\n"
            )

        prompt += (
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}."
        )
        # DEVICE_SUBSET wins over self_contained (every DEVICE_SUBSET source is
        # also self_contained). Without this guard the refine loop re-injects
        # "keep main()" every iteration, so a device-subset port can never
        # converge — it kept dragging back the unportable host driver.
        if port_mode == PortMode.DEVICE_SUBSET.value:
            prompt += (
                "\nDEVICE-SUBSET MODE: The original CUDA source has its own main(), "
                "but its host driver depends on code that cannot be ported (missing "
                "local headers / undefined host symbols). Output ONLY the __global__ "
                "and/or __device__ kernel function(s). Do NOT include main(), host "
                "helper functions, or a test harness — the system synthesizes the "
                "harness that drives your kernel."
            )
        elif self_contained:
            prompt += (
                "\nCRITICAL: The original CUDA source has its own main() function. Your "
                "ported_code MUST also include a complete main() with identical logic, "
                "ported to HIP APIs. Do NOT strip main() — it drives the full test."
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
                                         patterns: List[Dict],
                                         error_delta: int = 0,
                                         stagnation_count: int = 0,
                                         error_context: List[str] = None,
                                         self_contained: bool = False,
                                         port_mode: Optional[str] = None) -> str:
        """Build GLM prompt for compile-error analysis (TRIZ #28).

        When hipcc fails, GLM analyzes the compile errors + code and tells
        Kimi WHAT to fix structurally — not just "error: undefined hipMalloc"
        but "you forgot #include <hip/hip_runtime.h>, that's why hipMalloc is undefined."

        error_delta: prev - current (positive = improvement, negative = regression).
        stagnation_count: how many consecutive iterations without improvement.
        Both are derived from normalized error sets for stable tracking.

        error_context: the exact source lines at each error location,
        extracted deterministically by verifier.quick_compile_check(). This
        prompt only shows GLM the first 3000 chars of the code, and the
        honesty rule below forbids explaining errors it can't locate — so
        without these snippets, any error past char 3000 is structurally
        unanalyzable and GLM's output degrades to empty/priority-only fixes
        (the "1 fixes, 0 includes, 0 APIs" plateau in the 2026-07-09 run).

        self_contained: the ORIGINAL CUDA source defines its own main(). Without
        this, an ``undefined symbol: main`` gets a generic "fix the linker errors"
        answer, because nothing in the code excerpt explains it. Route() now restores
        main() mechanically before the analyst is ever called, so this flag exists to
        make the analyst's advice correct in the residual case where some other
        symbol failed to link and the driver is nonetheless present.
        """
        err_text = "\n".join(compile_errors[:10])
        ctx_text = ""
        if error_context:
            ctx_text = (
                "\nSOURCE AT ERROR LOCATIONS (exact lines from the compiled file — "
                "these count as 'found in the code' for the rule below):\n"
                + "\n---\n".join(error_context[:6]) + "\n"
            )

        regression_hint = ""
        if error_delta < 0:
            regression_hint = (
                f"\n\n⚠️ REGRESSION: The error count INCREASED by {abs(error_delta)}\n"
                f"since the last iteration. Your previous recommendation(s) may have made\n"
                f"things worse. Re-examine ALL errors — do NOT repeat the same fixes.\n"
                f"The root cause may be different from what the error messages suggest.\n"
            )
        elif stagnation_count >= 2:
            regression_hint = (
                f"\n\n⚠️ STAGNATION: {stagnation_count} iterations without improvement.\n"
                f"The current approach is not working. Try a COMPLETELY different strategy.\n"
            )

        # The excerpt below is the first 3000 chars, so a driver near the end of a
        # 15k-char program is invisible here. Saying so prevents the analyst from
        # reading its absence as the defect.
        self_contained_note = ""
        if port_mode == PortMode.DEVICE_SUBSET.value:
            # DEVICE_SUBSET intentionally drops the host driver — the harness is
            # synthesized. Telling the analyst to "restore main()" here (the
            # self_contained advice below) steers Kimi to re-import the exact
            # unportable host code that caused the errors, so the loop stagnates
            # (Δ+0) until the wall-clock budget is spent. Give the opposite advice.
            self_contained_note = (
                "\nCONTEXT: this is a DEVICE-SUBSET port. The original CUDA source has a "
                "main() and host helpers, but they depend on unportable code and were "
                "deliberately DROPPED; a synthesized harness drives the kernel. If any "
                "error names main(), a host helper, a test harness, cudaDeviceProp, "
                "findCudaDevice/checkCudaErrors, or another host-only symbol, the correct "
                "fix is to REMOVE that leaked host code from the port — NEVER to restore "
                "main(), add a driver, or add a harness. Only __global__/__device__ "
                "functions belong in the output.\n"
            )
        elif self_contained:
            self_contained_note = (
                "\nCONTEXT: the original CUDA source is a COMPLETE, SELF-CONTAINED program "
                "with its own main(). The code excerpt below may be truncated before that "
                "main() — do NOT report a missing driver as a defect unless a linker error "
                "names it. If one does, the correct fix is 'restore main() from the original "
                "CUDA source', never 'write a new main()' or 'add a test harness'.\n"
            )

        prompt = (
            f"You are a HIP/ROCm compile error analyst. Kimi generated code that fails to compile.\n"
            f"Analyze the compiler errors and tell Kimi EXACTLY what to fix.\n\n"
            f"COMPILER ERRORS (hipcc, iteration {iteration}):\n"
            f"{err_text}\n"
            f"{ctx_text}"
            f"{self_contained_note}\n"
            f"CURRENT CODE (first 3000 chars):\n"
            f"```hip\n{ported_code[:3000]}\n```\n\n"
            "Analyze each error and provide:\n"
            "1. Root cause for each error (not just the error message)\n"
            "2. The EXACT fix needed (specific API name, include, or type)\n"
            "3. Priority order (fix headers first, then types, then logic)\n"
            "4. CONFIDENCE (0-10) in each fix — how sure are you this will compile?\n"
            "   High confidence (8-10): certain the exact fix is correct\n"
            "   Medium (4-7): likely correct but may need adjustment\n"
            "   Low (1-3): speculative or unsure\n\n"
            "Common CUDA→HIP issues:\n"
            "- cuda_runtime.h → hip/hip_runtime.h (causes ALL cuda* functions to be undefined)\n"
            "- cudaMalloc → hipMalloc, cudaMemcpy → hipMemcpy, cudaFree → hipFree\n"
            "- cudaError_t → hipError_t, cudaSuccess → hipSuccess\n"
            "- checkCudaErrors() → remove or define wrapper\n"
            "- __shfl_*_sync mask: 0x1f (32-bit) → 0x3f (64-bit wavefront)\n"
            "- threadIdx.x threadIdx.y etc stay the same in HIP\n\n"
            "IMPORTANT: Only explain errors whose exact text (function name, variable,\n"
            "line content) you can actually find in the CURRENT CODE shown above. If an\n"
            "error references something that does NOT appear in that code block, do not\n"
            "invent a plausible-sounding root cause for it — set that fix's root_cause to\n"
            "\"NOT FOUND IN PROVIDED CODE\" and exact_fix to \"unknown\" instead of guessing.\n\n"
            f"{regression_hint}"
            'Respond with JSON: {"fixes": [{"error": str, "root_cause": str, '
            '"exact_fix": str, "priority": int, "confidence": int(0-10)}], "summary": str, '
            '"missing_includes": [str], "wrong_apis": [{"cuda": str, "hip": str}]}.'
        )
        return prompt

    # ── Main routing logic ──────────────────────────────────────

    def route(self, kernel_source: str, patterns: List[Dict],
              max_iterations: int = 10,
              on_phase=None,
              verifier=None,
              kernel_name: str = "test_kernel",
              max_seconds: Optional[float] = None,
              fast_path: bool = True,
              debug: Optional[bool] = None,
              debug_session=None) -> Dict:
        """Route kernel through the loop engineering pipeline.

        Loop: DeepSeek (plan) → GLM-5.2 (code) → [hipcc compile FIRST] → Kimi K2.7 (evaluate only if compile passes) → feedback → GLM refines

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
            max_seconds: Hard wall-clock budget. Defaults to MAX_PIPELINE_SECONDS.
                Pass 0 (or a negative value) to disable the budget entirely.
            fast_path: Try the mechanical hipify → compile → run shortcut before
                any LLM call. Set False to exercise the model loop directly.
            debug: Enable Phase 11 Debug Mode for this call. None (default)
                defers to the KERNEL_OLYMPICS_DEBUG / KERNEL_DEBUG_MODE
                environment variables, or to the value passed to __init__.
            debug_session: An existing DebugSession to record into. Pass one when
                the caller's unit of work is larger than a route() — main.py's
                pipeline also runs an authoritative verify() compile after this
                returns, and that compile belongs in the same session directory.
                Ownership follows creation: a session passed in here is NOT
                finalized by route(), because the caller has more to record.

        Returns:
            {"ported_code": ..., "confidence": ..., "changes": [...],
             "model_used": ..., "cost": ..., "orchestrator_passed": ...,
             "iterations_used": ..., "compile_errors": [...]}
        """
        # Ownership rule: whoever creates the session finalizes it. route()
        # creates one only when the caller did not supply it, and finalizes only
        # what it created — otherwise a summary is written before the caller has
        # finished recording, and every later artifact is missing from it.
        if debug_session is not None:
            self.debug = debug_session
            self._owns_debug_session = False
        else:
            want_debug = debug if debug is not None else self._debug_requested
            self.debug = DebugSession.create(kernel_name, enabled=want_debug)
            self._owns_debug_session = True

        if self.debug.enabled:
            if self._owns_debug_session:
                print(f"║  │  🐞 DEBUG MODE: session → {str(self.debug.dir)[:38]:<38}║")
            # Give the verifier the same session so the compiler stage records
            # its argv, environment and untruncated output into this directory.
            if verifier is not None and hasattr(verifier, "attach_debug_session"):
                verifier.attach_debug_session(self.debug)

        try:
            return self._route_impl(kernel_source, patterns, max_iterations,
                                    on_phase, verifier, kernel_name, max_seconds,
                                    fast_path)
        except BaseException as exc:
            # "Whenever execution terminates unexpectedly, automatically generate
            # a failure package." A KeyboardInterrupt is exactly such a
            # termination — and the one most likely to strand a long run — so we
            # catch BaseException, snapshot, and re-raise untouched. The snapshot
            # is written even for a borrowed session: the caller may never get
            # the chance to finalize it.
            self.debug.snapshot_failure(exc, reason=f"route() raised {type(exc).__name__}")
            if self._owns_debug_session:
                self.debug.finalize({"abort_reason": "exception"})
            raise
        finally:
            # A borrowed session stays attached to the verifier — the caller is
            # about to use it. One we created is done the moment route() returns.
            if (self._owns_debug_session and verifier is not None
                    and hasattr(verifier, "detach_debug_session")):
                verifier.detach_debug_session()

    def _route_impl(self, kernel_source: str, patterns: List[Dict],
                    max_iterations: int = 10,
                    on_phase=None,
                    verifier=None,
                    kernel_name: str = "test_kernel",
                    max_seconds: Optional[float] = None,
                    fast_path: bool = True) -> Dict:
        """The pipeline proper. See :meth:`route` for the contract."""
        if not self.api_key:
            no_key = {"ported_code": "", "confidence": 0,
                      "changes": ["No API key -- use template fallback"],
                      "model_used": "none", "cost": 0,
                      "orchestrator_passed": False, "iterations_used": 0,
                      "compile_errors": [], "compile_passed": False,
                      "best_attempt_code": "", "best_attempt_iteration": 0,
                      "best_attempt_confidence": 0.0,
                      "prompt_version": PROMPT_VERSION, "timed_out": False,
                      "fast_path_used": False, "hipify_transforms": 0}
            self.debug.transition("ABORTED", reason="no API key configured",
                                  validation_result=False)
            self._finalize_debug(no_key)
            return no_key

        # P0: start the wall-clock budget before the first LLM call. _call_model
        # reads self._deadline and clamps every request timeout to what is left.
        budget = MAX_PIPELINE_SECONDS if max_seconds is None else max_seconds
        deadline = Deadline(budget)
        self._deadline = deadline

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0,
                  "orchestrator_passed": False, "iterations_used": 0,
                  "compile_errors": [], "compile_passed": False,
                  "compile_error_history": [],
                  "best_attempt_code": "", "best_attempt_iteration": 0,
                  "best_attempt_confidence": 0.0,
                  "prompt_version": PROMPT_VERSION, "timed_out": False,
                  "fast_path_used": False, "hipify_transforms": 0}
        if not deadline.unlimited:
            result["changes"].append(
                f"[budget] Wall-clock limit {budget:.0f}s (prompt {PROMPT_VERSION})")

        # S3: best attempt tracking — code from the iteration that compiled the
        # furthest (highest iteration where compile passed). Used for caching on
        # verification failure so re-runs start from closest-working version.
        best_attempt_code = ""
        best_attempt_iteration = 0

        # A2A protocol: unique run ID for structured message full_ref keys
        # I3: Use UUID for reproducibility tracking and create run directory
        run_id = str(uuid.uuid4())[:8]
        self.run_id = f"{kernel_name}_{run_id}"
        run_dir = Path(f"runs/{self.run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        result["run_id"] = self.run_id

        # ── Determine port mode (WHOLE_PROGRAM vs DEVICE_SUBSET) ──
        # The spec's port_mode is the authoritative source for the harness
        # decision. If a hand-written spec already exists with port_mode set,
        # trust it; otherwise compute one from the CUDA source.
        #
        # BUG FIX: this previously ignored the spec entirely and used only the
        # computed heuristic — so nvidia_shfl_scan.json's hand-tagged
        # DEVICE_SUBSET was silently discarded, the porter was told to reproduce
        # the whole unportable NVIDIA sample, and the run timed out. Consult the
        # spec first (the "authoritative override" the comment always claimed),
        # then fall back to the source heuristic.
        port_mode = ModelRouter._compute_port_mode(kernel_source)
        spec_mode = ModelRouter._spec_port_mode(kernel_name, verifier)
        if spec_mode is not None:
            port_mode = spec_mode
        result["port_mode"] = port_mode.value

        # ── TRIZ #13 / #24: Auto-generate spec from CUDA source ──
        # Before ANY LLM call or compile check, parse the original CUDA source
        # for __global__ kernel signatures and generate a spec JSON file.
        # This ensures the verifier builds the correct harness on the first try,
        # instead of guessing (float*, float*, int) and failing with harness errors.
        # The spec auto-gen loop converges when the spec matches the kernel's
        # real signature — regex handles 90%+ of cases, LLM fallback covers the rest.
        try:
            _generated_spec = _auto_gen_spec(kernel_name, kernel_source)
            if _generated_spec is None:
                result["changes"].append(
                    f"[spec] No __global__ kernel found in source — skipping auto-generation")
            elif _generated_spec.get("_persisted", True) is False:
                # Bug 5: a hand-written spec already exists (no auto_generated
                # marker) — save_spec() refused to overwrite it. Using the
                # existing spec as-is rather than silently destroying it.
                result["changes"].append(
                    f"[spec] Skipped writing specs/{kernel_name}.json — a hand-written "
                    f"spec already exists there; using it as-is")
            else:
                result["changes"].append(f"[spec] Auto-generated specs/{kernel_name}.json from CUDA source")
        except Exception as e:
            result["changes"].append(f"[spec] Auto-generation failed: {e} — using generic harness fallback")

        # ── Phase 0: mechanical hipify, then a compile-first fast path ──
        # HIPIFY does every deterministic CUDA→HIP transform in ~19ms. The loop
        # was paying an LLM ~90s to reproduce them one token at a time.
        hipified_source, hipify_changelog = self._hipify_source(kernel_source)
        result["hipify_transforms"] = len(hipify_changelog)
        result["changes"].append(
            f"[hipify] {len(hipify_changelog)} mechanical transforms applied before any LLM call")

        # Debug Mode: the input stage is complete only now — "preprocessing
        # results" means the hipified draft, not the raw file, so it is logged
        # after hipify rather than at the top of route().
        self.debug.log_input(
            kernel_source,
            classifier_results={"patterns": patterns, "pattern_count": len(patterns or [])},
            patterns=patterns,
            preprocessed_source=hipified_source,
            preprocessing_changelog=hipify_changelog,
            run_id=self.run_id,
            max_iterations=max_iterations,
            budget_seconds=budget,
            prompt_version=PROMPT_VERSION,
            fast_path_requested=fast_path,
        )
        self.debug.transition("INPUT_RECEIVED",
                              reason=f"{len(hipify_changelog)} hipify transforms",
                              validation_result=True)

        residual = self._residual_cuda_symbols(hipified_source)
        if residual:
            result["changes"].append(
                f"[hipify] {len(residual)} CUDA symbols have no deterministic HIP spelling "
                f"({', '.join(residual[:5])}) — the coder is required")

        # Don't spend a compile discovering what the source already says. A kernel
        # using __shfl/warpSize needs lane arithmetic no regex can supply.
        needs_semantics = self._needs_wavefront_semantics(hipified_source, patterns)
        if needs_semantics:
            result["changes"].append(
                "[fast-path] skipped — kernel uses warp-level primitives whose semantics "
                "change on wavefront64; a mechanical port would compile and then crash")

        if (fast_path and not residual and not needs_semantics
                and verifier and hasattr(verifier, "quick_compile_check")):
            if on_phase: on_phase("hipify", "hipify (regex)", "mechanical CUDA→HIP, no LLM")
            cc = verifier.quick_compile_check(hipified_source, kernel_name=kernel_name)
            # isinstance guard: a bare MagicMock verifier reads as "no information",
            # never as a pass — the fast path must never be taken on a guess.
            if isinstance(cc, dict) and cc.get("compile_success") is True:
                # RUN-FIRST. A compile-pass is not a port: the 2026-07-09 kernel
                # compiled and then SIGSEGVed on shared memory sized blockDim/32.
                # Mechanical translation cannot fix wavefront64 semantics, so the
                # binary must actually run before we skip the models.
                rc = (verifier.quick_run_check(kernel_name)
                      if hasattr(verifier, "quick_run_check") else None)
                if isinstance(rc, dict) and rc.get("run_success") is True:
                    result["ported_code"] = hipified_source
                    result["compile_passed"] = True
                    result["compile_errors"] = []
                    result["orchestrator_passed"] = True
                    result["model_used"] = "hipify"
                    result["confidence"] = 85
                    result["iterations_used"] = 0
                    result["fast_path_used"] = True
                    result["best_attempt_code"] = hipified_source
                    result["best_attempt_confidence"] = 0.85
                    result["regex_changelog"] = hipify_changelog
                    result["cost"] = round(self.total_cost, 4)
                    result["changes"].append(
                        f"[fast-path] hipify output compiled AND ran clean — "
                        f"skipped DeepSeek, Kimi and GLM entirely (0 LLM calls)")
                    print(f"║  │  ⚡ FAST PATH: mechanical port compiled and ran — no LLM needed{'':<3}║")
                    if self.debug.enabled:
                        self.debug.log_static_analysis(hipified_source, generation=0)
                        self.debug.log_symbols(kernel_source, hipified_source, generation=0)
                        self.debug.transition("SUCCESS", reason="fast path: hipify compiled and ran",
                                              validation_result=True)
                        self._finalize_debug(result)
                    return result

                sig = (rc.get("signal") or f"exit {rc.get('run_exit_code')}"
                       ) if isinstance(rc, dict) else "no run information"
                result["changes"].append(
                    f"[fast-path] hipify output compiled but did not run clean ({sig}) — "
                    f"handing the draft to the coder for wavefront64 semantics")
            else:
                n_errs = len(cc.get("errors", [])) if isinstance(cc, dict) else "?"
                result["changes"].append(
                    f"[fast-path] hipify output did not compile ({n_errs} errors) — "
                    f"handing the draft to the coder")

        # Track pipeline phase outcomes for rubric scoring
        planner_success = False
        coder_success = False
        verify_success = False
        verify_passed = False
        evaluator_feedback = ""
        compile_passed = False  # TRIZ #23: track compile state as feedback signal
        deepseek_plan_output = ""

        # ── Phase 1: DeepSeek PLANS the port (reasoning model — prose OK) ──
        # P0: the planner runs on a slice, not on the whole clock. Its own timeout
        # is 120s of a 180s budget; unclamped it starves the coder and every
        # refinement behind it. The plan is advisory — the port is not.
        plan_cap = (None if deadline.unlimited
                    else max(min(PLAN_CAP,
                                 deadline.remaining()
                                 - COMPILE_RESERVE_SECONDS
                                 - REPAIR_RESERVE,
                                 ), 0.0))

        plan_skipped = plan_cap is not None and plan_cap < MIN_LLM_TIMEOUT_SECONDS
        if plan_skipped:
            # Planning at all would eat the coder's reservation. Skip it outright
            # rather than spend the clock discovering that.
            plan = AgentResult("deepseek", False, "", 0.0)
            result["changes"].append(
                f"[deepseek] Planning SKIPPED — a {budget:.0f}s budget leaves no slice for "
                f"a plan once the coder's share is reserved; porting without one")
            self.debug.transition("PLAN_SKIPPED",
                                  reason=f"budget {budget:.0f}s leaves no plan slice",
                                  validation_result=None)
        else:
            self._debug_stage = "02_planning"
            if on_phase: on_phase("plan", "DeepSeek-v4-pro", "planning CUDA→HIP strategy")
            # Hand the planner the mechanical draft so it plans the wavefront delta
            # rather than re-deriving the translation. Only when hipify actually did
            # something: on a source it could not transform, the draft carries no
            # information the original doesn't, and the full prompt is the honest one.
            delta_plan = bool(hipify_changelog)
            ds_prompt = self._build_deepseek_plan_prompt(
                kernel_source, patterns,
                hipified_source=(hipified_source if delta_plan else ""))
            # A delta plan is a checklist, not an essay. Output tokens are this
            # phase's latency, and 2048 of them is what a 38s plan is made of.
            plan = self._call_model("deepseek", ds_prompt,
                                    system_prompt=SYSTEM_PROMPTS.get("deepseek", ""),
                                    max_seconds=plan_cap,
                                    max_tokens_override=(PLAN_DELTA_MAX_TOKENS
                                                         if delta_plan else None))
            self._debug_stage = ""
            # I3: Log model I/O for reproducibility
            try:
                (run_dir / "phase1_plan_input.json").write_text(
                    json.dumps({"prompt": ds_prompt[:5000]}, indent=2), encoding="utf-8")
                (run_dir / "phase1_plan_output.json").write_text(
                    json.dumps({"model": "deepseek", "success": plan.success,
                                "output": plan.output[:5000]}, indent=2), encoding="utf-8")
            except (OSError, TypeError, ValueError) as _log_exc:
                logger.debug("run-dir logging skipped: %s", _log_exc)

            # Debug Mode: the plan is prose by design, so "parsed plan" here is
            # the checklist we can recover from it, and the validation report
            # states plainly that a plan is advisory — it never gates the port.
            if self.debug.enabled:
                self.debug.log_planning(
                    raw_response=plan.output,
                    extracted_plan=plan.output,
                    parsed_plan={"lines": [l for l in plan.output.splitlines() if l.strip()][:40]},
                    validation={
                        "advisory_only": True,
                        "non_empty": bool(plan.output.strip()),
                        "gate": "none — a failed plan never blocks the coder",
                    },
                    discarded="",
                    tokens=plan.tokens_used, latency_ms=plan.elapsed_ms,
                    model="deepseek", success=plan.success,
                    delta_plan=delta_plan, cap_seconds=plan_cap,
                )
                self.debug.transition(
                    "PLAN_GENERATED" if plan.success else "PLAN_FAILED",
                    reason=("plan produced" if plan.success else "planner call failed"),
                    validation_result=plan.success)

        if plan.success:
            planner_success = True
            deepseek_plan_output = plan.output
            cap_note = f", cap {plan_cap:.0f}s" if plan_cap is not None else ""
            result["changes"].append(
                f"[deepseek] Plan generated ({len(plan.output)} chars{cap_note})")
        elif not plan_skipped:
            # A planner that overran its slice is not a model failure. Name which it was,
            # so a slow endpoint is not mistaken for a broken one.
            if plan_cap is not None:
                result["changes"].append(
                    f"[deepseek] Planning did not finish within its {plan_cap:.0f}s slice "
                    f"— proceeding without plan; the coder's budget is intact")
            else:
                result["changes"].append("[deepseek] Planning FAILED — proceeding without plan")

        # ── Phase 2: GLM codes the initial port ──
        if on_phase: on_phase("code", "GLM-5.2", "generating HIP port from plan")
        # TRIZ #24 / Bug 1: self-contained-program detection and the "preserve
        # main()" instruction now both live inside _build_kimi_code_prompt
        # itself (which also stops truncating the source out from under that
        # main() — see docs/fix-plan-self-contained-programs.md).
        kimi_prompt = self._build_kimi_code_prompt(kernel_source, patterns,
                                                   deepseek_plan=deepseek_plan_output,
                                                   preprocessed_source=hipified_source,
                                                   port_mode=result.get("port_mode"))
        # P0: the coder may use at most CODEGEN_CAP seconds, less the compile
        # reserve (always protected). The REPAIR_RESERVE is not subtracted here
        # because no compile has happened yet — it's only needed once the repair
        # cycle begins after a failed compile.
        code_cap = (None if deadline.unlimited
                    else max(min(CODEGEN_CAP,
                                 deadline.remaining()
                                 - COMPILE_RESERVE_SECONDS,
                                 ), 0.0))
        # Output tokens are the coder's latency. Size the budget to the kernel.
        adaptive_tokens = self._compute_adaptive_max_tokens(hipified_source or kernel_source)
        self._debug_stage = "03_translation"
        code = self._call_model("glm", kimi_prompt,
                                system_prompt=SYSTEM_PROMPTS.get("glm", ""),
                                max_seconds=code_cap,
                                max_tokens_override=adaptive_tokens)
        self._debug_stage = ""
        self.debug.transition(
            "CODE_GENERATED" if code.success else "CODE_GENERATION_FAILED",
            reason=f"glm initial port ({len(code.output)} chars)",
            validation_result=code.success, iteration=0)
        # I3: Log GLM code generation
        try:
            (run_dir / "phase2_kimi_output.json").write_text(
                json.dumps({"model": "kimi27", "success": code.success,
                            "output": code.output[:5000]}, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as _log_exc:
            # I3 reproducibility logging must never take down a port. json.dumps
            # raises TypeError on a non-serializable value, which an OSError-only
            # guard let escape all the way out of route().
            logger.debug("run-dir logging skipped: %s", _log_exc)
        # TRIZ #23/#22/#3: baseline for the loop's error-delta and new-error tracking.
        # Seeded from the pre-loop compile check below (if it fails) so iteration 1's
        # delta is measured against real prior state instead of a phantom zero.
        prev_error_count = 0
        prev_errors_set = set()
        prev_errors_norm = set()  # TRIZ #3: normalized baseline for semantic diffing
        if code.success:
            coder_success = True
            # P0: the coder dropping main() on a self-contained program cost the
            # 2026-07-09 run its whole budget. Restore it here, before the first
            # compile, so no LLM phase ever sees the resulting link error.
            extracted, regex_changelog, main_restored, structural = self._postprocess_port(
                code.output, kernel_source, iteration=0, model="glm",
                tokens=code.tokens_used, latency_ms=code.elapsed_ms,
                port_mode=result.get("port_mode"))
            if main_restored:
                result["changes"].append(
                    "[main] Coder dropped main() from a self-contained program — "
                    "restored it from the original CUDA source (no LLM call)")
                print(f"║  │  🔧 MAIN RESTORED: coder dropped the driver — reattached{'':<7}║")
            else:
                # A declined restore is a decision, not an absence of one. Surface it.
                for _note in regex_changelog:
                    if _note.startswith("main() NOT restored"):
                        result["changes"].append(f"[main] {_note}")
                        print(f"║  │  ⚠ MAIN NOT RESTORED: driver needs code this port "
                              f"dropped{'':<6}║")
            result["ported_code"] = extracted
            result["regex_changelog"] = regex_changelog  # A3: track regex fixes
            result["structural"] = {
                "ok": structural.ok,
                "reason": structural.reason(),
                "missing_symbols": list(structural.missing_symbols),
                "warnings": list(structural.warnings),
                "errors": list(structural.errors),
            }
            # ── Pre-hipcc structural gate ──
            # A 60s hipcc call on an unbalanced brace, a truncation marker, or a
            # gutted file only teaches the compiler what a text-level check would
            # have caught in <1ms. When the gate rejects, we skip the compile and
            # let the loop feed the structural report to Kimi as targeted feedback
            # on the first refine — the budget saved here is what buys the extra
            # repair iterations F1's whole point is to enable.
            if not structural.ok:
                for _err in structural.errors:
                    result["changes"].append(f"[structural] REJECT: {_err}")
                print(f"║  │  🧱 STRUCTURAL REJECT: {structural.reason()[:38]:<38}║")
                self.debug.event("structural_reject", reason=structural.reason(),
                                 iteration=0, hipcc_skipped=True)
                evaluator_feedback = (
                    "STRUCTURAL VALIDATION FAILED — your last output cannot be "
                    "compiled because it is not valid C++ at the text level.\n"
                    + "; ".join(structural.errors)
                    + (f"\nSymbols missing vs the original: "
                       + ", ".join(structural.missing_symbols)
                       if structural.missing_symbols else "")
                    + "\nRe-emit the COMPLETE ported HIP kernel with balanced braces "
                      "and no truncation markers. Preserve every function present in "
                      "the original CUDA source."
                )
                # Skip the pre-loop hipcc — the refine loop below will pick up
                # this feedback and drive the next Kimi call.
                result["compile_errors"] = [
                    f"[structural] {e}" for e in structural.errors]
                result["compile_error_history"].append(
                    {"iteration": 0, "errors": list(result["compile_errors"])})
                prev_error_count = len(result["compile_errors"])
                prev_errors_set = set(result["compile_errors"])
                prev_errors_norm = prev_errors_set
                # Fall through past the initial hipcc block into the refine loop;
                # the two unconditional lines below still fire.
            elif verifier and hasattr(verifier, 'quick_compile_check'):
                if on_phase: on_phase("compile", "hipcc", "in-loop compilation check")
                with self.debug.stage("hipcc"):
                    cc = verifier.quick_compile_check(extracted, kernel_name=kernel_name)
                self.debug.transition(
                    "HIPCC_COMPILE",
                    reason=("compile passed" if cc["compile_success"]
                            else f"{len(cc.get('errors', []))} compile errors"),
                    validation_result=cc["compile_success"], iteration=0)
                if cc["compile_success"]:
                    result["changes"].append("[hipcc] In-loop compile: PASSED ✅")
                    compile_passed = True
                    # S3: save best attempt — code that compiled
                    best_attempt_code = extracted
                    best_attempt_iteration = 0
                else:
                    self.debug.count("compile_failures")
                    compile_errs = cc.get("errors", [])
                    # Bug 4: assign (not extend) — compile_errors always reflects the
                    # LATEST check's state, not an ever-growing accumulation of every
                    # error seen across every iteration. Full history is kept separately
                    # in compile_error_history for reporting.
                    result["compile_errors"] = list(compile_errs)
                    result["compile_error_history"].append({"iteration": 0, "errors": list(compile_errs)})
                    prev_error_count = len(compile_errs)  # TRIZ #23: baseline for prompt evolution
                    prev_errors_set = set(e.strip() for e in compile_errs if e.strip())  # TRIZ #22: baseline
                    prev_errors_norm = set(self._normalize_error(e) for e in compile_errs if e.strip())  # TRIZ #3: baseline
                    err_summary = "; ".join(compile_errs[:3]) if compile_errs else cc["compile_output"][:300]
                    result["changes"].append(f"[hipcc] In-loop compile FAILED: {err_summary[:120]}")
                    # Feed compile errors to GLM evaluator as additional feedback
                    # A2A protocol: structure ALL errors, not just first 3
                    all_errs = compile_errs if compile_errs else [cc["compile_output"][:300]]
                    # Check if the code was truncated. Read the RAW model output, not
                    # the extracted source: _sanitize_extracted deletes the marker so
                    # it can never reach hipcc, and asking the compiled file whether
                    # it still carries a marker we just removed always answers "no".
                    is_truncated = "TRUNCATED" in code.output
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
            # P0: distinguish "the model failed" from "we ran out of clock". The
            # budget can expire during this very first Kimi call, in which case
            # _call_model returns a failed AgentResult and this branch would
            # otherwise report a model failure and hide the real cause.
            # `code_cap` below the call floor is the same condition seen earlier:
            # the clock has not formally run out, but no call could have succeeded.
            starved = code_cap is not None and code_cap < MIN_LLM_TIMEOUT_SECONDS
            if deadline.exhausted() or starved:
                result["timed_out"] = True
                result["abort_reason"] = "pipeline_timeout"
                result["changes"].append(
                    f"[budget] Wall-clock limit {budget:.0f}s reached during initial "
                    f"code generation — no kernel was produced")
            else:
                result["changes"].append("[kimi27] Code generation FAILED")
            # Can't proceed without initial code
            result["cost"] = round(self.total_cost, 4)
            # No kernel was produced — that is an unexpected termination, and
            # the raw (failed) coder response is already on disk. Package it.
            self.debug.snapshot_failure(
                reason=result.get("abort_reason", "initial code generation failed"),
                context={"starved": starved, "code_cap": code_cap})
            self._finalize_debug(result)
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
        # prev_error_count / prev_errors_set / prev_errors_norm are intentionally NOT
        # reset here — they were seeded above from the pre-loop compile check (or
        # remain at their 0/empty defaults if that check passed outright). Resetting
        # them to 0/empty at this point used to make iteration 1's error delta a
        # phantom "-N" against a fake zero baseline, which fired the stagnation
        # detector on iteration 1 even when the port was genuinely improving.
        stagnation_count = 0  # TRIZ #15: count iterations with no improvement
        error_history = []  # TRIZ #17: cap error context to last 2 iterations
        norm_error_history = []  # A5: track normalized error frozensets for cycle detection
        replan_count = 0  # Bug 7: how many stagnation re-plans we've already used
        runtime_crash_count = 0  # RUN-FIRST: consecutive compile-pass-but-crash iterations
        kimi_plateau_count = 0  # C1: count consecutive iterations with same error set after Kimi refine
        # P1 (two-layer SIGSEGV): last code that hipcc accepted. Once set, a refine
        # that breaks compilation is discarded and this is restored (Layer 1).
        frozen_base_code = ""
        frozen_base_iteration = 0
        for iteration in range(1, max_iterations + 1):
            if not result["ported_code"]:
                break
            self.debug.event("iteration_start", reason=f"iteration {iteration}",
                             iteration=iteration,
                             remaining_seconds=(None if deadline.unlimited
                                                else round(deadline.remaining(), 1)))

            # ── P0: wall-clock budget check at the iteration boundary ──
            # Checked here (not mid-call) because an iteration is the unit of work
            # we can abandon cleanly: best_attempt_code already holds the furthest
            # compiling version, so stopping now returns real value instead of Ctrl+C.
            #
            # An iteration costs a compile plus at least one LLM call. Entering one
            # with less than that on the clock spends an uninterruptible hipcc run
            # to produce errors nobody will ever get to act on.
            iteration_floor = COMPILE_RESERVE_SECONDS + MIN_LLM_TIMEOUT_SECONDS
            if deadline.exhausted() or not deadline.has_at_least(iteration_floor):
                result["timed_out"] = True
                result["abort_reason"] = "pipeline_timeout"
                result["iterations_used"] = iteration - 1
                # T0.1: do not call it "the best compiling attempt" when nothing
                # compiled. compile_passed is the only thing that earns that word.
                if compile_passed and best_attempt_code:
                    kept = f"returning the best compiling attempt (iter {best_attempt_iteration})"
                else:
                    kept = ("returning the last attempt — nothing compiled, so it is "
                            "saved for manual hipcc, not served from cache")
                result["changes"].append(
                    f"[budget] Wall-clock limit {budget:.0f}s reached after "
                    f"{deadline.elapsed():.0f}s — stopping at iteration {iteration - 1} "
                    f"and {kept}")
                print(f"║  │  ⏱ TIMEOUT: {budget:.0f}s budget spent — returning best attempt{'':<10}║")
                break

            # ── Step 1: hipcc compile check FIRST ──────────────────────────
            # TRIZ #13: Reverse the order — compile before evaluate.
            # If compile fails, compile errors ARE the feedback. Skip GLM entirely.
            #
            # State object holds this iteration's authoritative outcome. Legacy
            # local names below are kept as mirrors so the ~200 existing readers
            # keep working unchanged, but every branch is required to populate
            # `state` so downstream code has a single, defaulted source of truth
            # (see IterationState docstring for the structural-reject bug this
            # was introduced to fix).
            state = IterationState(iteration=iteration)
            compile_failed_this_iter = False
            run_crashed_this_iter = False  # RUN-FIRST: set by the in-loop run check below
            replanned_this_iter = False
            glm_analysis = None
            linker_only = False
            # Compile-branch locals. Previously bound ONLY inside the compile-fail
            # `else` (~line 2828). After the structural gate landed, the informed
            # re-plan below could reach `compile_errs` on a structural reject —
            # where it was unbound — and crash route() with UnboundLocalError.
            # Defaulting them here is not a bandaid: the structural branch below
            # now fills `compile_errs` with the structural errors so the re-plan
            # gate degrades to a truthful "no compile errors to re-plan against"
            # rather than exploding.
            compile_errs: List[str] = []
            error_origins: List = []
            all_harness_origin = False
            cc: Optional[dict] = None
            # ── Pre-compile structural gate for THIS iteration's code ──
            # The refine that produced result["ported_code"] set result["structural"];
            # if it flagged a hard defect, hipcc will only reproduce parser noise the
            # coder never introduced. Convert the structural report into synthetic
            # compile errors and skip hipcc so the next refine iteration hits.
            iter_structural = result.get("structural")
            if iter_structural and not iter_structural.get("ok", True):
                # Structural gate → first-class failure path. hipcc is NOT run;
                # compile-error consumers (informed re-plan wording, GLM error
                # analyst) will gate on `state.compile_ran` below rather than
                # infer their state from the presence of `compile_errs`.
                state.gate = "structural"
                state.structural_ok = False
                state.structural_reject = True
                state.structural_errors = list(iter_structural.get("errors", []))
                state.structural_missing = list(iter_structural.get("missing_symbols", []))
                compile_failed_this_iter = True
                struct_errs = [f"[structural] {e}"
                               for e in iter_structural.get("errors", [])]
                if iter_structural.get("missing_symbols"):
                    struct_errs.append(
                        "[structural] symbols dropped: "
                        + ", ".join(iter_structural["missing_symbols"][:8]))
                # Mirror the structural errors onto the compile-error slot so
                # readers that key on it (result["compile_error_history"], the
                # PromptOptimizer's evolve_prompt loop, the Kimi refine
                # feedback) see meaningful content, and truthiness reads like
                # `if compile_errs:` remain honest — the iteration DID produce
                # a list of things to fix, just not from hipcc.
                state.compile_errs = list(struct_errs)
                compile_errs = list(struct_errs)
                result["compile_errors"] = struct_errs
                result["compile_error_history"].append(
                    {"iteration": iteration, "errors": list(struct_errs)})
                try:
                    (run_dir / f"iteration_{iteration}_structural.json").write_text(
                        json.dumps({
                            "iteration": iteration,
                            "gate": state.gate,
                            "structural_reject": state.structural_reject,
                            "compile_ran": state.compile_ran,
                            "compile_success": False,
                            "structural": iter_structural,
                            "ported_code_preview": (result["ported_code"] or "")[:2000],
                        }, indent=2), encoding="utf-8")
                except (OSError, TypeError, ValueError) as _log_exc:
                    logger.debug("structural log skipped: %s", _log_exc)
                result["changes"].append(
                    f"[structural] Iter {iteration}: rejected before hipcc — "
                    + "; ".join(iter_structural.get("errors", [])[:2])[:120])
                print(f"║  │  🧱 STRUCTURAL GATE (iter {iteration}): hipcc skipped "
                      f"— see structural feedback{'':<3}║")
                self.debug.transition(
                    "STRUCTURAL_REJECT",
                    reason="; ".join(iter_structural.get("errors", []))[:160],
                    validation_result=False, iteration=iteration, hipcc_skipped=True)
                self.debug.event("structural_reject", iteration=iteration,
                                 reason="hipcc skipped — text-level defect")
                evaluator_feedback = (
                    "STRUCTURAL VALIDATION FAILED (iter {}). hipcc was NOT run — "
                    "the code has defects that would only produce parser noise.\n"
                    "Errors: {}\n"
                    "Missing symbols vs original: {}\n"
                    "Re-emit the COMPLETE port with balanced braces, no truncation "
                    "markers, and every function from the original CUDA present."
                ).format(
                    iteration,
                    "; ".join(iter_structural.get("errors", [])),
                    ", ".join(iter_structural.get("missing_symbols", [])) or "(none reported)",
                )
                # Fall through past the hipcc block into the refine step.
            elif verifier and hasattr(verifier, 'quick_compile_check'):
                state.gate = "compile"
                state.compile_ran = True
                if on_phase: on_phase("compile", "hipcc", f"compile-first check (attempt {iteration}/{max_iterations})")
                with self.debug.stage("hipcc"):
                    cc = verifier.quick_compile_check(result["ported_code"], kernel_name=kernel_name)
                self.debug.transition(
                    "HIPCC_COMPILE",
                    reason=("compile passed" if cc["compile_success"]
                            else f"{len(cc.get('errors', []))} compile errors"),
                    validation_result=cc["compile_success"], iteration=iteration)
                # I3: Log iteration compile result for reproducibility.
                # Now includes the structural score and dropped symbols so a failing
                # iteration is diagnosable from its JSON alone.
                try:
                    (run_dir / f"iteration_{iteration}_compile.json").write_text(
                        json.dumps({"iteration": iteration,
                                    "compile_success": cc["compile_success"],
                                    "errors": cc.get("errors", [])[:8],
                                    "structural": iter_structural or {},
                                    "ported_code_preview": (result["ported_code"] or "")[:2000]},
                                   indent=2), encoding="utf-8")
                except (OSError, TypeError, ValueError) as _log_exc:
                    logger.debug("run-dir logging skipped: %s", _log_exc)
                if cc["compile_success"]:
                    state.compile_success = True
                    # Compile passed — release the repair reserve so the
                    # verification phase can draw on it.
                    self._repair_released = True
                    result["changes"].append(
                        f"[hipcc] Compile-first check {iteration}: PASSED ✅")
                    result["compile_errors"] = []
                    compile_passed = True
                    # S3: save best attempt — highest iteration with passing compile
                    if iteration > best_attempt_iteration:
                        best_attempt_code = result["ported_code"]
                        best_attempt_iteration = iteration
                        result["changes"].append(
                            f"[best-attempt] Saved iter {iteration} code as best attempt")

                    # P1 (two-layer): this code compiles. Freeze it as Layer 1 so a
                    # later refine that breaks the build can be rolled back to here.
                    frozen_base_code = result["ported_code"]
                    frozen_base_iteration = iteration

                    # ── RUN-FIRST: compile-pass is not convergence — run the binary ──
                    # The compile check just linked a real executable into the loop
                    # build dir; running it costs ~1s and is the highest-authority
                    # oracle available. The 2026-07-09 run declared victory here,
                    # then verify() found a SIGSEGV with no feedback path back —
                    # while GLM's semantic eval had ALREADY flagged the likely
                    # cause (__shfl_up_sync width) and the loop discarded it.
                    # isinstance guard: mocked verifiers in tests return MagicMock,
                    # which must read as "no run info" (pass), not crash.
                    if verifier and hasattr(verifier, 'quick_run_check'):
                        with self.debug.stage("run"):
                            rc = verifier.quick_run_check(kernel_name)
                        if isinstance(rc, dict):
                            self.debug.write_json(
                                "09_compiler", f"run_iter{iteration}", rc)
                            self.debug.transition(
                                "BINARY_RUN",
                                reason=("ran clean" if rc.get("run_success")
                                        else f"crashed: {rc.get('signal') or rc.get('run_exit_code')}"),
                                validation_result=rc.get("run_success"),
                                iteration=iteration)
                        if isinstance(rc, dict) and rc.get("run_success") is True:
                            runtime_crash_count = 0  # consecutive-crash counter
                            result["changes"].append(
                                f"[run] Iter {iteration}: binary ran clean (exit 0)")
                        if isinstance(rc, dict) and rc.get("run_success") is False:
                            state.gate = "run"
                            state.run_crashed = True
                            run_crashed_this_iter = True
                            runtime_crash_count += 1
                            self.debug.count("runtime_crashes")
                            sig = rc.get("signal") or f"exit {rc.get('run_exit_code')}"
                            crash_output = (rc.get("run_output") or "").strip()
                            result["changes"].append(
                                f"[run] Iter {iteration}: compiled but CRASHED at runtime "
                                f"({sig}) — crash {runtime_crash_count}/3")
                            print(f"║  │  💥 RUNTIME CRASH: {sig} — compiled code is not working code{'':<8}║")
                            # Feed the crash to GLM's semantic eval (Step 2 below
                            # passes evaluator_feedback into the eval prompt) and,
                            # through GLM's issues, to Kimi's refine.
                            evaluator_feedback = (
                                f"⚠️ RUNTIME CRASH (iteration {iteration}): the code COMPILED but the "
                                f"binary crashed with {sig}.\n"
                                + (f"Program output before crash:\n{crash_output[:500]}\n"
                                   if crash_output else
                                   "No output was captured — the process died before flushing stdout, "
                                   "so the crash is likely early (host-side setup, first kernel launch, "
                                   "or an out-of-bounds access in shared-memory indexing sized for "
                                   "warp 32 vs wavefront 64).\n")
                                + "Compilation is NOT the goal — a working binary is. Find the runtime "
                                  "defect (common: __shfl width/mask semantics on wavefront64, shared "
                                  "memory sized as blockDim/32 vs blockDim/64, out-of-bounds indexing, "
                                  "ignored allocation failures) and fix it.\n"
                            )
                    if run_crashed_this_iter and runtime_crash_count >= 3:
                        result["changes"].append(
                            f"[run] 3 consecutive runtime crashes — aborting; keeping "
                            f"best compiling attempt (iter {best_attempt_iteration})")
                        print(f"║  │  🛑 ABORT: 3 runtime crashes — refinement is not converging{'':<12}║")
                        result["iterations_used"] = iteration
                        result["abort_reason"] = "runtime_stagnation"
                        break
                else:
                    state.compile_success = False
                    compile_failed_this_iter = True
                    compile_errs = cc.get("errors", [])
                    error_origins = cc.get("error_origins", [])
                    state.compile_errs = list(compile_errs)
                    state.error_origins = list(error_origins)
                    self.debug.count("compile_failures")
                    # P0: a link failure carries no information any model can act on.
                    # The missing-main case is already gone by here — _postprocess_port
                    # reattaches the driver before the first compile — so what remains
                    # is an undefined symbol the coder must inline. The planner and the
                    # analyst are skipped for it (see the guards below); on 2026-07-09
                    # they cost 38.2s + 12.9s + 38.1s to say nothing.
                    linker_only = self._is_linker_only(compile_errs, error_origins)
                    state.linker_only = linker_only

                    # ── P1 (two-layer SIGSEGV): reject a regressing Layer 2 ──
                    # We had code that compiled (Layer 1), asked Kimi to fix a runtime
                    # crash, and got back code that no longer builds. That is exactly
                    # the 2026-07-09 regression: iteration 4 (1 error) → iteration 5
                    # (5 errors) because the crash "fix" rewrote the whole kernel.
                    # A non-compiling kernel is strictly worse than a compiling one
                    # that crashes, so discard Layer 2 and stop — the caller gets
                    # Layer 1 back via the recovery block after the loop.
                    if frozen_base_code and result["ported_code"] != frozen_base_code:
                        result["changes"].append(
                            f"[two-layer] Refine at iter {iteration} broke a build that "
                            f"compiled at iter {frozen_base_iteration} "
                            f"({len(compile_errs)} errors) — discarding it and keeping "
                            f"the frozen base kernel")
                        print(f"║  │  🧊 LAYER-2 REJECTED: refine broke the build — reverting{'':<8}║")
                        result["ported_code"] = frozen_base_code
                        result["abort_reason"] = "layer2_rejected"
                        result["iterations_used"] = iteration
                        break

                    # TRIZ #24 (hidden resource): the exact source lines at each
                    # error location, extracted deterministically from the file
                    # hipcc actually compiled. Without this, every agent reasons
                    # from error STRINGS alone — the 2026-07-09 run burned 6
                    # iterations on errors living in post-processor-injected
                    # shim lines nobody ever looked at.
                    err_ctx_block = ""
                    if cc.get("error_context"):
                        err_ctx_block = (
                            "\nSOURCE AT ERROR LOCATIONS (exact lines from the file hipcc compiled — "
                            "fix THESE lines, do not guess):\n"
                            + "\n---\n".join(cc["error_context"][:6]) + "\n"
                        )
                    # Bug 5: computed deterministically from line numbers in verifier.py
                    # (_classify_error_origin), not guessed by an LLM — an error whose line
                    # falls outside the ported kernel's spliced range in the harness is one
                    # Kimi cannot fix, because it's not in code Kimi ever wrote or saw.
                    all_harness_origin = cc.get("all_harness_origin", False)
                    state.all_harness_origin = all_harness_origin
                    # Bug 6: a missing main() at link time is the single most
                    # actionable signal available here — the spec says this
                    # program is self-contained, and the linker is saying the
                    # ported code doesn't define main(). A raw "undefined
                    # symbol: main" string doesn't tell Kimi what happened;
                    # spell it out instead of letting it guess.
                    #
                    # Reaching this means the mechanical restore could not run (the
                    # original's main() would not extract), so the coder is the last
                    # resort — hence the explicit instruction in the feedback below.
                    main_link_error = self._is_missing_main_error(
                        compile_errs, error_origins)
                    # Bug 4: assign, don't accumulate — see note at the pre-loop check above.
                    result["compile_errors"] = list(compile_errs)
                    result["compile_error_history"].append({"iteration": iteration, "errors": list(compile_errs)})

                    # ── Bug 5, trigger 1: harness-origin errors are unfixable by Kimi ──
                    # No refinement iteration can fix a line that isn't in the code Kimi
                    # wrote — that's the root finding of the nvidia_shfl_scan failure
                    # (docs/fix-plan-harness-and-diagnostics.md). Abort immediately rather
                    # than burning the remaining iteration budget on guaranteed repeats.
                    if all_harness_origin and compile_errs:
                        result["changes"].append(
                            f"[hipcc] Iter {iteration}: all {len(compile_errs)} error(s) originate "
                            f"in harness/driver code, not the ported kernel — Kimi cannot fix "
                            f"these. Aborting early instead of repeating {max_iterations - iteration} "
                            f"more guaranteed-identical iterations.")
                        print(f"║  │  🛑 ABORT: errors are in the test harness, not the ported "
                              f"kernel — see spec coverage{'':<8}║")
                        result["iterations_used"] = iteration
                        result["abort_reason"] = "harness_origin"
                        break

                    # ── Semantic Translation Repair Engine (deterministic, pre-LLM) ──
                    # Before spending a GLM analysis + DeepSeek re-plan + Kimi refine
                    # cycle on these errors, try to recover the lost semantic
                    # information from the ORIGINAL CUDA source. A dropped macro,
                    # __device__ helper or struct is restored with a minimal additive
                    # edit — no model call, fully deterministic. Guarded to the errors
                    # that live in the ported kernel (harness-origin and linker-only
                    # cases are handled/aborted above and cannot be fixed by restoring
                    # a symbol into the port). Only adopted when it strictly reduces
                    # the error count, so it can never make the build worse.
                    if (compile_errs and not linker_only
                            and not state.structural_reject):
                        if on_phase:
                            on_phase("repair", "semantic-repair",
                                     f"restoring dropped symbols from CUDA (iter {iteration})")
                        repaired = self._attempt_semantic_repair(
                            kernel_source, result["ported_code"], compile_errs,
                            verifier, kernel_name, iteration)
                        if repaired is not None:
                            patched_code, repaired_cc = repaired
                            n_fixed = len(compile_errs) - len(repaired_cc.get("errors", []))
                            result["ported_code"] = patched_code
                            result["changes"].append(
                                f"[semantic-repair] Iter {iteration}: restored dropped "
                                f"symbol(s) from original CUDA source — "
                                f"errors {len(compile_errs)}→{len(repaired_cc.get('errors', []))} "
                                f"(deterministic, no LLM)")
                            print(f"║  │  🩹 SEMANTIC REPAIR: recovered {n_fixed} error(s) "
                                  f"from CUDA source{'':<12}║")
                            # Re-enter the loop: the top recompiles the patched code
                            # and takes the success branch (or surfaces the remaining,
                            # now-fewer errors for the next repair/LLM pass).
                            continue

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
                    result["changes"].append(
                        f"[hipcc] Iter {iteration}: {current_err_count} errors "
                        f"(delta: {error_delta:+d}, new: {len(new_errors_norm)}, "
                        f"resolved: {len(resolved_errors)})")
                    self.debug.event(
                        "compile_errors", iteration=iteration,
                        reason=f"{current_err_count} errors (delta {error_delta:+d})",
                        error_count=current_err_count, error_delta=error_delta,
                        new_errors=sorted(new_errors_norm),
                        resolved_errors=sorted(resolved_errors))

                    # LIVE VISIBILITY: Print error details during loop, not after.
                    # verifier._compile() already strips the temp build-dir path prefix
                    # from these lines, so [:58] now shows "file:line:col: error: msg"
                    # instead of the path alone (the original bug here truncated at the
                    # exact character the path ended and the message began).
                    top_errs = compile_errs[:2] if compile_errs else ["(no error lines parsed)"]
                    for err_line in top_errs:
                        clean = err_line.strip()[:58]
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

                    # ── C1: Kimi plateau detection ──
                    # Same normalized error set for 2+ consecutive iterations means
                    # Kimi changed the code but produced identical errors. Switch
                    # to shim-injection mode (extern int declarations).
                    prev_norm_frozen = norm_error_history[-2] if len(norm_error_history) >= 2 else None
                    if prev_norm_frozen is not None and current_norm_frozen == prev_norm_frozen:
                        kimi_plateau_count += 1
                        result["changes"].append(
                            f"[kimi-plateau] Same error set persists after Kimi refine "
                            f"(kimi_plateau_count={kimi_plateau_count})")
                        print(f"║  │  🗻 KIMI PLATEAU: same errors after refine "
                              f"(count={kimi_plateau_count}){'':<26}║")

                        if kimi_plateau_count >= 2:
                            # Extract undeclared identifier names from error messages
                            undeclared_ids = set()
                            for err in compile_errs:
                                matches = re.findall(
                                    r"error: use of undeclared identifier ['\"](\w+)['\"]",
                                    err)
                                undeclared_ids.update(matches)

                            if undeclared_ids:
                                # Inject extern int declarations at top of ported code
                                shim_lines = "\n".join(
                                    f"extern int {name};" for name in sorted(undeclared_ids))
                                _code_before_shim = result["ported_code"]
                                result["ported_code"] = shim_lines + "\n" + result["ported_code"]
                                # A shim injection is a patch the pipeline authored
                                # itself, not one a model proposed. It is recorded
                                # the same way so the patch history is complete.
                                self.debug.log_patch(
                                    before=_code_before_shim, after=result["ported_code"],
                                    iteration=iteration, source_label="shim_injection",
                                    rationale=f"extern int shims for undeclared: "
                                              f"{', '.join(sorted(undeclared_ids))}",
                                    confidence=None)
                                result["changes"].append(
                                    f"[kimi-plateau] Injected extern int shims for "
                                    f"{len(undeclared_ids)} undeclared identifiers: "
                                    f"{', '.join(sorted(undeclared_ids))}")
                                print(f"║  │  💉 SHIM INJECTION: extern int for "
                                      f"{len(undeclared_ids)} undeclared IDs{'':<27}║")

                                # Re-compile one more time (no more Kimi)
                                if verifier and hasattr(verifier, 'quick_compile_check'):
                                    with self.debug.stage("hipcc"):
                                        cc_retry = verifier.quick_compile_check(
                                            result["ported_code"], kernel_name=kernel_name)
                                    self.debug.transition(
                                        "HIPCC_COMPILE",
                                        reason="recompile after shim injection",
                                        validation_result=cc_retry.get("compile_success", False),
                                        iteration=iteration)
                                    if cc_retry.get("compile_success", False):
                                        result["compile_errors"] = []
                                        compile_passed = True
                                        result["changes"].append(
                                            f"[kimi-plateau] Shim injection fixed it "
                                            f"— compile PASSED ✅")
                                        print(f"║  │  ✅ SHIM FIX: compile passed after "
                                              f"extern int injection{'':<16}║")
                                        break
                                    else:
                                        result["changes"].append(
                                            f"[kimi-plateau] Shim injection did NOT fix "
                                            f"compile — aborting")
                                        print(f"║  │  ❌ SHIM FAILED: extern int injection "
                                              f"didn't fix — aborting{'':<10}║")
                                        result["iterations_used"] = iteration
                                        result["abort_reason"] = "kimi_plateau"
                                        break
                            else:
                                # No undeclared identifiers found — just abort
                                result["changes"].append(
                                    f"[kimi-plateau] No undeclared identifiers "
                                    f"to shim — aborting")
                                print(f"║  │  ❌ KIMI PLATEAU: no undeclared IDs "
                                      f"to shim — aborting{'':<13}║")
                                result["iterations_used"] = iteration
                                result["abort_reason"] = "kimi_plateau"
                                break
                    else:
                        # Error set changed after Kimi refine — reset plateau counter
                        kimi_plateau_count = 0

                    # ── TRIZ #13/#24: GLM→DeepSeek escalation BEFORE next Kimi refine ──
                    # Whenever the loop is stuck (no improvement), route back through
                    # DeepSeek to regenerate the CUDA→HIP strategy. The new plan
                    # replaces deepseek_plan_output so the next Kimi refine uses a
                    # different strategy — fixes "stuck" runs where Kimi faithfully
                    # reproduces the same broken output under the same plan.
                    #
                    # Triggers (any is sufficient):
                    #   (a) kimi_plateau_count >= 1 → normalized error set literally
                    #       identical across iterations (rigid plateau)
                    #   (b) stagnation_count >= 2  → 2 iterations without improvement
                    #       (looser plateau — catches the 1-error-rotated case above)
                    # `not linker_only`: a re-plan cannot resolve a link failure. The
                    # strategy was never the problem — a symbol is simply absent.
                    if ((kimi_plateau_count >= 1 or stagnation_count >= 2)
                            and replan_count == 0
                            and not linker_only
                            and iteration < max_iterations):
                        trigger = ("kimi_plateau=" + str(kimi_plateau_count)
                                   if kimi_plateau_count >= 1
                                   else "stagnation=" + str(stagnation_count))
                        result["changes"].append(
                            f"[orch] GLM→DeepSeek escalation triggered ({trigger}). "
                            f"Re-planning CUDA→HIP strategy with GLM error context."
                        )
                        print(f"║  │  🔄 GLM→DeepSeek: re-plan with error context "
                              f"({trigger}){'':<10}║")
                        if on_phase: on_phase("plan", "DeepSeek-v4-pro",
                            f"re-planning from GLM ({trigger})")
                        replan_prompt = self._build_deepseek_replan_prompt(
                            kernel_source, patterns, result["ported_code"],
                            compile_errs, deepseek_plan_output,
                        )
                        self._debug_stage = "02_planning"
                        re_plan = self._call_model(
                            "deepseek", replan_prompt,
                            system_prompt=SYSTEM_PROMPTS.get("deepseek", ""),
                        )
                        self._debug_stage = ""
                        replan_count += 1
                        replanned_this_iter = True
                        self.debug.count("replans")
                        self.debug.transition(
                            "REPLAN", reason=f"escalation ({trigger})",
                            validation_result=re_plan.success, iteration=iteration,
                            replan_count=replan_count)
                        if re_plan.success:
                            deepseek_plan_output = re_plan.output
                            result["changes"].append(
                                f"[deepseek] Re-plan from GLM escalation landed: "
                                f"fresh strategy (iter {iteration}, replan_count={replan_count})"
                            )
                            # Don't reset stagnation_count/kimi_plateau_count —
                            # let hard-stagnation abort still fire if even the
                            # new strategy fails.

                    # Re-plan after GLM analysis is now further down (after the
                    # GLM error block). Earlier placement (here) made the visual
                    # order DeepSeek > Kimi > DeepSeek > GLM, but the spec is
                    # DeepSeek > Kimi > GLM > DeepSeek > Kimi.

                    # ── Bug 5, trigger 2 / Bug 7: hard stagnation abort ──
                    # After multiple re-plans failed to converge, more iterations won't help.
                    # P0: was `stagnation_count >= 3 and replan_count >= max_iterations // 2`
                    # (i.e. 5 re-plans at the default max_iterations=10), which took ~6
                    # iterations to reach — the run was Ctrl+C'd long before it could fire.
                    if (stagnation_count >= STAGNATION_ABORT_THRESHOLD
                            and replan_count >= MAX_REPLANS):
                        result["changes"].append(
                            f"[hipcc] Hard stagnation: {stagnation_count} iterations with no "
                            f"improvement after {replan_count} DeepSeek re-plans already failed "
                            f"to help. Aborting — another re-plan is unlikely to succeed where "
                            f"those didn't.")
                        print(f"║  │  🛑 ABORT: fresh strategy also stagnated — giving up{'':<20}║")
                        result["iterations_used"] = iteration
                        result["abort_reason"] = "hard_stagnation"
                        break

                    # TRIZ #15: After the first stagnant iterations, escalate to a DeepSeek
                    # re-plan — but only once (see hard-stagnation abort above for what
                    # happens next).
                    if (stagnation_count >= STAGNATION_ABORT_THRESHOLD
                            and replan_count == 0 and not linker_only
                            and iteration < max_iterations):
                        result["changes"].append(
                            f"[hipcc] Stagnation detected ({stagnation_count} iterations no improvement) "
                            f"— escalating to DeepSeek re-plan")
                        print(f"║  │  🔄 STAGNATION: {stagnation_count} iters no improvement — re-planning{'':<24}║")
                        if on_phase: on_phase("plan", "DeepSeek-v4-pro",
                            f"re-planning due to stagnation (iter {iteration})")
                        replan_prompt = self._build_deepseek_replan_prompt(
                            kernel_source, patterns, result["ported_code"],
                            compile_errs, deepseek_plan_output,
                        )
                        self._debug_stage = "02_planning"
                        re_plan = self._call_model(
                            "deepseek", replan_prompt,
                            system_prompt=SYSTEM_PROMPTS.get("deepseek", ""),
                        )
                        self._debug_stage = ""
                        replan_count += 1
                        replanned_this_iter = True
                        self.debug.count("replans")
                        self.debug.transition(
                            "REPLAN", reason=f"stagnation={stagnation_count}",
                            validation_result=re_plan.success, iteration=iteration,
                            replan_count=replan_count)
                        if re_plan.success:
                            deepseek_plan_output = re_plan.output
                            result["changes"].append(
                                f"[deepseek] Re-plan generated (stagnation recovery, "
                                f"fresh strategy vs. previous plan)")
                            # Bug 7: deliberately NOT resetting stagnation_count to 0 here.
                            # The old behavior erased the evidence the loop was stuck,
                            # letting it re-plan indefinitely. Leaving the counter running
                            # means the hard-stagnation abort above can actually fire once
                            # this fresh strategy also fails to show improvement.
                        # Keep the same compile errors for Kimi, but with fresh plan

                    err_summary = "; ".join(compile_errs[:3]) if compile_errs else cc["compile_output"][:300]
                    result["changes"].append(
                        f"[hipcc] Compile-first check {iteration}: FAILED: {err_summary[:120]}")

                    if linker_only:
                        result["changes"].append(
                            f"[linker] Iter {iteration}: all {len(compile_errs)} diagnostics are "
                            f"link failures — skipped the GLM analyst and both DeepSeek re-plans "
                            f"(neither can supply a missing symbol)")
                        print(f"║  │  🔗 LINK-ONLY: skipping GLM + DeepSeek — "
                              f"nothing to plan{'':<12}║")

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
                        + err_ctx_block
                        + "\n\nFocus on:\n"
                        "- Missing HIP API calls (cuda* not converted to hip*)\n"
                        "- Undefined functions/macros (checkCudaErrors, etc.)\n"
                        "- Type mismatches (hipError_t vs cudaError_t)\n"
                        "- Missing or wrong #include directives\n"
                        + (f"- ⚠️ Stagnation: {stagnation_count} iterations without improvement — try a DIFFERENT approach\n"
                           if stagnation_count > 0 else "")
                        + ("\n⚠️ LINKER ERROR: your ported_code is missing a main() function. "
                           "The original CUDA source is a complete, self-contained program with "
                           "its own main() — you MUST include a full main() with identical logic, "
                           "ported to HIP APIs. Do not drop it.\n"
                           if main_link_error else "")
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
                    #
                    # Guarded by `not all_harness_origin`: if every error's line number
                    # falls outside the ported kernel's range in the harness, the errors
                    # are not in code Kimi wrote — asking GLM to explain them produces a
                    # confident, plausible, WRONG root cause (it cannot say "this isn't in
                    # the code you gave me" because it was never asked to consider that).
                    # See docs/fix-plan-harness-and-diagnostics.md, "Regression introduced
                    # by a0d2bc5". Bug 5 aborts the loop entirely in this case (below);
                    # skipping the analyst call here avoids paying for it either way.
                    #
                    # `not linker_only` for the same reason: the analyst is asked to
                    # find a root cause in the code it is shown, and a link failure's
                    # root cause is a symbol that is not there. It answers anyway.
                    if (compile_errs and iteration < max_iterations
                            and not all_harness_origin and not linker_only):
                        if on_phase: on_phase("analyze", "Kimi K2.7",
                            f"analyzing compile errors for Kimi (iter {iteration})")
                        print(f"║  │  🔍 GLM analyzing {len(compile_errs)} compile errors for Kimi{'':<30}║")
                        glm_err_prompt = self._build_glm_error_analysis_prompt(
                            result["ported_code"], compile_errs, iteration, patterns,
                            error_delta=error_delta, stagnation_count=stagnation_count,
                            error_context=cc.get("error_context"),
                            self_contained=self._is_self_contained(kernel_source),
                            port_mode=result.get("port_mode"))
                        self._debug_stage = "10_evaluation"
                        glm_err = self._call_model(
                            "kimi27", glm_err_prompt,
                            system_prompt=SYSTEM_PROMPTS.get("kimi_error_analyst", ""),
                            prefill='{"fixes":'  # TRIZ #9: force JSON
                        )
                        self._debug_stage = ""
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

                            # Debug Mode: the raw response is already on disk via
                            # _call_model; this records what the four parse
                            # strategies made of it — the step where a confident
                            # analysis silently becomes an empty one.
                            if self.debug.enabled:
                                self.debug.log_evaluation(
                                    raw_response="",  # already persisted by _call_model
                                    parsed=glm_analysis, model="glm",
                                    iteration=iteration, mode="error_analysis",
                                    root_cause=[f.get("root_cause") for f in
                                                (glm_analysis or {}).get("fixes", [])],
                                    recommended_fixes=(glm_analysis or {}).get("fixes", []),
                                    confidence=(glm_analysis or {}).get("confidence"),
                                    parse_strategy=("failed" if glm_analysis is None
                                                    else "one of 4 JSON strategies"),
                                    compile_error_count=len(compile_errs),
                                )

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

            # ── TRIZ #13 + #24: GLM → DeepSeek informed re-plan, AFTER GLM analysis ──
            # User-specified loop: DeepSeek(plan) > Kimi(code) > GLM(analyze) >
            # DeepSeek(re-plan informed by GLM) > Kimi(refine under fresh plan).
            #
            # The block fires AFTER glm_analysis is computed so we can include it in
            # the re-plan prompt. Earlier blocks elsewhere fired BEFORE GLM and
            # gave DeepSeek raw compile errors only — visual order was broken and
            # the re-plan prompt was uninformed.
            #
            # Triggers: compile failed AND we have re-plan budget left AND no
            # re-plan already ran this iteration AND no hard-stagnation abort.
            #
            # P0: this used to fire on EVERY failing iteration (~80s of DeepSeek each,
            # up to max_iterations//2 = 5 times), and on a stagnant iteration it fired
            # a *second* time right after the escalation above — two plans, one used.
            # Capping at MAX_REPLANS and skipping when replanned_this_iter is what
            # takes a stagnating run from ~3min/iteration to roughly one Kimi call.
            # `not state.structural_reject`: DeepSeek's job is to re-plan a
            # CUDA→HIP porting strategy against the compiler's semantic errors.
            # A structural reject means the model dropped a brace, truncated the
            # file, or left a "// ... rest of code" marker — a text-level defect
            # that no re-plan can address. Kimi already has the structural
            # feedback via evaluator_feedback; re-planning here spends ~80s of
            # DeepSeek time for a prompt whose "RECURRING COMPILE ERRORS"
            # section would just re-echo "[structural] unbalanced braces".
            #
            # Historical crash: before this guard, `compile_errs` was unbound
            # on structural reject (see IterationState docstring) and the
            # `and compile_errs` truthiness check on the next line raised
            # UnboundLocalError at router.py:3352.
            if (compile_failed_this_iter
                    and not state.structural_reject
                    and not replanned_this_iter
                    and not linker_only
                    and replan_count < MAX_REPLANS
                    and iteration < max_iterations
                    and compile_errs
                    and result.get("abort_reason") != "hard_stagnation"):
                glm_summary = ""
                if glm_analysis is not None:
                    try:
                        glm_summary = (
                            f"\nGLM ANALYST SAYS ({len(compile_errs)} errors):\n"
                            f"{json.dumps(glm_analysis, indent=2)[:1500]}\n"
                        )
                    except (TypeError, ValueError) as e:
                        logger.debug("Failed to dump glm_analysis: %s", e)
                replan_prompt = (
                    f"You produced a CUDA→HIP plan earlier that yielded code still failing to compile.\n"
                    f"This is re-plan attempt {replan_count + 1}/{MAX_REPLANS}.\n"
                    f"{glm_summary}\n"
                    f"RECURRING COMPILE ERRORS:\n{chr(10).join(compile_errs[:8])}\n\n"
                    f"LAST HIP CODE:\n```hip\n{result.get('ported_code', '')[:3000]}\n```\n\n"
                    f"ORIGINAL CUDA:\n```cuda\n{kernel_source[:3000]}\n```\n\n"
                    "Produce a DIFFERENT porting strategy from the previous plan. "
                    "Be specific and concrete, not mechanical substitution."
                )
                result["changes"].append(
                    f"[orch] GLM→DeepSeek informed re-plan "
                    f"({replan_count + 1}/{MAX_REPLANS}, iter {iteration})"
                )
                print(f"║  │  🔄 GLM→DeepSeek: informed re-plan "
                      f"({replan_count + 1}/{MAX_REPLANS}){'':<14}║")
                if on_phase: on_phase("replan", "DeepSeek-v4-pro",
                    f"informed re-plan after GLM (iter {iteration})")
                self._debug_stage = "02_planning"
                re_plan = self._call_model(
                    "deepseek", replan_prompt,
                    system_prompt=SYSTEM_PROMPTS.get("deepseek", ""),
                )
                self._debug_stage = ""
                replan_count += 1
                self.debug.count("replans")
                self.debug.transition(
                    "REPLAN", reason=f"GLM-informed re-plan {replan_count}/{MAX_REPLANS}",
                    validation_result=re_plan.success, iteration=iteration,
                    replan_count=replan_count)
                if re_plan.success:
                    deepseek_plan_output = re_plan.output
                    result["changes"].append(
                        f"[deepseek] Informed re-plan landed "
                        f"(iter {iteration}, {len(re_plan.output)} chars)"
                    )
                    print(f"║  │  🧠 DeepSeek-v4-pro re-plan landed "
                          f"({len(re_plan.output)} chars){'':<14}║")

            result["iterations_used"] = iteration
            # Snapshot the iteration's authoritative outcome so tests and any
            # post-mortem inspection can see which gate produced the final
            # verdict without having to re-derive it from side effects.
            state.glm_analysis = glm_analysis
            state.replanned = replanned_this_iter
            result["last_iteration_state"] = {
                "iteration": state.iteration,
                "gate": state.gate,
                "structural_reject": state.structural_reject,
                "structural_errors": list(state.structural_errors),
                "structural_missing": list(state.structural_missing),
                "compile_ran": state.compile_ran,
                "compile_success": state.compile_success,
                "compile_errs_count": len(state.compile_errs),
                "linker_only": state.linker_only,
                "run_crashed": state.run_crashed,
                "replanned": state.replanned,
                "repair_mode": state.repair_mode,
            }

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
                if on_phase: on_phase("evaluate", "Kimi K2.7", f"semantic eval (attempt {iteration}/{max_iterations}, compile passed)")
                result["changes"].append(
                    f"[kimi27] Evaluating code (attempt {iteration}/{max_iterations}, compile passed)")
                self._debug_stage = "10_evaluation"
                evaluator = self._call_model(
                    "kimi27", eval_prompt,
                    system_prompt=SYSTEM_PROMPTS.get("kimi27", ""),
                    prefill='{"pass":'  # TRIZ #9: force JSON start, prevent prose
                )
                self._debug_stage = ""

                if not evaluator.success:
                    result["changes"].append(
                        f"[glm] Call failed (iteration {iteration})")
                    self.debug.transition("EVALUATION_FAILED",
                                          reason="GLM call failed",
                                          validation_result=False, iteration=iteration)
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

                if self.debug.enabled:
                    self.debug.log_evaluation(
                        raw_response="",  # already persisted by _call_model
                        parsed=parsed, model="glm", iteration=iteration,
                        mode="semantic_eval",
                        root_cause=(parsed or {}).get("verdict"),
                        recommended_fixes=(parsed or {}).get("issues", []),
                        confidence=(parsed or {}).get("confidence"),
                        parse_strategy=("failed — all 4 strategies" if parsed is None
                                        else "one of 4 JSON strategies"),
                    )
                    self.debug.transition(
                        "SEMANTIC_EVALUATION",
                        reason=("unparseable response" if parsed is None
                                else ("pass" if parsed.get("pass") else "issues found")),
                        validation_result=(None if parsed is None else parsed.get("pass")),
                        iteration=iteration)

                if parsed is None:
                    # ── TRIZ #20: Continuation of useful action ──
                    # Don't break the loop on parse failure. Extract whatever
                    # feedback we can from the raw response and continue refining.
                    self.debug.count("evaluator_parse_failures")
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
            # Converged when: compile passed AND the binary RAN AND GLM passed.
            # RUN-FIRST: a compile-pass + GLM-pass on a binary that SIGSEGVs is
            # not convergence — the 2026-07-09 run shipped exactly that.
            if parsed is not None and parsed.get("pass", False) and not run_crashed_this_iter:
                verify_success = True
                verify_passed = True
                result["changes"].append(
                    f"[glm] Passed semantic evaluation (iteration {iteration})")
                # Compile already passed (we only ran GLM because it did),
                # or there's no verifier (GLM pass is sufficient in that case).
                result["orchestrator_passed"] = True
                self.debug.transition(
                    "SUCCESS", reason="compile + run + semantic evaluation all passed",
                    validation_result=True, iteration=iteration)
                break  # Truly converged — compile + run + GLM all passed

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

            # ── TRIZ #20 (Continuation): Only refine when compile FAILED ──
            # If compile passed, the binary RAN, and GLM found only semantic
            # issues — LOG them but do NOT refine: a running binary is the
            # goal, and refining working code risks regression.
            #
            # CAVEAT on the old "proven: refine always regresses" claim: that
            # proof came from the run where _fix_ported_code was CORRUPTING
            # every refinement (shim before hip include, #define 64 64) —
            # the regression was the fixer's, not refinement's. The policy
            # survives, but only for code that actually RUNS.
            #
            # RUN-FIRST exception: if the binary crashed, this is NOT working
            # code — GLM's semantic findings (which flagged __shfl_up_sync
            # width before the 2026-07-09 segfault, and were discarded here)
            # plus the crash info become the refine feedback instead.
            if not compile_failed_this_iter and parsed is not None and not run_crashed_this_iter:
                if not parsed.get("pass", False):
                    result["changes"].append(
                        f"[glm] Iteration {iteration}: semantic issues found but "
                        f"compile passed and binary ran — keeping working code, not refining")
                # Break: compiled + ran + GLM ran = done. Don't refine.
                break

            # Runtime crash: merge GLM's semantic findings into the crash
            # feedback so Kimi refines against BOTH signals.
            if run_crashed_this_iter and parsed is not None:
                glm_issues = parsed.get("issues", [])
                glm_fb = parsed.get("feedback", "")
                if glm_issues or glm_fb:
                    evaluator_feedback += (
                        "\nGLM SEMANTIC FINDINGS on the crashing code (treat these as "
                        "likely crash causes):\n"
                        + "\n".join(f"- {i}" for i in glm_issues[:5])
                        + (f"\n{glm_fb[:400]}" if glm_fb else "")
                    )

            if iteration < max_iterations:
                # Loop back: Kimi refines with feedback
                # ── Honest logging: gate on whether hipcc actually ran ──
                # compile_failed_this_iter can be True even when hipcc never
                # ran (structural/lexical gate).  The local variable name is
                # the historical artifact; state.gate / state.structural_reject
                # are the authoritative source.  Check the structural errors
                # for "[lexical]" prefix to distinguish structural from lexical
                # reject (lexical failures are folded into the structural result
                # by _postprocess_port).
                iter_s = result.get("structural", {}) or {}
                iter_errs = iter_s.get("errors", []) if isinstance(iter_s, dict) else []
                has_lexical = any(
                    e.startswith("[lexical]") for e in iter_errs
                )
                if state.structural_reject:
                    feedback_label = (
                        "lexical feedback" if has_lexical else "structural feedback"
                    )
                elif state.compile_ran and compile_failed_this_iter:
                    feedback_label = "compiler diagnostics"
                elif run_crashed_this_iter:
                    feedback_label = "runtime crash + GLM findings"
                else:
                    feedback_label = "GLM feedback"

                # Set repair_mode for stage-contract tracking
                if state.structural_reject:
                    state.repair_mode = "lexical" if has_lexical else "structural"
                elif state.compile_ran and compile_failed_this_iter:
                    state.repair_mode = "compiler"
                elif run_crashed_this_iter:
                    state.repair_mode = "compiler"  # same pipeline as compile
                else:
                    state.repair_mode = None

                # ── P2: budget-aware dispatch ──
                # The iteration-boundary check above ran BEFORE this iteration's
                # compile, GLM analysis and re-plan spent their share. By the time we
                # reach the refine, the clock may hold less than a Kimi call needs —
                # on 2026-07-09 one started with ~31s left, was killed mid-flight by
                # the deadline, and returned nothing. A call that cannot finish is
                # strictly worse than no call: it costs the clock and yields no code.
                refine_cap = (None if deadline.unlimited
                              else max(deadline.remaining() - COMPILE_RESERVE_SECONDS, 0.0))
                if refine_cap is not None and refine_cap < MIN_LLM_TIMEOUT_SECONDS:
                    result["timed_out"] = True
                    result["abort_reason"] = "pipeline_timeout"
                    result["iterations_used"] = iteration
                    result["changes"].append(
                        f"[budget] {deadline.remaining():.0f}s left after iteration "
                        f"{iteration} — not enough for a refine plus the {COMPILE_RESERVE_SECONDS}s "
                        f"compile reserve. Skipping the call rather than starting one the "
                        f"deadline would kill in flight.")
                    print(f"║  │  ⏱ BUDGET: too little left to refine — stopping cleanly{'':<8}║")
                    break

                if on_phase: on_phase("refine", "GLM-5.2", f"refining with {feedback_label} [mode={state.repair_mode}] (iter {iteration}→{iteration+1})")
                # TRIZ #15: Evolve prompt based on compile error patterns
                evolved = opt.evolve_prompt(result.get("compile_errors", []))
                # version_id already carries its own "v" ("v2"), and this is the
                # PromptOptimizer's checklist version — not router.PROMPT_VERSION.
                result["changes"].append(
                    f"[checklist {evolved.version_id}] Evolved: {len(evolved.checklist)} items")
                refine_prompt = self._build_kimi_refine_prompt(
                    kernel_source, result["ported_code"],
                    evaluator_feedback, patterns,
                    deepseek_plan=deepseek_plan_output,
                    iteration=iteration + 1,
                    checklist_override=evolved.checklist,
                    stagnation_count=stagnation_count,
                    regex_changelog=result.get("regex_changelog"),
                    # P1: only constrain Kimi to patch-on-top when the failure is a
                    # RUNTIME crash on code that compiles. On a compile failure there
                    # is no good baseline to freeze, and a rewrite is what we want.
                    frozen_base_code=(frozen_base_code if run_crashed_this_iter else ""),
                    preprocessed_source=hipified_source,
                    structural_report=result.get("structural"),
                    port_mode=port_mode.value,
                )
                self.debug.transition("PATCH_GENERATION",
                                      reason=f"refine on {feedback_label} (mode={state.repair_mode})",
                                      iteration=iteration)
                self._debug_stage = "03_translation"
                refine = self._call_model(
                    "glm", refine_prompt,
                    system_prompt=SYSTEM_PROMPTS.get("glm", ""),
                    max_seconds=refine_cap,
                    max_tokens_override=adaptive_tokens,
                )
                self._debug_stage = ""
                if refine.success:
                    code_before_refine = result["ported_code"]
                    extracted, regex_changelog, main_restored, structural = self._postprocess_port(
                        refine.output, kernel_source, iteration=iteration,
                        model="kimi27", tokens=refine.tokens_used,
                        latency_ms=refine.elapsed_ms,
                        port_mode=result.get("port_mode"))
                    # Every repair iteration is stored separately: the diff between
                    # what the loop had and what the coder returned, computed from
                    # the two texts rather than taken from the model's word for it.
                    self.debug.log_patch(
                        before=code_before_refine, after=extracted,
                        iteration=iteration, source_label="refine",
                        rationale=f"refined against {feedback_label}",
                        confidence=refine.confidence,
                        structural_ok_after=structural.ok)
                    self.debug.transition(
                        "PATCH_APPLICATION",
                        reason=(f"refine applied (iter {iteration})" if structural.ok else
                                f"refine applied but failed validation (iter {iteration}) "
                                f"— the next iteration's gate will reject it"),
                        validation_result=structural.ok, iteration=iteration)
                    if main_restored:
                        result["changes"].append(
                            f"[main] Refine at iter {iteration} dropped main() — "
                            f"restored from the original CUDA source")
                    result["ported_code"] = extracted
                    result["regex_changelog"] = regex_changelog  # A3: track regex fixes
                    result["structural"] = {
                        "ok": structural.ok,
                        "reason": structural.reason(),
                        "missing_symbols": list(structural.missing_symbols),
                        "warnings": list(structural.warnings),
                        "errors": list(structural.errors),
                    }
                    if not structural.ok:
                        # A refine that broke the file structurally is the same
                        # failure mode as the initial-port reject above: hipcc
                        # will fail with parser noise the model never introduced.
                        # Flag it so the next iteration's compile-fail branch
                        # treats it as a structural regression rather than a
                        # real error to chase.
                        for _err in structural.errors:
                            result["changes"].append(
                                f"[structural] refine at iter {iteration}: {_err}")
                        print(f"║  │  🧱 STRUCTURAL REJECT (refine): "
                              f"{structural.reason()[:31]:<31}║")
                    result["changes"].append(
                        f"[kimi27] Refined with {feedback_label} "
                        f"(iteration {iteration} → {iteration + 1})")
                else:
                    # P2: the refine that just failed consumed clock. A 1.5x-timeout
                    # retry issued with nothing left is the in-flight kill again, one
                    # call later — re-read the deadline instead of trusting refine_cap.
                    retry_cap = (None if deadline.unlimited
                                 else max(deadline.remaining() - COMPILE_RESERVE_SECONDS, 0.0))
                    if retry_cap is not None and retry_cap < MIN_LLM_TIMEOUT_SECONDS:
                        result["timed_out"] = True
                        result["abort_reason"] = "pipeline_timeout"
                        result["iterations_used"] = iteration
                        result["changes"].append(
                            f"[glm] Refinement failed (iteration {iteration}) and "
                            f"{deadline.remaining():.0f}s remain — no room to retry. Keeping "
                            f"the previous code.")
                        break
                    result["changes"].append(
                        f"[glm] Refinement failed (iteration {iteration}), retrying with 1.5x timeout...")
                    self.debug.count("refine_retries")
                    self.debug.event("refine_retry", iteration=iteration,
                                     reason="refine call failed — 1.5x timeout retry")
                    # S4: Retry once with increased timeout — API failures are transient
                    original_timeout = MODEL_CATALOG["glm"]["timeout"]
                    MODEL_CATALOG["glm"]["timeout"] = int(original_timeout * 1.5)
                    self._debug_stage = "03_translation"
                    retry_refine = self._call_model(
                        "glm", refine_prompt,
                        system_prompt=SYSTEM_PROMPTS.get("glm", ""),
                        max_seconds=retry_cap,
                        max_tokens_override=adaptive_tokens,
                    )
                    self._debug_stage = ""
                    MODEL_CATALOG["kimi27"]["timeout"] = original_timeout
                    if retry_refine.success:
                        code_before_retry = result["ported_code"]
                        extracted, regex_changelog, main_restored, structural = self._postprocess_port(
                            retry_refine.output, kernel_source, iteration=iteration,
                            model="kimi27", tokens=retry_refine.tokens_used,
                            latency_ms=retry_refine.elapsed_ms,
                            port_mode=result.get("port_mode"))
                        self.debug.log_patch(
                            before=code_before_retry, after=extracted,
                            iteration=iteration, source_label="refine_retry",
                            rationale=f"retry after failed refine ({feedback_label})",
                            confidence=retry_refine.confidence,
                            structural_ok_after=structural.ok)
                        if main_restored:
                            result["changes"].append(
                                f"[main] Retry at iter {iteration} dropped main() — "
                                f"restored from the original CUDA source")
                        result["ported_code"] = extracted
                        result["regex_changelog"] = regex_changelog
                        result["structural"] = {
                            "ok": structural.ok,
                            "reason": structural.reason(),
                            "missing_symbols": list(structural.missing_symbols),
                            "warnings": list(structural.warnings),
                            "errors": list(structural.errors),
                        }
                        if not structural.ok:
                            for _err in structural.errors:
                                result["changes"].append(
                                    f"[structural] retry at iter {iteration}: {_err}")
                        result["changes"].append(
                            f"[kimi27] Retry succeeded with {feedback_label} "
                            f"(iteration {iteration} → {iteration + 1})")
                    else:
                        result["changes"].append(
                            f"[kimi27] Retry also failed — keeping previous code "
                            f"(iteration {iteration})")
                    # Don't break — keep previous iteration's ported_code
                    # and let the loop continue with next iteration
            # else: max iterations reached, accept current output

        result["compile_passed"] = compile_passed
        result["prompt_versions"] = opt.get_stats()  # TRIZ #15/#23: prompt evolution summary

        # ── TRIZ #20 (Continuation): explicit handling for natural end of loop ──
        # Before this block, the for-loop exited because range() ended. If compile
        # never passed AND no abort_reason was set, that's the "limit cycle hits the
        # ceiling, return stale code" failure mode. Make it visible: set the reason
        # and print the line so the UI shows the loop didn't bail silently.
        if not compile_passed and "abort_reason" not in result:
            result["abort_reason"] = "max_iterations_exhausted"
            result["iterations_used"] = iteration
            result["changes"].append(
                f"[hipcc] Loop exhausted max_iterations={max_iterations} "
                f"without compile pass — returning best_attempt as fallback."
            )
            print(f"║  │  ⏱ MAX ITERATIONS REACHED ({max_iterations}) "
                  f"— returning best attempt{'':<12}║")
            print(f"║  │  💡 To force more loops: increase --iter or "
                  f"fix the underlying strategy{'':<4}║")

        # ── Phase 4: Gemma 4 final verification ──
        # Gemma 4 here is deliberate, not a fallback for GLM: it is the only entry in
        # MODEL_CATALOG with local_first=True, so this phase runs on the MI300X's own
        # vLLM when one is up and costs nothing. GLM-5.2 remains the in-loop evaluator
        # and error analyst; the two roles never swap. It reuses SYSTEM_PROMPTS["glm"]
        # because the task — "judge this HIP kernel, answer in JSON" — is the same one.
        #
        # When the budget is already spent this call returns a failed AgentResult in
        # ~0s (_call_model checks deadline.exhausted() first), which is why a timed-out
        # run still shows a "final verification 0.0s" line.
        if result["ported_code"]:
            if on_phase: on_phase("verify", "Gemma 4", "final verification")
            gemma_prompt = self._build_glm_evaluate_prompt(
                result["ported_code"], patterns,
                regex_changelog=result.get("regex_changelog"),
            )
            self._debug_stage = "10_evaluation"
            verify = self._call_model("gemma4", gemma_prompt,
                                      system_prompt=SYSTEM_PROMPTS.get("glm", ""))
            self._debug_stage = ""
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
        # P0/P1: if we stopped early (budget exhausted, or a refine broke a build
        # that previously worked), the last ported_code is the broken one. Hand back
        # the furthest-compiling version instead — that is what "return the best
        # compiling code so far" means, and it is what the demo needs to show.
        if (result.get("abort_reason") in ("pipeline_timeout", "layer2_rejected")
                and best_attempt_code
                and result["ported_code"] != best_attempt_code):
            result["ported_code"] = best_attempt_code
            result["compile_passed"] = True
            result["compile_errors"] = []
            result["changes"].append(
                f"[recover] Returned best compiling code (iter {best_attempt_iteration}) "
                f"instead of the last, non-compiling attempt")

        # S3: populate best-attempt result fields for caching.
        # If no attempt ever compiled, fall back to the last ported_code (even if
        # it didn't compile) — something is better than nothing for cold-start.
        if best_attempt_code:
            result["best_attempt_code"] = best_attempt_code
            result["best_attempt_iteration"] = best_attempt_iteration
            # Confidence discount: iteration/max_iterations * 0.85 base.
            # A code that compiled at iter 3/10 gets ~0.25; one that compiled
            # at iter 7/10 gets ~0.59. This keeps best-attempt caches below
            # verified thresholds so real verification can still override.
            ratio = best_attempt_iteration / max(max_iterations, 1)
            result["best_attempt_confidence"] = round(ratio * 0.85, 4)
        elif result["ported_code"]:
            # No iteration ever compiled — still save the last code attempt
            # with a very low confidence so re-runs have a starting point.
            result["best_attempt_code"] = result["ported_code"]
            result["best_attempt_iteration"] = result.get("iterations_used", 0)
            result["best_attempt_confidence"] = 0.10
        result["cost"] = round(self.total_cost, 4)

        # ── Phase 11: close the debug session ──
        # The terminal transition names the outcome, so `state_trace.jsonl` ends
        # with a verdict rather than trailing off after the last compile.
        if self.debug.enabled:
            abort = result.get("abort_reason", "")
            converged = result.get("orchestrator_passed") or result.get("compile_passed")
            self.debug.transition(
                "SUCCESS" if converged and not abort else "FAILURE",
                reason=abort or ("converged" if converged else "did not converge"),
                validation_result=bool(converged))
            # A run that did not reach a compiling kernel is a failure worth
            # packaging, even though route() returned normally rather than
            # raising: "terminates unexpectedly" is about the outcome, not the
            # control flow.
            if not converged:
                self.debug.snapshot_failure(
                    reason=abort or "pipeline finished without a compiling kernel",
                    context={"iterations_used": result.get("iterations_used"),
                             "timed_out": result.get("timed_out"),
                             "compile_errors": result.get("compile_errors", [])})
            summary_path = self._finalize_debug(result)
            result["debug_session_dir"] = str(self.debug.dir)
            if summary_path:
                print(f"║  │  🐞 Debug summary → {str(summary_path)[:42]:<42}║")
        return result

    def _call_model(self, model_key: str, prompt: str,
                    system_prompt: str = "",
                    prefill: str = "",
                    max_seconds: Optional[float] = None,
                    max_tokens_override: Optional[int] = None) -> AgentResult:
        """Call a model and record the complete exchange under Debug Mode.

        Phase 11: this wrapper is the single choke point through which every
        provider response passes, so "no intermediate response is ever lost"
        holds by construction rather than by remembering to log at each call
        site. The raw text is on disk BEFORE any parser, extractor or validator
        touches it — which is precisely the text a post-mortem needs when a
        parse is what went wrong.

        A failed call is recorded too: an empty output with an error string is
        as diagnostic as a successful one, and it is what a timeout looks like.
        """
        if not self.debug.enabled:
            return self._call_model_impl(model_key, prompt, system_prompt,
                                         prefill, max_seconds, max_tokens_override)

        calls_before = len(self.call_log)
        cost_before = self.total_cost
        stage = self._debug_stage or ""
        with self.debug.stage(f"llm:{model_key}"):
            result = self._call_model_impl(model_key, prompt, system_prompt,
                                           prefill, max_seconds, max_tokens_override)

        # The endpoint and any error text live in the call_log entries this call
        # appended; read them back rather than threading a return channel
        # through _call_model_impl's many exit points.
        new_entries = self.call_log[calls_before:]
        endpoint = next((e.get("source", "") for e in reversed(new_entries)
                         if e.get("source")), "")
        error = next((e.get("error", "") for e in reversed(new_entries)
                      if e.get("error")), "")

        self.debug.record_llm_call(
            model=model_key,
            tokens=result.tokens_used,
            cost=self.total_cost - cost_before,
            latency_ms=result.elapsed_ms,
            success=result.success,
            stage=stage,
            raw_response=result.output,
            prompt=prompt,
            system_prompt=system_prompt,
            endpoint=endpoint,
            error=error,
            finish_reason="length" if "TRUNCATED" in (result.output or "") else "",
        )
        return result

    def _call_model_impl(self, model_key: str, prompt: str,
                         system_prompt: str = "",
                         prefill: str = "",
                         max_seconds: Optional[float] = None,
                         max_tokens_override: Optional[int] = None) -> AgentResult:
        model_info = MODEL_CATALOG[model_key]
        model_id = model_info["id"]
        local_first = model_info.get("local_first", False)
        model_timeout = model_info.get("timeout", 90)
        # Output tokens are the coder's real latency. Never raise the catalog
        # ceiling — only lower it to what this particular kernel needs.
        catalog_tokens = model_info.get("max_tokens", 1024)
        max_tokens = (min(catalog_tokens, max_tokens_override)
                      if max_tokens_override else catalog_tokens)
        # A phase cap. The model's own timeout is a ceiling for a call in
        # isolation; this is the ceiling for a call inside a budgeted pipeline.
        if max_seconds is not None:
            model_timeout = min(model_timeout, max_seconds)
        t0 = time.perf_counter()
        # The cap bounds the whole phase, retry included — otherwise a 36s cap
        # becomes 36s + a 72s retry, and the reservation it was protecting is gone.
        phase_end = (t0 + max_seconds) if max_seconds is not None else None

        def _phase_remaining() -> float:
            if phase_end is None:
                return float("inf")
            return phase_end - time.perf_counter()

        # P0: never start a request that cannot finish inside the pipeline budget.
        # Without this a single kimi27 call (180s timeout, retried at 2x = 360s)
        # outlives the whole 180s budget on its own.
        deadline = getattr(self, "_deadline", None) or Deadline(0)
        if deadline.exhausted():
            self.call_log.append({"model": model_key, "error": "pipeline budget exhausted"})
            return AgentResult(model_key, False, "", 0.0, 0,
                               round((time.perf_counter() - t0) * 1000, 1))

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
                        "max_tokens": min(max_tokens, model_info.get("max_tokens", 512)),
                    }).encode()
                    req = urllib.request.Request(
                        "http://localhost:8000/v1/chat/completions",
                        data=data_bytes,
                        headers={"Content-Type": "application/json"}
                    )
                    local_timeout = deadline.clamp_timeout(min(30, _phase_remaining()))
                    if local_timeout < MIN_LLM_TIMEOUT_SECONDS:
                        raise PipelineTimeoutError(
                            f"{deadline.remaining():.1f}s left — too little for a local call")
                    with urllib.request.urlopen(req, timeout=local_timeout) as resp:
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
                        "max_tokens": max_tokens,
                        "temperature": model_info.get("temperature", 0.2),
                    }
                    # Use json_schema for DeepSeek (strict), json_object for others
                    schema = JSON_SCHEMAS.get(model_key)
                    if schema:
                        payload["response_format"] = schema
                    data_bytes = json.dumps(payload).encode()
                    # TRIZ #11: Retry once with 2x timeout on timeout failure.
                    # P0: both the first attempt and the 2x retry are clamped to the
                    # pipeline's remaining budget, so the retry can no longer turn a
                    # 180s call into a 540s one that outlives the whole run.
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
                            attempt_timeout = deadline.clamp_timeout(
                                min(model_timeout * (attempt + 1), _phase_remaining()))
                            if attempt_timeout < MIN_LLM_TIMEOUT_SECONDS:
                                raise PipelineTimeoutError(
                                    f"{deadline.remaining():.1f}s left — too little for a "
                                    f"{model_key} call")
                            with urllib.request.urlopen(req, timeout=attempt_timeout) as resp:
                                raw = resp.read()
                            break  # success — exit retry loop
                        except urllib.error.URLError as e:
                            # Don't burn the remaining budget — or the phase's slice —
                            # on a retry that cannot fit.
                            if ("timed out" in str(e).lower() and attempt == 0
                                    and not deadline.exhausted()
                                    and _phase_remaining() >= MIN_LLM_TIMEOUT_SECONDS):
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
                    # Truncation detection: if finish_reason is "length", the output was cut off.
                    # The marker MUST be a C++ line comment. It was an HTML comment
                    # (`<!-- ... -->`), and _extract_code's fallback strategies run to the
                    # end of the text, so it was spliced into the source handed to hipcc.
                    # At file scope `<!--` is not C++: clang reports "expected external
                    # declaration" at the `<` (col 1) and "expected unqualified-id" at the
                    # `--` (col 3) — two errors on one line that no model can fix, because
                    # the model never wrote them. route() still detects it via
                    # `"TRUNCATED" in extracted`, which this spelling preserves.
                    if finish_reason == "length":
                        content += "\n// TRUNCATED: output hit max_tokens limit"
                    # TRIZ #9: Prepend prefill to content — the API returns
                    # only the continuation, we need the full string for parsing
                    if prefill:
                        content = prefill + content
                    cost = tokens / 1000 * model_info["cost_per_1k"]
                    self.total_cost += cost
                    self.call_log.append({"model": model_key, "tokens": tokens, "cost": cost})
                    return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                       tokens, round((time.perf_counter()-t0)*1000, 1))
            except PipelineTimeoutError as e:
                # Budget is gone. Falling through to the next endpoint would just
                # issue another doomed request — stop and let route() return the
                # best compiling attempt it already has.
                self.call_log.append({"model": model_key, "error": str(e)[:200]})
                return AgentResult(model_key, False, "", 0.0, 0,
                                   round((time.perf_counter() - t0) * 1000, 1))
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
                        with urllib.request.urlopen(
                                req, timeout=max(deadline.clamp_timeout(model_timeout),
                                                 MIN_LLM_TIMEOUT_SECONDS)) as resp:
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
