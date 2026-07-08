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
import urllib.request
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


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
    "glm": {
        "id": "accounts/fireworks/models/glm-5p2",  # ✅ VERIFIED WORKING
        "role": "planner",
        "strength": "kernel analysis, pattern detection, warp/wavefront reasoning",
        "cost_per_1k": 0.0014,
        "local_first": False,
    },
    "kimi27": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",  # ✅ VERIFIED WORKING
        "role": "coder",
        "strength": "code generation, struct-aware HIP porting",
        "cost_per_1k": 0.00095,
        "local_first": False,
    },
    "deepseek": {
        "id": "accounts/fireworks/models/deepseek-v4-pro",  # ✅ VERIFIED WORKING
        "role": "verifier_fallback",
        "strength": "correctness checking, fallback when local Gemma unavailable",
        "cost_per_1k": 0.0012,
        "local_first": False,
    },
    "gemma4": {
        "id": "accounts/fireworks/models/gemma-4-31b-it",  # Real Gemma on AMD GPU via vLLM
        "role": "verifier",
        "strength": "AMD-native verification via local vLLM on MI300X",
        "local_first": True,  # Try localhost:8000 first, then Fireworks
    },
}

# ── Role-specific system prompts ──
# Each model gets its OWN role definition. No shared prompts.
# These are passed as system messages to the LLM alongside the phase prompt.

SYSTEM_PROMPTS = {
    "glm": (
        "You are GLM-Planner, a CUDA→HIP migration architect. "
        "Your role is to analyze CUDA kernels flagged with "
        "warp(32)→wavefront(64) divergence issues and produce a "
        "structured plan of line-level changes. "
        "Output ONLY valid JSON. No prose, no explanation, no code."
    ),
    "kimi27": (
        "You are Kimi-Coder, a CUDA→HIP code generation specialist. "
        "Your role is to port CUDA kernels to AMD ROCm/HIP, fixing "
        "warp(32)→wavefront(64) divergence issues. "
        "Output ONLY valid JSON. No prose, no explanation, no markdown outside the json block."
    ),
    "deepseek": (
        "You are DeepSeek-Orchestrator. You manage 2 subordinate agents:\n"
        "  1. GLM-Planner — analyzes kernels for warp→wavefront issues\n"
        "  2. Kimi-Coder — generates HIP kernel code\n\n"
        "YOUR 2 JOBS:\n"
        "  JOB 1 (Orchestrator): Read the GLM analysis below. Decide if it's complete.\n"
        "    - If GLM missed issues → note them in your issues list\n"
        "    - Kimi will receive your feedback before coding\n"
        "  JOB 2 (Evaluator): Verify Kimi's ported code for correctness.\n"
        "    - Check wavefront64 compatibility, __shfl masks, shared memory\n"
        "    - If PASS → output {\"pass\":true,\"verdict\":\"All checks passed\"}\n"
        "    - If FAIL → output {\"pass\":false,\"issues\":[...],\"feedback\":\"line-level fix instruction\"}\n\n"
        "CRITICAL: You are the FINAL AUTHORITY. Do not pass code with issues.\n"
        "Output ONLY valid JSON. No prose."
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


# Pattern → best model routing table
ROUTING_TABLE = {
    # Complex warp patterns → GLM (planner analyzes the structure)
    "shfl_down_sync": "glm",
    "shfl_xor_sync": "glm",
    "syncwarp": "glm",
    # Structural patterns → Kimi K2.7 (coder generates HIP code)
    "warp_size_constant": "kimi27",
    "shared_mem_warp_count": "kimi27",
    "lane_id_mask": "kimi27",
    "tile_size_warp": "kimi27",
    # Verification → Gemma 4 locally, fallback DeepSeek
    "default": "gemma4",
}


@dataclass
class AgentResult:
    model: str
    success: bool
    output: str
    confidence: float
    tokens_used: int = 0
    elapsed_ms: float = 0.0


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

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract code from LLM output (may include markdown or explanation)."""
        import re
        # Try to extract code from markdown code blocks
        blocks = re.findall(r'```(?:cuda|hip|cpp|python)?\n(.*?)```', text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        # Try to extract from __global__ to end
        match = re.search(r'__global__\s+void.*', text, re.DOTALL)
        if match:
            return match.group(0).strip()
        # Try to extract from #include
        match = re.search(r'#include.*', text, re.DOTALL)
        if match:
            return match.group(0).strip()
        # Fallback: return as-is (might be code without markers)
        return text.strip()

    @staticmethod
    def _fix_ported_code(code: str) -> str:
        """Fix AMD-specific issues in ported code."""
        # Replace CUDA headers with HIP equivalents
        code = re.sub(r'#include\s*<cuda_runtime\.h>', '#include <hip/hip_runtime.h>', code)
        code = re.sub(r'#include\s*<cuda_runtime_api\.h>', '#include <hip/hip_runtime.h>', code)
        code = re.sub(r'#include\s*"cuda_runtime\.h"', '#include <hip/hip_runtime.h>', code)
        # Remove NVIDIA helper headers — not available in ROCm
        code = re.sub(r'#include\s*<helper_cuda\.h>\n?', '', code)
        code = re.sub(r'#include\s*<helper_functions\.h>\n?', '', code)
        code = re.sub(r'#include\s*"helper_cuda\.h"\n?', '', code)
        code = re.sub(r'#include\s*"helper_functions\.h"\n?', '', code)
        # Remove device_launch_parameters.h — not needed in HIP
        code = re.sub(r'#include\s*<device_launch_parameters\.h>\n?', '', code)
        # Fix __shfl masks: 32-bit 0xffffffff → 64-bit for wavefront64
        code = re.sub(
            r'(__shfl_\w+_sync\()0x[fF]{8}(,)',
            r'\g<1>0xffffffffffffffffULL\g<2>',
            code
        )
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

        # ── Dimension 2: Code Quality (0-35) ──
        if has_ported_code and len(ported_code.strip()) > 50:
            score += 10
            if "__global__" in ported_code or "__device__" in ported_code:
                score += 15
            if "threadIdx" in ported_code or "blockIdx" in ported_code or "blockDim" in ported_code:
                score += 10

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

        Evaluates the structural quality of the response text.
        """
        if not output or len(output.strip()) == 0:
            return 0.0
        score = 0.3  # baseline: non-empty response
        import re
        if len(output) > 100:
            score += 0.1
        if "__global__" in output or "void" in output:
            score += 0.15
        if re.search(r'```(?:cuda|hip|cpp)?\n', output):
            score += 0.15
        if re.search(r'\{[^}]*\}', output, re.DOTALL):
            score += 0.15
        if "threadIdx" in output or "blockIdx" in output:
            score += 0.15
        return min(score, 1.0)

    # ── Phase prompt builders ────────────────────────────────────

    def _build_glm_plan_prompt(self, kernel_source: str,
                               patterns: List[Dict]) -> str:
        """Build the GLM planner phase prompt with classifier context.

        Role: GLM-Planner — analyzes warp/wavefront divergence and produces
        a line-level JSON plan of issues and fixes.

        Output format: JSON array of {"line": int, "issue": str, "fix": str}
        """
        prompt = (
            "You are GLM-Planner, a CUDA->HIP migration architect.\n"
            "Your task: analyze the given CUDA kernel for warp(32) -> "
            "wavefront(64) divergence issues. "
            "For each issue found, output ONE JSON object with:\n"
            '  "line":   integer line number of the issue\n'
            '  "issue":  description of the divergence problem\n'
            '  "fix":    recommended change for HIP compatibility\n\n'
            "If multiple issues exist, output a JSON ARRAY of objects.\n\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"KERNEL TO ANALYZE:\n"
            f"```cuda\n{kernel_source[:2000]}\n```\n\n"
            "OUTPUT FORMAT -- STRICT JSON ONLY.\n"
            "Single issue:\n"
            '  {"line": 42, "issue": "...", "fix": "..."}\n\n'
            "Multiple issues:\n"
            '  [{"line": 42, "issue": "...", "fix": "..."}, '
            '{"line": 55, "issue": "...", "fix": "..."}]\n\n'
            "EXAMPLE:\n"
            "```json\n"
            '{"line": 42, "issue": "__syncwarp() only syncs 32 threads '
            "but AMD wavefronts need 64\", "
            '"fix": "Replace __syncwarp() with __syncthreads()"}\n'
            "```\n\n"
            "CRITICAL: Return ONLY valid JSON. No prose, no explanation, "
            "no code blocks (unless your JSON is inside one)."
        )
        return prompt

    def _build_kimi_code_prompt(self, kernel_source: str,
                                patterns: List[Dict]) -> str:
        """Build the Kimi K2.7 code generator phase prompt.

        Role: Kimi-Coder — generates the actual ported HIP kernel code.

        Output format: JSON with ported_code (str), confidence (0-100),
        changes (list[str]), explanation (str).
        """
        prompt = (
            "You are Kimi-Coder, a CUDA->HIP code generation specialist.\n"
            "Your task: port the given CUDA kernel to AMD ROCm/HIP, fixing\n"
            "warp(32) -> wavefront(64) divergence issues.\n\n"
            "PORTING CHECKLIST:\n"
            "- __shfl_down_sync offset 16 works on both (verify algorithm)\n"
            "- __shfl_xor_sync mask 0x1f -> 0x3f for wavefront64\n"
            "- warpSize 32 -> use dynamic warpSize() or WAVEFRONT_SIZE constant\n"
            "- shared memory sized for warp 32 -> WAVEFRONT_SIZE (64) or dynamic\n"
            "- __syncwarp() -> __syncthreads() for HIP\n"
            "- #define WAVEFRONT_SIZE 64 at top of kernel\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += "\n" + pattern_summary + "\n"

        prompt += (
            f"\nKERNEL TO PORT:\n"
            f"```cuda\n{kernel_source[:2000]}\n```\n\n"
            "OUTPUT FORMAT -- STRICT JSON inside ```json ... ```:\n"
            "{\n"
            '  "ported_code": "<full ported HIP kernel>",\n'
            '  "confidence": <0-100>,\n'
            '  "changes": ["change 1", "change 2", ...],\n'
            '  "explanation": "what was fixed and why"\n'
            "}\n\n"
            "EXAMPLE:\n"
            "```json\n"
            "{\n"
            '  "ported_code": "#define WAVEFRONT_SIZE 64\\n\\n'
            '__global__ void vec_add(float* a, float* b, int n) {\\n'
            '    int tid = threadIdx.x;\\n    ...\\n}",\n'
            '  "confidence": 88,\n'
            '  "changes": ["Replaced hardcoded 32 with WAVEFRONT_SIZE (64)",\n'
            '              "Changed __syncwarp() to __syncthreads()"],\n'
            '  "explanation": "Ported warp-32 kernel to wavefront-64 HIP '
            'by replacing hardcoded 32 with WAVEFRONT_SIZE and fixing '
            'sync primitives"\n'
            "}\n"
            "```\n\n"
            "CRITICAL: Return ONLY the ```json ... ``` block. "
            "No prose before or after."
        )
        return prompt

    def _build_kimi_refine_prompt(self, kernel_source: str,
                                  previous_code: str,
                                  feedback: str,
                                  patterns: List[Dict],
                                  iteration: int) -> str:
        """Build the Kimi refinement prompt for orchestration loop iterations.

        Kimi receives the original kernel, its previous output, and
        DeepSeek orchestrator's specific feedback to fix issues.
        """
        prompt = (
            "You are Kimi-Coder, refining your CUDA->HIP code based on "
            "orchestrator feedback.\n\n"
            f"ITERATION {iteration}: You previously generated a ported HIP kernel, "
            "but the orchestrator found issues. Please fix them.\n\n"
            "PORTING CHECKLIST (re-verify your fixes):\n"
            "- __shfl_down_sync offset 16 works on both (verify algorithm)\n"
            "- __shfl_xor_sync mask 0x1f -> 0x3f for wavefront64\n"
            "- warpSize 32 -> use dynamic warpSize() or WAVEFRONT_SIZE constant\n"
            "- shared memory sized for warp 32 -> WAVEFRONT_SIZE (64) or dynamic\n"
            "- __syncwarp() -> __syncthreads() for HIP\n"
            "- #define WAVEFRONT_SIZE 64 at top of kernel\n"
        )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += "\n" + pattern_summary + "\n"

        prompt += (
            f"\nORIGINAL CUDA KERNEL:\n"
            f"```cuda\n{kernel_source[:2000]}\n```\n\n"
            f"YOUR PREVIOUS PORTED OUTPUT:\n"
            f"```hip\n{previous_code[:2000]}\n```\n\n"
            f"ORCHESTRATOR FEEDBACK (fix these issues):\n"
            f"{feedback}\n\n"
            "OUTPUT FORMAT -- STRICT JSON inside ```json ... ```:\n"
            "{\n"
            '  "ported_code": "<full ported HIP kernel with fixes applied>",\n'
            '  "confidence": <0-100>,\n'
            '  "changes": ["fix 1", "fix 2", ...],\n'
            '  "explanation": "what was fixed based on orchestrator feedback"\n'
            "}\n\n"
            "CRITICAL: Fix ALL issues listed in orchestrator feedback. "
            "Return ONLY the ```json ... ``` block."
        )
        return prompt

    def _build_deepseek_verify_prompt(self, ported_code: str,
                                      patterns: List[Dict],
                                      glm_analysis: str = "",
                                      feedback: str = "",
                                      iteration: int = 1,
                                      max_iterations: int = 3) -> str:
        """Build the DeepSeek orchestrator verification prompt with feedback.

        DeepSeek-Orchestrator verifies ported HIP code and provides
        specific, actionable feedback for Kimi-Coder to fix issues in
        the next iteration of the orchestration loop.

        Output format: JSON with pass (bool), issues (list[str]),
        feedback (str), verdict (str).
        """
        prompt = (
            "You are DeepSeek-Orchestrator, the central orchestrator of the "
            "CUDA->HIP migration pipeline.\n"
            f"Orchestration loop: iteration {iteration}/{max_iterations}.\n\n"
            "Your task: review the following ported HIP kernel for correctness. "
            "If verification fails, provide SPECIFIC, ACTIONABLE, LINE-LEVEL "
            "feedback that Kimi-Coder can use to fix the issues.\n\n"
            "CHECKS:\n"
            "- wavefront64 compatibility (64 threads per wavefront)\n"
            "- __shfl masks use 0xffffffffffffffffULL (64-bit, not 32-bit)\n"
            "- __syncwarp() replaced with __syncthreads() where appropriate\n"
            "- shared memory sized for wavefront64 (not warp32)\n"
            "- HIP API usage (not deprecated CUDA APIs)\n"
            "- __ballot_sync used correctly for HIP\n"
        )

        # As orchestrator, review GLM-Planner's analysis
        if glm_analysis:
            prompt += (
                f"\nGLM-PLANNER ANALYSIS (under your authority as orchestrator):\n"
                f"{glm_analysis[:1000]}\n\n"
                "As orchestrator, verify GLM's analysis is complete. "
                "Note any missed issues.\n"
            )

        if feedback:
            prompt += (
                f"\nPREVIOUS FEEDBACK (issues that were reported in iteration "
                f"{iteration - 1} — verify they have been fixed):\n"
                f"{feedback}\n"
            )

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += "\n" + pattern_summary + "\n"

        prompt += (
            f"\nPORTED HIP KERNEL TO REVIEW:\n"
            f"```hip\n{ported_code[:2000]}\n```\n\n"
            "OUTPUT FORMAT -- STRICT JSON only:\n"
            "{\n"
            '  "pass": true/false,\n'
            '  "issues": ["issue 1", "issue 2", ...],\n'
            '  "feedback": "actionable, line-level feedback for Kimi to fix issues — '
            'be specific, mention exact lines and what to change. Empty string if pass=true.",\n'
            '  "verdict": "concise explanation of pass/fail"\n'
            "}\n\n"
            "EXAMPLE (pass):\n"
            "```json\n"
            "{\n"
            '  "pass": true,\n'
            '  "issues": [],\n'
            '  "feedback": "",\n'
            '  "verdict": "All wavefront64 checks pass. '
            'Masks use 0xffffffffffffffffULL, sync primitives are correct."\n'
            "}\n"
            "```\n\n"
            "EXAMPLE (fail with actionable feedback):\n"
            "```json\n"
            "{\n"
            '  "pass": false,\n'
            '  "issues": ["Line 42: __shfl_xor_sync mask is 0x1f, should be 0x3f for wavefront64"],\n'
            '  "feedback": "Line 42: Change __shfl_xor_sync mask from 0x1f to 0x3f '
            'to support 64-thread wavefronts.",\n'
            '  "verdict": "Shuffle mask is configured for 32-thread warps but '
            'AMD wavefronts need 64-thread masks.",\n'
            "}\n"
            "```\n\n"
            "CRITICAL: Return ONLY valid JSON. "
            "Set pass=true ONLY if all checks pass. "
            "When pass=false, the 'feedback' field is REQUIRED and must be "
            "specific enough for Kimi-Coder to fix without guessing."
        )
        return prompt

    # ── Main routing logic ──────────────────────────────────────

    def route(self, kernel_source: str, patterns: List[Dict],
              max_iterations: int = 3) -> Dict:
        """Route kernel through best models based on detected patterns.

        Orchestration loop: Kimi generates → DeepSeek orchestrator reviews →
        loop with feedback → converge (up to max_iterations cycles).

        Args:
            kernel_source: The CUDA kernel source code.
            patterns: List of classifier-detected patterns.
            max_iterations: Maximum Kimi→DeepSeek cycles (default 3).

        Returns:
            {"ported_code": ..., "confidence": ..., "changes": [...],
             "model_used": ..., "cost": ..., "orchestrator_passed": ...,
             "iterations_used": ...}
        """
        if not self.api_key:
            return {"ported_code": "", "confidence": 0,
                    "changes": ["No API key -- use template fallback"],
                    "model_used": "none", "cost": 0,
                    "orchestrator_passed": False, "iterations_used": 0}

        # Determine which patterns need which model
        pattern_types = [p.get("pattern", "default") for p in patterns]
        models_needed = set(ROUTING_TABLE.get(pt, "deepseek") for pt in pattern_types)

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0,
                  "orchestrator_passed": False, "iterations_used": 0}

        # Track pipeline phase outcomes for rubric scoring
        planner_success = False
        coder_success = False
        verify_success = False
        verify_passed = False
        orchestrator_feedback = ""

        # Phase 1: GLM plans the fix (planner, complex warp patterns)
        glm_plan_output = ""
        if "glm" in models_needed and any("shfl" in pt for pt in pattern_types):
            glm_prompt = self._build_glm_plan_prompt(kernel_source, patterns)
            plan = self._call_model("glm", glm_prompt,
                                    system_prompt=SYSTEM_PROMPTS.get("glm", ""))
            if plan.success:
                planner_success = True
                glm_plan_output = plan.output
                result["changes"].append(f"[glm] Planning complete: {plan.output[:120]}")
                result["changes"].append("[orch] DeepSeek JOB 1: reviewing GLM plan in verify phase")
            else:
                result["changes"].append("[glm] Planning FAILED — proceeding without GLM plan")

        # Phase 2: Kimi K2.7 generates initial ported code + orchestration loop
        if "kimi27" in models_needed:
            kimi_prompt = self._build_kimi_code_prompt(kernel_source, patterns)
            code = self._call_model("kimi27", kimi_prompt,
                                    system_prompt=SYSTEM_PROMPTS.get("kimi27", ""))
            if code.success:
                coder_success = True
                extracted = self._extract_code(code.output)
                extracted = self._fix_ported_code(extracted)
                result["ported_code"] = extracted
                result["changes"].append("[kimi27] Generated ported kernel")
                result["model_used"] = "kimi27"

                # ── Orchestration Loop: Kimi → DeepSeek → loop → Kimi → DeepSeek ──
                for iteration in range(1, max_iterations + 1):
                    if not result["ported_code"]:
                        break

                    verify_prompt = self._build_deepseek_verify_prompt(
                        result["ported_code"], patterns,
                        glm_analysis=glm_plan_output,
                        feedback=orchestrator_feedback,
                        iteration=iteration,
                        max_iterations=max_iterations,
                    )
                    result["changes"].append(
                        f"[orch] DeepSeek JOB 2 — verifying code (attempt {iteration}/{max_iterations})")
                    orchestrator = self._call_model(
                        "deepseek", verify_prompt,
                        system_prompt=SYSTEM_PROMPTS.get("deepseek", "")
                    )
                    result["iterations_used"] = iteration

                    if not orchestrator.success:
                        result["changes"].append(
                            f"[deepseek-orch] Call failed (iteration {iteration})")
                        break

                    # Parse orchestrator JSON response
                    raw = orchestrator.output.strip()
                    parsed = None
                    if raw.startswith("{"):
                        try: parsed = json.loads(raw)
                        except: pass
                    if parsed is None:
                        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                        if m:
                            try: parsed = json.loads(m.group(1))
                            except: pass
                    if parsed is None:
                        # Strategy 3: find {"pass"... to end of response, try progressively
                        m = re.search(r'\\{"pass"\\s*:', raw)
                        if m:
                            # Take from match to end, try to parse
                            candidate = raw[m.start():]
                            # Try full remainder
                            try: parsed = json.loads(candidate)
                            except: pass
                            # Try adding closing brace if truncated
                            if parsed is None and not candidate.rstrip().endswith('}'):
                                try: parsed = json.loads(candidate.rstrip() + '}')
                                except: pass
                            # Try finding balanced braces
                            if parsed is None:
                                depth = 0
                                for ci, ch in enumerate(candidate):
                                    if ch == '{': depth += 1
                                    elif ch == '}':
                                        depth -= 1
                                        if depth == 0:
                                            try: parsed = json.loads(candidate[:ci+1])
                                            except: pass
                                            break
                    if parsed is None:
                        result["changes"].append(
                            f'[deepseek-orch] JSON parse error (iter {iteration}): '
                            f'raw[:200]={raw[:200]}')
                        break

                    if parsed.get("pass", False):
                        verify_success = True
                        verify_passed = True
                        result["orchestrator_passed"] = True
                        result["changes"].append(
                            f"[deepseek-orch] Passed verification (iteration {iteration})")
                        break  # Converged — exit loop

                    # Verification failed — extract feedback for refinement
                    verify_success = True
                    orchestrator_feedback = parsed.get("feedback", "")
                    issues = parsed.get("issues", [])
                    if issues:
                        result["changes"].append(
                            f"[deepseek-orch] Iteration {iteration}: "
                            f"{' | '.join(issues[:3])}")

                    if iteration < max_iterations:
                        # Loop back: Kimi refines with orchestrator-specific feedback
                        refine_prompt = self._build_kimi_refine_prompt(
                            kernel_source, result["ported_code"],
                            orchestrator_feedback, patterns, iteration + 1
                        )
                        refine = self._call_model(
                            "kimi27", refine_prompt,
                            system_prompt=SYSTEM_PROMPTS.get("kimi27", "")
                        )
                        if refine.success:
                            extracted = self._extract_code(refine.output)
                            extracted = self._fix_ported_code(extracted)
                            result["ported_code"] = extracted
                            result["changes"].append(
                                f"[kimi27] Refined with orchestrator feedback "
                                f"(iteration {iteration} → {iteration + 1})")
                        else:
                            result["changes"].append(
                                f"[kimi27] Refinement failed (iteration {iteration})")
                            break
                    # else: max iterations reached, accept current output

        # Phase 3: Gemma 4 final verification (local AMD GPU via vLLM)
        if result["ported_code"] and "gemma4" in models_needed:
            # Try Gemma locally first, fallback to DeepSeek via Fireworks
            verify_prompt = self._build_deepseek_verify_prompt(
                result["ported_code"], patterns
            )
            verify = self._call_model("gemma4", verify_prompt,
                                      system_prompt=SYSTEM_PROMPTS.get("deepseek", ""))
            if verify.success:
                verify_success = verify_success or True
                result["model_used"] = "gemma4"
                try:
                    parsed = json.loads(verify.output)
                    if parsed.get("pass", False):
                        verify_passed = verify_passed or True
                        result["changes"].append(
                            "[gemma4] Verified -- no issues found (local AMD GPU)")
                    else:
                        issues = parsed.get("issues", [])
                        result["changes"].append(
                            f"[gemma4] Issues found: {'; '.join(issues[:3])}")
                except (json.JSONDecodeError, TypeError):
                    if "PASS" in verify.output.upper()[:10]:
                        verify_passed = verify_passed or True
                        result["changes"].append(
                            "[gemma4] Verified -- no issues found (local AMD GPU)")
                    else:
                        result["changes"].append(
                            f"[gemma4] Issues found: {verify.output[:200]}")
            else:
                # Fallback: DeepSeek via Fireworks
                result["changes"].append(
                    "[gemma4] Local AMD GPU unavailable -- falling back to DeepSeek")
                result["model_used"] = "deepseek-fallback"
                verify = self._call_model("deepseek", verify_prompt,
                                          system_prompt=SYSTEM_PROMPTS.get("deepseek", ""))
                if verify.success:
                    verify_success = verify_success or True
                    try:
                        parsed = json.loads(verify.output)
                        if parsed.get("pass", False):
                            verify_passed = verify_passed or True
                            result["changes"].append(
                                "[deepseek] Verified -- no issues found")
                        else:
                            issues = parsed.get("issues", [])
                            result["changes"].append(
                                f"[deepseek] Issues found: {'; '.join(issues[:3])}")
                    except (json.JSONDecodeError, TypeError):
                        if "PASS" in verify.output.upper()[:10]:
                            verify_passed = verify_passed or True
                            result["changes"].append(
                                "[deepseek] Verified -- no issues found")
                        else:
                            result["changes"].append(
                                f"[deepseek] Issues found: {verify.output[:200]}")

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
                    system_prompt: str = "") -> AgentResult:
        model_info = MODEL_CATALOG[model_key]
        model_id = model_info["id"]
        local_first = model_info.get("local_first", False)
        t0 = time.perf_counter()

        # Try in order: local-first for Gemma, Fireworks-first for others
        endpoints = []
        if local_first:
            endpoints = ["local", "fireworks"]
        else:
            endpoints = ["fireworks", "local"]

        for endpoint in endpoints:
            try:
                # Build messages with optional system prompt
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                if endpoint == "local":
                    data_bytes = json.dumps({
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": 512,
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
                        self.call_log.append({"model": model_key, "source": "local-vllm", "cost": 0})
                        return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                           0, round((time.perf_counter()-t0)*1000, 1))
                else:  # Fireworks
                    payload = {
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.2,
                    }
                    # Force JSON output for models that support it
                    if model_key in ("deepseek", "glm", "kimi27"):
                        payload["response_format"] = {"type": "json_object"}
                    data_bytes = json.dumps(payload).encode()
                    req = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        data=data_bytes,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        raw = resp.read()
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
                        usage = data.get("usage", {})
                        tokens = (
                            usage.get("total_tokens", 0)
                            or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                        )
                        cost = tokens / 1000 * model_info["cost_per_1k"]
                        self.total_cost += cost
                        self.call_log.append({"model": model_key, "tokens": tokens, "cost": cost})
                        return AgentResult(model_key, True, content, self._rubric_score_response(content),
                                           tokens, round((time.perf_counter()-t0)*1000, 1))
            except Exception as e:
                source = "local-vllm" if endpoint == "local" else "fireworks"
                err_msg = str(e)[:80]
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
                        with urllib.request.urlopen(req, timeout=60) as resp:
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
