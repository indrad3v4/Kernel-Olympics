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
    "deepseek": {
        "id": "accounts/fireworks/models/deepseek-v4-pro",  # ✅ VERIFIED WORKING
        "role": "planner",          # CHANGED: reasoning model → planner (prose OK)
        "strength": "deep reasoning, chain-of-thought planning, pattern analysis",
        "cost_per_1k": 0.0012,
        "local_first": False,
        "max_tokens": 2048,      # planner needs room for reasoning
        "temperature": 0.3,      # creativity for diverse plans
    },
    "kimi27": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",  # ✅ VERIFIED WORKING
        "role": "coder",            # UNCHANGED
        "strength": "code generation, struct-aware HIP porting",
        "cost_per_1k": 0.00095,
        "local_first": False,
        "max_tokens": 4096,      # coder needs room for full kernel output
        "temperature": 0.1,      # code generation needs precision
    },
    "glm": {
        "id": "accounts/fireworks/models/glm-5p2",  # ✅ VERIFIED WORKING
        "role": "evaluator",         # CHANGED: was planner → now evaluator (strict JSON)
        "strength": "structured JSON output, correctness checking, wavefront64 validation",
        "cost_per_1k": 0.0014,
        "local_first": False,
        "max_tokens": 1024,      # evaluator output is compact (pass/fail + issues)
        "temperature": 0.0,      # deterministic evaluation
    },
    "gemma4": {
        "id": "accounts/fireworks/models/gemma-4-31b-it",  # Real Gemma on AMD GPU via vLLM
        "role": "verifier",         # local AMD GPU verification (Gemma Prize)
        "strength": "AMD-native verification via local vLLM on MI300X",
        "cost_per_1k": 0.0,
        "local_first": True,     # Try localhost:8000 first, then Fireworks
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

# ── JSON schemas for response_format (enforced by Fireworks API) ──
# Using json_object (not json_schema) for compatibility — the system prompt
# already defines the exact shape. json_schema is stricter but not all models support it.

JSON_SCHEMAS = {
    "glm": {  # GLM is now evaluator — strict schema for pass/fail JSON
        "type": "json_schema",
        "json_schema": {
            "name": "EvaluationResult",
            "schema": {
                "type": "object",
                "properties": {
                    "pass": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "feedback": {"type": "string"},
                    "verdict": {"type": "string"},
                },
                "required": ["pass", "issues", "feedback", "verdict"],
            },
        },
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
        "DO NOT include reasoning, explanations, or chain-of-thought. "
        "Output ONLY the JSON object, nothing before or after."
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
        """Fix AMD-specific issues in ported code.

        Post-processing safety net applied after every Kimi code generation
        and refinement pass. Catches common issues the LLM may miss.
        """
        # Replace CUDA headers with HIP equivalents
        code = re.sub(r'#include\s*<cuda_runtime\.h>', '#include <hip/hip_runtime.h>', code)
        code = re.sub(r'#include\s*<cuda_runtime_api\.h>', '#include <hip/hip_runtime.h>', code)
        code = re.sub(r'#include\s*"cuda_runtime\.h"', '#include <hip/hip_runtime.h>', code)
        # Remove NVIDIA helper headers — not available in ROCm
        code = re.sub(r'#include\s*<helper_cuda\.h>\n?', '', code)
        code = re.sub(r'#include\s*<helper_functions\.h>\n?', '', code)
        code = re.sub(r'#include\s*"helper_cuda\.h"\n?', '', code)
        code = re.sub(r'#include\s*"helper_functions\.h"\n?', '', code)
        # Remove project-specific .cuh headers — not available in HIP port
        code = re.sub(r'#include\s*"[^"]*\.cuh"\n?', '', code)
        code = re.sub(r'#include\s*<[^>]*\.cuh>\n?', '', code)
        # Remove device_launch_parameters.h — not needed in HIP
        code = re.sub(r'#include\s*<device_launch_parameters\.h>\n?', '', code)

        # Inject #define WAVEFRONT_SIZE 64 if not already defined
        # ROCm wavefront is 64 on gfx9 (MI300/MI250). CUDA warp is 32.
        if not re.search(r'#define\s+WAVEFRONT_SIZE', code):
            # Insert after the last #include line (or at top if no includes)
            include_lines = list(re.finditer(r'#include\s+[<"].*[>"]\n', code))
            if include_lines:
                last_include = include_lines[-1]
                insert_pos = last_include.end()
                code = code[:insert_pos] + '#define WAVEFRONT_SIZE 64\n' + code[insert_pos:]
            else:
                code = '#define WAVEFRONT_SIZE 64\n' + code

        # Fix __shfl_xor_sync mask: 0x1f (5-bit, warp32) → 0x3f (6-bit, wavefront64)
        code = re.sub(
            r'(__shfl_xor_sync\s*\()0x1f(,)',
            r'\g<1>0x3f\g<2>',
            code
        )
        # Fix __shfl_down_sync mask: 32-bit 0xffffffff → 64-bit for wavefront64
        code = re.sub(
            r'(__shfl_\w+_sync\()0x[fF]{8}(,)',
            r'\g<1>0xffffffffffffffffULL\g<2>',
            code
        )
        # Replace __syncwarp() with __syncthreads() for wavefront64 safety
        code = re.sub(r'\b__syncwarp\s*\(\s*\)', '__syncthreads()', code)

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
            prompt += f"DeepSeek Planner's plan (follow this):\n{deepseek_plan[:2000]}\n\n"

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"```cuda\n{kernel_source[:6000]}\n```\n\n"
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}."
        )
        return prompt

    def _build_kimi_refine_prompt(self, kernel_source: str,
                                  previous_code: str,
                                  feedback: str,
                                  patterns: List[Dict],
                                  deepseek_plan: str = "",
                                  iteration: int = 1) -> str:
        """Build the Kimi refinement prompt for orchestration loop iterations.

        Kimi receives the original kernel, its previous output, and
        GLM evaluator's specific feedback to fix issues.
        """
        prompt = (
            f"Fix your ported HIP kernel based on evaluator feedback (iteration {iteration}).\n\n"
            "CHECKLIST:\n"
            "- __shfl_xor_sync mask 0x1f → 0x3f for wavefront64\n"
            "- __shfl_down_sync masks → 0xffffffffffffffffULL (64-bit)\n"
            "- warpSize 32 → WAVEFRONT_SIZE 64 or dynamic\n"
            "- shared memory sized for warp 32 → WAVEFRONT_SIZE (64)\n"
            "- __syncwarp() → __syncthreads()\n"
            "- #define WAVEFRONT_SIZE 64 at top\n"
            "- Replace #include <cuda_runtime.h> → #include <hip/hip_runtime.h>\n"
            "- Remove #include <helper_cuda.h>, <helper_functions.h>, <device_launch_parameters.h>\n"
            "- Remove ALL #include \"*.cuh\" local headers\n\n"
        )

        if deepseek_plan:
            prompt += f"DeepSeek Planner's plan (reference):\n{deepseek_plan[:1500]}\n\n"

        pattern_summary = _format_patterns_summary(patterns)
        if pattern_summary:
            prompt += pattern_summary + "\n"

        prompt += (
            f"Original CUDA:\n```cuda\n{kernel_source[:4000]}\n```\n\n"
            f"Your previous output:\n```hip\n{previous_code[:4000]}\n```\n\n"
            f"Evaluator feedback (fix ALL):\n{feedback}\n\n"
            "Respond with JSON: {\"ported_code\": str, \"confidence\": 0-100, "
            "\"changes\": [str], \"explanation\": str}."
        )
        return prompt

    def _build_glm_evaluate_prompt(self, ported_code: str,
                                   patterns: List[Dict],
                                   deepseek_plan: str = "",
                                   feedback: str = "",
                                   iteration: int = 1,
                                   max_iterations: int = 3) -> str:
        """Build the GLM evaluator prompt.

        Role: GLM-Evaluator — strict JSON output. Checks ported code for
        wavefront64 correctness, CUDA remnants, and compilation safety.
        System prompt already defines role + JSON contract.
        """
        prompt = f"Evaluate this ported HIP kernel (iteration {iteration}/{max_iterations}).\n\n"

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
            prompt += f"Planner's plan (reference):\n{deepseek_plan[:800]}\n\n"

        if feedback:
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

    # ── Main routing logic ──────────────────────────────────────

    def route(self, kernel_source: str, patterns: List[Dict],
              max_iterations: int = 3) -> Dict:
        """Route kernel through the loop engineering pipeline.

        Loop: DeepSeek (plan) → Kimi (code) → GLM (evaluate) → feedback → DeepSeek (re-plan)

        Args:
            kernel_source: The CUDA kernel source code.
            patterns: List of classifier-detected patterns.
            max_iterations: Maximum Kimi→GLM cycles (default 3).

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

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0,
                  "orchestrator_passed": False, "iterations_used": 0}

        # Track pipeline phase outcomes for rubric scoring
        planner_success = False
        coder_success = False
        verify_success = False
        verify_passed = False
        evaluator_feedback = ""
        deepseek_plan_output = ""

        # ── Phase 1: DeepSeek PLANS the port (reasoning model — prose OK) ──
        ds_prompt = self._build_deepseek_plan_prompt(kernel_source, patterns)
        plan = self._call_model("deepseek", ds_prompt,
                                system_prompt=SYSTEM_PROMPTS.get("deepseek", ""))
        if plan.success:
            planner_success = True
            deepseek_plan_output = plan.output
            result["changes"].append(f"[deepseek] Plan generated ({len(plan.output)} chars)")
        else:
            result["changes"].append("[deepseek] Planning FAILED — proceeding without plan")

        # ── Phase 2: Kimi CODES the initial port ──
        kimi_prompt = self._build_kimi_code_prompt(kernel_source, patterns,
                                                   deepseek_plan=deepseek_plan_output)
        code = self._call_model("kimi27", kimi_prompt,
                                system_prompt=SYSTEM_PROMPTS.get("kimi27", ""))
        if code.success:
            coder_success = True
            extracted = self._extract_code(code.output)
            extracted = self._fix_ported_code(extracted)
            result["ported_code"] = extracted
            result["changes"].append("[kimi27] Generated ported kernel")
            result["model_used"] = "kimi27"
        else:
            result["changes"].append("[kimi27] Code generation FAILED")
            # Can't proceed without initial code
            result["cost"] = round(self.total_cost, 4)
            return result

        # ── Phase 3: GLM EVALUATES → loop → Kimi refines → GLM re-evaluates ──
        for iteration in range(1, max_iterations + 1):
            if not result["ported_code"]:
                break

            eval_prompt = self._build_glm_evaluate_prompt(
                result["ported_code"], patterns,
                deepseek_plan=deepseek_plan_output,
                feedback=evaluator_feedback,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            result["changes"].append(
                f"[glm] Evaluating code (attempt {iteration}/{max_iterations})")
            evaluator = self._call_model(
                "glm", eval_prompt,
                system_prompt=SYSTEM_PROMPTS.get("glm", "")
            )
            result["iterations_used"] = iteration

            if not evaluator.success:
                result["changes"].append(
                    f"[glm] Call failed (iteration {iteration})")
                break

            # Parse GLM evaluator JSON response
            # GLM follows json_schema — should be clean JSON, but keep fallbacks
            raw = evaluator.output.strip()
            parsed = None

            # Strategy 1: pure JSON
            if raw.startswith("{"):
                try: parsed = json.loads(raw)
                except: pass

            # Strategy 2: JSON inside ```json ... ``` markdown
            if parsed is None:
                m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                if m:
                    try: parsed = json.loads(m.group(1))
                    except: pass

            # Strategy 3: find {"pass" ... anywhere (safety net)
            if parsed is None:
                m = re.search(r'\{"pass"\s*:', raw)
                if m:
                    candidate = raw[m.start():]
                    try: parsed = json.loads(candidate)
                    except: pass
                    if parsed is None:
                        # balanced braces
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
                                        except: pass
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
                result["changes"].append(
                    f'[glm] JSON parse error (iter {iteration}): '
                    f'raw[:200]={raw[:200]}')
                break

            if parsed.get("pass", False):
                verify_success = True
                verify_passed = True
                result["orchestrator_passed"] = True
                result["changes"].append(
                    f"[glm] Passed evaluation (iteration {iteration})")
                break  # Converged — exit loop

            # Evaluation failed — extract feedback for refinement
            verify_success = True
            evaluator_feedback = parsed.get("feedback", "")
            issues = parsed.get("issues", [])
            if issues:
                result["changes"].append(
                    f"[glm] Iteration {iteration}: "
                    f"{' | '.join(issues[:3])}")

            if iteration < max_iterations:
                # Loop back: Kimi refines with GLM evaluator feedback
                refine_prompt = self._build_kimi_refine_prompt(
                    kernel_source, result["ported_code"],
                    evaluator_feedback, patterns,
                    deepseek_plan=deepseek_plan_output,
                    iteration=iteration + 1,
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
                        f"[kimi27] Refined with evaluator feedback "
                        f"(iteration {iteration} → {iteration + 1})")
                else:
                    result["changes"].append(
                        f"[kimi27] Refinement failed (iteration {iteration})")
                    break
            # else: max iterations reached, accept current output

        # ── Phase 4: Gemma 4 final verification (local AMD GPU via vLLM) ──
        if result["ported_code"]:
            gemma_prompt = self._build_glm_evaluate_prompt(
                result["ported_code"], patterns
            )
            verify = self._call_model("gemma4", gemma_prompt,
                                      system_prompt=SYSTEM_PROMPTS.get("glm", ""))
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
                result["changes"].append(
                    "[gemma4] Local AMD GPU unavailable")

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
                        "max_tokens": model_info.get("max_tokens", 1024),
                        "temperature": model_info.get("temperature", 0.2),
                    }
                    # Use json_schema for DeepSeek (strict), json_object for others
                    schema = JSON_SCHEMAS.get(model_key)
                    if schema:
                        payload["response_format"] = schema
                    data_bytes = json.dumps(payload).encode()
                    req = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        data=data_bytes,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=90) as resp:
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
                        with urllib.request.urlopen(req, timeout=90) as resp:
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
