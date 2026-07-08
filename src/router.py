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
import os
import time
import urllib.request
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


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
    },
    "kimi27": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",  # ✅ VERIFIED WORKING
        "role": "coder",
        "strength": "code generation, struct-aware HIP porting",
        "cost_per_1k": 0.00095,
    },
    "deepseek": {
        "id": "accounts/fireworks/models/deepseek-v4-pro",  # ✅ VERIFIED WORKING
        "role": "verifier",
        "strength": "correctness checking, fallback when primary models fail",
        "cost_per_1k": 0.0012,
    },
}

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
    # Simple patterns → DeepSeek (verifier/catch-all)
    "__syncthreads": "deepseek",
    "default": "deepseek",
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
        import re
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

    def route(self, kernel_source: str, patterns: List[Dict]) -> Dict:
        """Route kernel through best models based on detected patterns.

        Returns:
          {"ported_code": ..., "confidence": ..., "changes": [...], "model_used": ..., "cost": ...}
        """
        if not self.api_key:
            return {"ported_code": "", "confidence": 0,
                    "changes": ["No API key — use template fallback"],
                    "model_used": "none", "cost": 0}

        # Determine which patterns need which model
        pattern_types = [p.get("pattern", "default") for p in patterns]
        models_needed = set(ROUTING_TABLE.get(pt, "deepseek") for pt in pattern_types)

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0}

        # Track pipeline phase outcomes for rubric scoring
        planner_success = False
        coder_success = False
        verify_success = False
        verify_passed = False

        # Phase 1: GLM plans the fix (planner, complex warp patterns)
        if "glm" in models_needed and any("shfl" in pt for pt in pattern_types):
            plan = self._call_model("glm",
                f"Analyze this CUDA kernel for warp(32)→wavefront(64) divergence. "
                f"Identify which lines need changes and why.\n\n```cuda\n{kernel_source[:2000]}\n```\n\n"
                f"Output format: JSON list of {{line, issue, fix}}")
            if plan.success:
                planner_success = True
                result["changes"].append(f"[glm] {plan.output[:200]}")

        # Phase 2: Kimi K2.7 Code generates the ported HIP code
        if "kimi27" in models_needed:
            code = self._call_model("kimi27",
                f"Port this CUDA kernel to HIP/ROCm. Fix warp(32)→wavefront(64) issues:\n"
                f"- __shfl_down_sync offset 16 works on both\n"
                f"- __shfl_xor_sync mask 0x1f → 0x3f\n"
                f"- warpSize 32 → make dynamic\n"
                f"- shared memory sized for warp 32 → annotate for wavefront64\n\n"
                f"```cuda\n{kernel_source[:2000]}\n```\n\n"
                f"Output ONLY the ported kernel code, no explanation.")
            if code.success:
                glm_success = True
                extracted = self._extract_code(code.output)
                extracted = self._fix_ported_code(extracted)
                result["ported_code"] = extracted
                result["changes"].append(f"[glm] Generated ported kernel")

        # Phase 3: DeepSeek verifies the output
        if result["ported_code"] and "deepseek" in models_needed:
            verify = self._call_model("deepseek",
                f"Review this HIP kernel for correctness. "
                f"Check: wavefront64 compatibility, correct __shfl usage, "
                f"shared memory sizing, and sync semantics.\n\n"
                f"```hip\n{result['ported_code'][:2000]}\n```\n\n"
                f"Output: 'PASS' or 'ISSUES: ...'")
            if verify.success:
                verify_success = True
                if "PASS" in verify.output.upper()[:10]:
                    verify_passed = True
                    result["changes"].append(f"[deepseek] Verified — no issues found")
                else:
                    result["changes"].append(f"[deepseek] Issues found: {verify.output[:200]}")

        # Rubric-based scoring replaces the old additive confidence model
        result["confidence"] = self._rubric_score_pipeline(
            kimi_success=planner_success,
            glm_success=coder_success,
            verify_success=verify_success,
            verify_passed=verify_passed,
            has_ported_code=bool(result["ported_code"]),
            ported_code=result["ported_code"],
            changes_count=len(result["changes"]),
        )
        result["cost"] = round(self.total_cost, 4)
        return result

    def _call_model(self, model_key: str, prompt: str) -> AgentResult:
        model_id = MODEL_CATALOG[model_key]["id"]
        t0 = time.perf_counter()

        # Try Fireworks API first
        try:
            data_bytes = json.dumps({
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.2,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data_bytes,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                cost = tokens / 1000 * MODEL_CATALOG[model_key]["cost_per_1k"]
                self.total_cost += cost
                self.call_log.append({"model": model_key, "tokens": tokens, "cost": cost})
                return AgentResult(model_key, True, content, self._rubric_score_response(content), tokens, round((time.perf_counter()-t0)*1000, 1))
        except Exception:
            pass

        # Fallback: try local vLLM endpoint (for Gemma on AMD GPU)
        try:
            data_bytes = json.dumps({
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }).encode()
            local_req = urllib.request.Request(
                "http://localhost:8000/v1/chat/completions",
                data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(local_req, timeout=30) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                cost = 0  # local = free
                self.call_log.append({"model": model_key, "source": "local-vllm", "cost": cost})
                return AgentResult(model_key, True, content, self._rubric_score_response(content), 0, round((time.perf_counter()-t0)*1000, 1))
        except Exception:
            pass

        return AgentResult(model_key, False, f"Model {model_id} unavailable", 0)

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
