"""
Model Router — Lightweight agent orchestration for CUDA→ROCm porting.

Architecture:
  Risk Classifier → Model Router → Kimi (planner) OR GLM (coder) OR Gemma (verifier)
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

MODEL_CATALOG = {
    "kimi": {
        "id": "accounts/fireworks/models/kimi-k2p6",  # ✅ VERIFIED WORKING
        "role": "planner",
        "strength": "complex kernel logic, multi-step reasoning",
        "cost_per_1k": 0.00095,
    },
    "glm": {
        "id": "accounts/fireworks/models/glm-5p2",  # ✅ VERIFIED WORKING
        "role": "coder",
        "strength": "accurate code generation, struct understanding",
        "cost_per_1k": 0.0014,
    },
    "gemma4": {
        "id": "accounts/fireworks/models/gemma-4-31b-it",
        "role": "verifier",
        "strength": "cheap, fast, good at spotting errors",
        "cost_per_1k": 0.0003,
        "access": "dedicated deployment only — use local vLLM on AMD GPU instead",
    },
}

# Pattern → best model routing table
ROUTING_TABLE = {
    "shfl_down_sync": "kimi",
    "shfl_xor_sync": "kimi",
    "syncwarp": "kimi",
    "warp_size_constant": "glm",
    "shared_mem_warp_count": "glm",
    "lane_id_mask": "glm",
    "tile_size_warp": "glm",
    # Generic patterns → cheapest model
    "__syncthreads": "gemma4",
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
      3. Kimi plans the fix structure (if complex)
      4. GLM generates the code
      5. Gemma verifies the output
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
        models_needed = set(ROUTING_TABLE.get(pt, "gemma4") for pt in pattern_types)

        result = {"ported_code": "", "confidence": 0,
                  "changes": [], "model_used": "", "cost": 0}

        # Phase 1: Kimi plans the fix (if complex patterns present)
        if "kimi" in models_needed and any("shfl" in pt for pt in pattern_types):
            plan = self._call_model("kimi",
                f"Analyze this CUDA kernel for warp(32)→wavefront(64) divergence. "
                f"Identify which lines need changes and why.\n\n```cuda\n{kernel_source[:2000]}\n```\n\n"
                f"Output format: JSON list of {{line, issue, fix}}")
            if plan.success:
                result["changes"].append(f"[kimi] {plan.output[:200]}")
                result["confidence"] += 0.3

        # Phase 2: GLM generates the code
        if "glm" in models_needed or True:  # always try glm for code gen
            code = self._call_model("glm",
                f"Port this CUDA kernel to HIP/ROCm. Fix warp(32)→wavefront(64) issues:\n"
                f"- __shfl_down_sync offset 16 works on both\n"
                f"- __shfl_xor_sync mask 0x1f → 0x3f\n"
                f"- warpSize 32 → make dynamic\n"
                f"- shared memory sized for warp 32 → annotate for wavefront64\n\n"
                f"```cuda\n{kernel_source[:2000]}\n```\n\n"
                f"Output ONLY the ported kernel code, no explanation.")
            if code.success:
                extracted = self._extract_code(code.output)
                result["ported_code"] = extracted
                result["changes"].append(f"[glm] Generated ported kernel")
                result["confidence"] += 0.4

        # Phase 3: Gemma verifies the output
        if result["ported_code"] and "gemma4" in models_needed:
            verify = self._call_model("gemma4",
                f"Review this HIP kernel for correctness. "
                f"Check: wavefront64 compatibility, correct __shfl usage, "
                f"shared memory sizing, and sync semantics.\n\n"
                f"```hip\n{result['ported_code'][:2000]}\n```\n\n"
                f"Output: 'PASS' or 'ISSUES: ...'")
            if verify.success:
                if "PASS" in verify.output.upper()[:10]:
                    result["confidence"] += 0.3
                    result["changes"].append(f"[gemma4] Verified — no issues found")
                else:
                    result["changes"].append(f"[gemma4] Issues found: {verify.output[:200]}")
                    result["confidence"] -= 0.2

        result["confidence"] = min(result["confidence"], 1.0) * 100
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
                return AgentResult(model_key, True, content, 0.7, tokens, round((time.perf_counter()-t0)*1000, 1))
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
                return AgentResult(model_key, True, content, 0.85, 0, round((time.perf_counter()-t0)*1000, 1))
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
