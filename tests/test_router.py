"""I1: Tests for the actual porting loop — mocked convergence, error normalization,
cycle detection, and rubric scoring. These test ModelRouter directly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch, MagicMock
import pytest
from router import ModelRouter, AgentResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    """Create a ModelRouter with a dummy API key."""
    return ModelRouter(api_key="test_key")


CUDA_KERNEL_EXAMPLE = """
#include <cuda_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    float val = input[tid];
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __shfl_up_sync(0xffffffff, val, stride);
        input[tid] = val;
    }
}
"""

HIP_CODE_OK = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    float val = input[tid];
    for (int stride = 1; stride < 64; stride *= 2) {
        __shfl_up_sync(0xffffffffffffffffULL, val, stride);
        input[tid] = val;
    }
}
"""

HIP_CODE_WITH_CUDA = """
#include <cuda_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    float* d_input;
    cudaMalloc(&d_input, n * sizeof(float));
}
"""


# ── I1: Error normalization tests ─────────────────────────────────────────────

class TestNormalizeError:
    """TRIZ #3/#22: Error normalization strips volatile parts before diffing."""

    def test_strips_line_numbers(self, router):
        """Same error at different line numbers normalizes equal."""
        e1 = "/tmp/build/file.cpp:67:5: error: use of undeclared identifier 'hipMalloc'"
        e2 = "/tmp/build/file.cpp:68:3: error: use of undeclared identifier 'hipMalloc'"
        assert router._normalize_error(e1) == router._normalize_error(e2)

    def test_different_messages_stay_distinct(self, router):
        """Different error messages stay distinct after normalization."""
        e1 = "file.cpp:67: error: undeclared 'hipMalloc'"
        e2 = "file.cpp:67: error: undeclared 'hipFree'"
        assert router._normalize_error(e1) != router._normalize_error(e2)

    def test_strips_temp_paths(self, router):
        """Temp build paths are stripped from normalized errors."""
        e1 = "/tmp/verifier_build_abc123/file.cpp:42: error: foo"
        e2 = "/tmp/verifier_build_xyz789/file.cpp:42: error: foo"
        assert router._normalize_error(e1) == router._normalize_error(e2)

    def test_empty_error(self, router):
        """Empty or whitespace-only errors normalize to empty string."""
        assert router._normalize_error("") == ""
        assert router._normalize_error("   ") == ""


# ── I1: Rubric scoring tests (A9) ──────────────────────────────────────────────

class TestRubricScoring:
    """A9: Rubric should reward HIP APIs, not CUDA keywords."""

    def test_hip_code_scores_higher_than_cuda_code(self, router):
        """HIP code with no CUDA remnants should score higher than code with CUDA."""
        hip_score = router._rubric_score_response(HIP_CODE_OK)
        cuda_score = router._rubric_score_response(HIP_CODE_WITH_CUDA)
        assert hip_score > cuda_score

    def test_cuda_remnants_penalized(self, router):
        """Code with CUDA remnants should not get the no-remnant bonus."""
        score_with_cuda = router._rubric_score_response(HIP_CODE_WITH_CUDA)
        score_clean_hip = router._rubric_score_response(HIP_CODE_OK)
        # Clean HIP code should get the full bonus
        assert score_clean_hip > score_with_cuda

    def test_empty_output_scores_zero(self, router):
        """Empty output should score 0.0."""
        assert router._rubric_score_response("") == 0.0
        assert router._rubric_score_response("   ") == 0.0

    def test_pipeline_score_rewards_hip_apis(self, router):
        """Pipeline rubric should give higher score to code with HIP APIs."""
        hip_score = ModelRouter._rubric_score_pipeline(
            True, True, True, True, True, HIP_CODE_OK, 5)
        cuda_score = ModelRouter._rubric_score_pipeline(
            True, True, True, True, True, HIP_CODE_WITH_CUDA, 5)
        assert hip_score > cuda_score


# ── I1: A2A Message protocol tests ─────────────────────────────────────────────

class TestA2AMessage:
    """A2A structured message protocol — replaces blob truncation."""

    def test_build_plan_message_extracts_mappings(self, router):
        """_build_deepseek_plan_message should extract API mappings from plan text."""
        plan = "Replace cudaMalloc with hipMalloc. Change cuda_runtime.h to hip/hip_runtime.h."
        msg = router._build_deepseek_plan_message(plan, CUDA_KERNEL_EXAMPLE)
        assert "hipMalloc" in msg.summary or "hip" in msg.summary.lower()
        assert len(msg.priority_details) > 0

    def test_build_error_feedback_message_includes_all_errors(self, router):
        """_build_error_feedback_message should include ALL errors, not just first 3."""
        errs = [f"error {i}: undeclared identifier 'foo_{i}'" for i in range(10)]
        msg = router._build_error_feedback_message(errs, iteration=1)
        # Summary should mention 10 errors
        assert "10" in msg.summary
        # Should have 10 priority details (ALL errors)
        assert len(msg.priority_details) == 10

    def test_a2a_to_prompt_within_budget(self, router):
        """to_prompt should stay within character budget."""
        errs = [f"error {i}: very long error message about undeclared identifier 'foo_{i}'" for i in range(50)]
        msg = router._build_error_feedback_message(errs, iteration=1)
        rendered = msg.to_prompt(max_chars=500)
        assert len(rendered) <= 600  # small buffer for formatting

    def test_a2a_to_prompt_summary_always_present(self, router):
        """Summary should always be in the rendered prompt, even if details truncated."""
        msg = router._build_error_feedback_message(
            ["error: undeclared 'hipMalloc'"], iteration=1)
        rendered = msg.to_prompt(max_chars=100)
        assert "1 compile error" in rendered or "undeclared" in rendered


# ── I1: Convergence loop tests (mocked) ────────────────────────────────────────

class TestConvergenceLoop:
    """Test the actual porting loop with mocked LLM calls."""

    @patch.object(ModelRouter, '_call_model')
    def test_successful_port_first_try(self, mock_call, router):
        """DeepSeek plans, Kimi codes, hipcc compiles, GLM passes — 1 iteration."""
        # Call sequence: deepseek(plan) → kimi27(code) → glm(eval) → gemma4(verify)
        mock_call.side_effect = [
            AgentResult("deepseek", True, "Plan: replace cudaMalloc with hipMalloc", 0.1),
            AgentResult("kimi27", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1),
            AgentResult("glm", True, '{"pass": true, "feedback": "looks good"}', 0.1),
            AgentResult("gemma4", True, '{"pass": true}', 0.1),
        ]
        mock_verifier = MagicMock()
        mock_verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""
        }
        mock_verifier.verify.return_value = {
            "compile_success": True, "passed": True,
            "compile_output": "", "output": ""
        }

        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE,
            patterns=[],
            max_iterations=1,
            verifier=mock_verifier,
            kernel_name="test_kernel"
        )
        # Should succeed with 0 compile errors (hipcc passes first try)
        assert result["ported_code"] is not None

    @patch.object(ModelRouter, '_call_model')
    def test_compile_failure_triggers_refinement(self, mock_call, router):
        """hipcc fails → GLM analyzes → Kimi refines — verify Kimi called at least twice."""
        glm_err = AgentResult("glm", True, '{"fixes": [{"action": "Replace cudaMalloc with hipMalloc", "priority": 1}], "missing_includes": ["hip/hip_runtime.h"]}', 0.1)
        glm_eval = AgentResult("glm", True, '{"pass": true}', 0.1)
        kimi_bad = AgentResult("kimi27", True, f"```cpp\n{HIP_CODE_WITH_CUDA}\n```", 0.1)
        kimi_good = AgentResult("kimi27", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1)
        gemma = AgentResult("gemma4", True, '{"pass": true}', 0.1)

        kimi_call_count = [0]
        def call_side_effect(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return AgentResult("deepseek", True, "Plan: replace cudaMalloc with hipMalloc", 0.1)
            if model_key == "kimi27":
                kimi_call_count[0] += 1
                return kimi_bad if kimi_call_count[0] == 1 else kimi_good
            if model_key == "glm":
                sysp = kwargs.get("system_prompt", "")
                if "error analyst" in sysp:
                    return glm_err
                return glm_eval
            if model_key == "gemma4":
                return gemma
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = call_side_effect

        mock_verifier = MagicMock()
        # [pre-loop] compile fail → [loop iter1] compile fail → [loop iter1 refine] pass → enough passes
        mock_verifier.quick_compile_check.side_effect = [
            {"compile_success": False, "errors": ["error: use of undeclared identifier 'cudaMalloc'"], "output": ""},
            {"compile_success": False, "errors": ["error: use of undeclared identifier 'cudaMalloc'"], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
        ]
        mock_verifier.verify.return_value = {
            "compile_success": True, "passed": True,
            "compile_output": "", "output": ""
        }

        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE,
            patterns=[],
            max_iterations=3,
            verifier=mock_verifier,
            kernel_name="test_refine"
        )
        # Kimi should have been called at least twice (initial + refine)
        assert kimi_call_count[0] >= 2, f"Kimi called {kimi_call_count[0]} times, expected >=2"

    @patch.object(ModelRouter, '_call_model')
    def test_stagnation_triggers_replan(self, mock_call, router):
        """Multiple stagnant iterations should trigger DeepSeek re-planning."""
        glm_err = AgentResult("glm", True, '{"fixes": [{"action": "Replace cudaMalloc with hipMalloc", "priority": 1}]}', 0.1)
        glm_eval = AgentResult("glm", True, '{"pass": true}', 0.1)
        kimi_bad = AgentResult("kimi27", True, f"```cpp\n{HIP_CODE_WITH_CUDA}\n```", 0.1)
        kimi_good = AgentResult("kimi27", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1)
        gemma = AgentResult("gemma4", True, '{"pass": true}', 0.1)

        ds_call_count = [0]
        def call_side_effect(model_key, *args, **kwargs):
            if model_key == "deepseek":
                ds_call_count[0] += 1
                if ds_call_count[0] == 1:
                    return AgentResult("deepseek", True, "Plan v1: replace cudaMalloc", 0.1)
                return AgentResult("deepseek", True, "Plan v2: use hipMalloc completely", 0.1)
            if model_key == "kimi27":
                # Bad code until DeepSeek re-plans, then good
                if ds_call_count[0] <= 1:
                    return kimi_bad
                return kimi_good
            if model_key == "glm":
                sysp = kwargs.get("system_prompt", "")
                if "error analyst" in sysp:
                    return glm_err
                return glm_eval
            if model_key == "gemma4":
                return gemma
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = call_side_effect

        mock_verifier = MagicMock()
        # [pre-loop] fail + [loop iter1] fail + [loop iter2] fail + [loop iter3] fail + pass for after re-plan
        mock_verifier.quick_compile_check.side_effect = [
            {"compile_success": False, "errors": ["error: cudaMalloc undeclared"], "output": ""},
            {"compile_success": False, "errors": ["error: cudaMalloc undeclared"], "output": ""},
            {"compile_success": False, "errors": ["error: cudaMalloc undeclared"], "output": ""},
            {"compile_success": False, "errors": ["error: cudaMalloc undeclared"], "output": ""},
            {"compile_success": False, "errors": ["error: cudaMalloc undeclared"], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
            {"compile_success": True, "errors": [], "output": ""},
        ]
        mock_verifier.verify.return_value = {
            "compile_success": True, "passed": True,
            "compile_output": "", "output": ""
        }

        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE,
            patterns=[],
            max_iterations=8,
            verifier=mock_verifier,
            kernel_name="test_stagnation"
        )
        # DeepSeek should have been called at least twice (initial + re-plan)
        assert ds_call_count[0] >= 2, f"DeepSeek called {ds_call_count[0]} times, expected >=2"
