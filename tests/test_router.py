"""I1: Tests for the actual porting loop — mocked convergence, error normalization,
cycle detection, and rubric scoring. These test ModelRouter directly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
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
        """DeepSeek plans, GLM codes, hipcc compiles, Kimi passes — 1 iteration."""
        # Call sequence: deepseek(plan) → glm(code) → kimi27(eval) → gemma4(verify)
        mock_call.side_effect = [
            AgentResult("deepseek", True, "Plan: replace cudaMalloc with hipMalloc", 0.1),
            AgentResult("glm", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1),
            AgentResult("kimi27", True, '{"pass": true, "feedback": "looks good"}', 0.1),
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
        """hipcc fails → Kimi analyzes → GLM refines — verify GLM called at least twice."""
        eval_err = AgentResult("kimi27", True, '{"fixes": [{"action": "Replace cudaMalloc with hipMalloc", "priority": 1}], "missing_includes": ["hip/hip_runtime.h"]}', 0.1)
        eval_pass = AgentResult("kimi27", True, '{"pass": true}', 0.1)
        coder_bad = AgentResult("glm", True, f"```cpp\n{HIP_CODE_WITH_CUDA}\n```", 0.1)
        coder_good = AgentResult("glm", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1)
        gemma = AgentResult("gemma4", True, '{"pass": true}', 0.1)

        glm_call_count = [0]
        def call_side_effect(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return AgentResult("deepseek", True, "Plan: replace cudaMalloc with hipMalloc", 0.1)
            if model_key == "glm":
                glm_call_count[0] += 1
                return coder_bad if glm_call_count[0] == 1 else coder_good
            if model_key == "kimi27":
                sysp = kwargs.get("system_prompt", "")
                if "error analyst" in sysp:
                    return eval_err
                return eval_pass
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
        # GLM should have been called at least twice (initial + refine)
        assert glm_call_count[0] >= 2, f"GLM coder called {glm_call_count[0]} times, expected >=2"

    @patch.object(ModelRouter, '_call_model')
    def test_stagnation_triggers_replan(self, mock_call, router):
        """Multiple stagnant iterations should trigger DeepSeek re-planning."""
        eval_err = AgentResult("kimi27", True, '{"fixes": [{"action": "Replace cudaMalloc with hipMalloc", "priority": 1}]}', 0.1)
        eval_pass = AgentResult("kimi27", True, '{"pass": true}', 0.1)
        coder_bad = AgentResult("glm", True, f"```cpp\n{HIP_CODE_WITH_CUDA}\n```", 0.1)
        coder_good = AgentResult("glm", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1)
        gemma = AgentResult("gemma4", True, '{"pass": true}', 0.1)

        ds_call_count = [0]
        def call_side_effect(model_key, *args, **kwargs):
            if model_key == "deepseek":
                ds_call_count[0] += 1
                if ds_call_count[0] == 1:
                    return AgentResult("deepseek", True, "Plan v1: replace cudaMalloc", 0.1)
                return AgentResult("deepseek", True, "Plan v2: use hipMalloc completely", 0.1)
            if model_key == "glm":
                # Bad code until DeepSeek re-plans, then good
                if ds_call_count[0] <= 1:
                    return coder_bad
                return coder_good
            if model_key == "kimi27":
                sysp = kwargs.get("system_prompt", "")
                if "error analyst" in sysp:
                    return eval_err
                return eval_pass
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


# ── RUN-FIRST: compile-pass is not convergence — the binary must also run ─────

class TestRunFirstLoop:
    """The 2026-07-09 run compiled on iteration 1, GLM flagged the likely
    crash cause, the loop discarded the finding and declared victory — then
    the binary SIGSEGVed in verify() with no feedback path back. The loop
    now runs the binary in-loop after every passing compile."""

    @staticmethod
    def _call_side_effect(model_key, *args, **kwargs):
        if model_key == "deepseek":
            return AgentResult("deepseek", True, "Plan: port shfl", 0.1)
        if model_key == "glm":
            return AgentResult("glm", True, f"```cpp\n{HIP_CODE_OK}\n```", 0.1)
        if model_key == "kimi27":
            return AgentResult("kimi27", True,
                '{"pass": true, "issues": ["__shfl_up_sync uses width as width param"], "feedback": "check shfl width"}', 0.1)
        return AgentResult(model_key, True, '{"pass": true}', 0.1)

    @patch.object(ModelRouter, '_call_model')
    def test_runtime_crash_triggers_refine_not_convergence(self, mock_call, router):
        """Compile passes but the binary SIGSEGVs → the loop must refine,
        not break — then converge once the binary runs clean."""
        mock_call.side_effect = self._call_side_effect
        mock_verifier = MagicMock()
        mock_verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""
        }
        mock_verifier.quick_run_check.side_effect = [
            {"run_success": False, "run_exit_code": -11, "signal": "SIGSEGV", "run_output": ""},
            {"run_success": True, "run_exit_code": 0, "signal": "", "run_output": "TEST PASSED"},
        ]
        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE, patterns=[],
            max_iterations=3, verifier=mock_verifier, kernel_name="test_runfirst",
        )
        assert result["orchestrator_passed"] is True
        assert result["iterations_used"] == 2, (
            "loop must NOT declare convergence on the crashing iteration")
        crash_entries = [c for c in result["changes"] if "CRASHED at runtime" in c]
        assert crash_entries, "the crash must be recorded in the changes log"
        assert "SIGSEGV" in crash_entries[0]

    @patch.object(ModelRouter, '_call_model')
    def test_three_runtime_crashes_abort(self, mock_call, router):
        """Persistent crashes must abort with runtime_stagnation, not burn
        the full iteration budget."""
        mock_call.side_effect = self._call_side_effect
        mock_verifier = MagicMock()
        mock_verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""
        }
        mock_verifier.quick_run_check.return_value = {
            "run_success": False, "run_exit_code": -11, "signal": "SIGSEGV", "run_output": ""
        }
        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE, patterns=[],
            max_iterations=10, verifier=mock_verifier, kernel_name="test_runcrash",
        )
        assert result["abort_reason"] == "runtime_stagnation"
        assert result["orchestrator_passed"] is False
        assert result["iterations_used"] <= 3

    @patch.object(ModelRouter, '_call_model')
    def test_mock_verifier_without_run_info_still_converges(self, mock_call, router):
        """A verifier whose quick_run_check returns a non-dict (MagicMock in
        every pre-existing test) must read as 'no run info', not a crash —
        preserving the old convergence behavior."""
        mock_call.side_effect = self._call_side_effect
        mock_verifier = MagicMock()
        mock_verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""
        }
        # quick_run_check left as bare MagicMock → returns MagicMock, not dict
        result = router.route(
            kernel_source=CUDA_KERNEL_EXAMPLE, patterns=[],
            max_iterations=2, verifier=mock_verifier, kernel_name="test_norun",
        )
        assert result["orchestrator_passed"] is True
        assert result["iterations_used"] == 1


# ── Bug 1: self-contained programs must not be truncated ──────────────────────

SELF_CONTAINED_SOURCE = """
#include <cuda_runtime.h>
__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL) {
    int tid = threadIdx.x;
}

""" + ("// padding line to push the real entry point past a 6000-character window\n" * 250) + """
int main(int argc, char *argv[]) {
    shfl_scan_test<<<4, 64>>>(nullptr, 0, nullptr);
    return 0;
}
"""


class TestSelfContainedPromptTruncation:
    """Bug 1: GLM must see a self-contained program's own main(), even when
    the source is long enough that the old fixed character slices would have
    cut it off (see docs/fix-plan-self-contained-programs.md)."""

    def test_source_is_long_enough_to_have_broken_the_old_truncation(self):
        # Sanity check on the fixture itself — if this fails, the fixture no
        # longer exercises the bug and the tests below would pass vacuously.
        assert len(SELF_CONTAINED_SOURCE) > 6000
        assert SELF_CONTAINED_SOURCE[:6000].find("int main") == -1

    def test_is_self_contained_detects_main(self, router):
        assert router._is_self_contained(SELF_CONTAINED_SOURCE) is True
        assert router._is_self_contained(CUDA_KERNEL_EXAMPLE) is False

    def test_glm_code_prompt_contains_full_main(self, router):
        prompt = router._build_kimi_code_prompt(SELF_CONTAINED_SOURCE, patterns=[])
        assert "int main(int argc, char *argv[])" in prompt
        assert "CRITICAL" in prompt and "main()" in prompt

    def test_glm_refine_prompt_does_not_truncate_previous_code(self, router):
        previous_code = SELF_CONTAINED_SOURCE  # pretend GLM echoed it back
        prompt = router._build_kimi_refine_prompt(
            kernel_source=SELF_CONTAINED_SOURCE,
            previous_code=previous_code,
            feedback="fix the warp shuffle mask",
            patterns=[],
        )
        assert "int main(int argc, char *argv[])" in prompt

    def test_deepseek_plan_prompt_does_not_truncate_self_contained_source(self, router):
        prompt = router._build_deepseek_plan_prompt(SELF_CONTAINED_SOURCE, patterns=[])
        assert "int main(int argc, char *argv[])" in prompt

    def test_bare_kernel_still_truncates_at_budget(self, router):
        long_bare_kernel = CUDA_KERNEL_EXAMPLE + ("// pad\n" * 3000)
        assert len(long_bare_kernel) > 6000
        prompt = router._build_kimi_code_prompt(long_bare_kernel, patterns=[])
        assert len(prompt) < len(long_bare_kernel) + 2000  # truncated, not echoed whole


# ── Bug 3: missing local .cuh headers must not be silently ignored ────────────

class TestUnresolvedLocalHeaders:

    def test_detects_missing_cuh(self, router):
        source = '#include "shfl_integral_image.cuh"\n__global__ void k(int* a) {}\n'
        missing = router._unresolved_local_headers(source)
        assert "shfl_integral_image.cuh" in missing

    def test_no_false_positive_for_present_header(self, router, tmp_path):
        # A header that genuinely exists anywhere under sample_kernels/
        # (rglob, not just the same directory) must not be flagged.
        sample_dir = Path(__file__).resolve().parent.parent / "sample_kernels"
        probe = sample_dir / "_test_tmp_header_for_unit_test.cuh"
        probe.write_text("// unit-test probe header\n", encoding="utf-8")
        try:
            source = '#include "_test_tmp_header_for_unit_test.cuh"\n'
            missing = router._unresolved_local_headers(source)
            assert "_test_tmp_header_for_unit_test.cuh" not in missing
        finally:
            probe.unlink()

    def test_glm_code_prompt_warns_about_missing_header(self, router):
        source = (
            '#include "shfl_integral_image.cuh"\n'
            '__global__ void shfl_scan_test(int *data) {}\n'
        )
        prompt = router._build_kimi_code_prompt(source, patterns=[])
        assert "shfl_integral_image.cuh" in prompt
        assert "DROPPED" in prompt


# ── Bug 2: NVIDIA helper_cuda/helper_functions compat shims ───────────────────

class TestFixPortedCodeHelperShims:

    def test_adds_shim_when_helper_symbols_present(self, router):
        code = (
            "#include <hip/hip_runtime.h>\n"
            "int main() {\n"
            "    int dev = findCudaDevice(0, nullptr);\n"
            "    StopWatchInterface *hTimer = NULL;\n"
            "    sdkCreateTimer(&hTimer);\n"
            "    sdkStartTimer(&hTimer);\n"
            "    sdkStopTimer(&hTimer);\n"
            "    float et = sdkGetTimerValue(&hTimer);\n"
            "    return 0;\n"
            "}\n"
        )
        fixed = router._fix_ported_code(code)
        assert "struct StopWatchInterface" in fixed
        assert "static inline int findCudaDevice" in fixed
        assert "static inline void sdkCreateTimer" in fixed

    def test_no_shim_when_helper_symbols_absent(self, router):
        code = "#include <hip/hip_runtime.h>\n__global__ void kernel() { return; }\n"
        fixed = router._fix_ported_code(code)
        assert "StopWatchInterface" not in fixed

    def test_shim_not_duplicated_on_second_pass(self, router):
        code = (
            "#include <hip/hip_runtime.h>\n"
            "int main() { findCudaDevice(0, nullptr); return 0; }\n"
        )
        first = router._fix_ported_code(code)
        second = router._fix_ported_code(first)
        # shims were injected (1+ occurrences)
        assert first.count("StopWatchInterface") >= 1
        # re-applying the fix must NOT duplicate the shim block
        assert second.count("StopWatchInterface") == first.count("StopWatchInterface")
