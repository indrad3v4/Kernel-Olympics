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
    """Bug 1: Kimi must see a self-contained program's own main(), even when
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

    def test_kimi_code_prompt_contains_full_main(self, router):
        prompt = router._build_kimi_code_prompt(SELF_CONTAINED_SOURCE, patterns=[])
        assert "int main(int argc, char *argv[])" in prompt
        assert "CRITICAL" in prompt and "main()" in prompt

    def test_kimi_refine_prompt_does_not_truncate_previous_code(self, router):
        previous_code = SELF_CONTAINED_SOURCE  # pretend Kimi echoed it back
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

    def test_kimi_code_prompt_warns_about_missing_header(self, router):
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
        once = router._fix_ported_code(code)
        twice = router._fix_ported_code(once)
        assert twice.count("struct StopWatchInterface") == 1

    def test_shim_compiles_standalone_hipSetDevice_declared_before_use(self, router):
        """Regression for the 2026-07-09 notebook run (col-56 errors).

        403601d moved the shim BEFORE the code's first #include so its
        symbols are declared before use — but the shim's findCudaDevice body
        calls hipSetDevice, declared only in hip/hip_runtime.h, which then
        comes AFTER the shim. Every self-contained port failed with
        'use of undeclared identifier' at the exact column of hipSetDevice
        (56), re-injected on every refine iteration. The shim must be
        position-independent: it includes hip/hip_runtime.h itself.
        """
        # NVIDIA samples open with a long copyright banner before includes.
        code = (
            "/*\n * Copyright NVIDIA...\n */\n"
            "#include <hip/hip_runtime.h>\n"
            "__global__ void k(int* d) {}\n"
            "int main() { findCudaDevice(0, nullptr); return 0; }\n"
        )
        fixed = router._fix_ported_code(code)
        use_pos = fixed.find("hipSetDevice")
        assert use_pos != -1
        decl_pos = fixed.find("#include <hip/hip_runtime.h>")
        assert decl_pos != -1 and decl_pos < use_pos, (
            "shim calls hipSetDevice before any hip/hip_runtime.h include — "
            "this is the 'use of undeclared identifier' at col 56")
        assert "#pragma once" not in fixed  # meaningless in a main file; clang warns

    def test_warpSize_substitution_must_not_corrupt_define(self, router):
        """Regression for the 2026-07-09 notebook run (29:9 macro error).

        Kimi plausibly emits '#define warpSize 64'; the blanket
        warpSize→64 regex turned it into '#define 64 64' →
        'error: macro name must be an identifier' at col 9.
        """
        code = (
            "#include <hip/hip_runtime.h>\n"
            "#define warpSize 64\n"
            "__global__ void k(int* d) { int lane = threadIdx.x % warpSize; }\n"
        )
        fixed = router._fix_ported_code(code)
        assert "#define 64" not in fixed
        # The non-define use should still be substituted (or resolve to 64
        # via the surviving macro) — either way, no corrupted macro name.
        assert "macro" not in fixed  # sanity: no error text leaked in
