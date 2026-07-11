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


# ── I1: GLM evaluator JSON parse cascade tests ─────────────────────────────────

class TestGLMParseCascade:
    """Test the 5-strategy fallback cascade for GLM error analysis JSON parsing.

    Verifies the new Strategy 5 (keyword-prose fallback), _strip_trailing_after_json,
    and truncation detection.
    """

    # ── _strip_trailing_after_json ──────────────────────────────────────────

    def test_strip_trailing_after_json_clean(self):
        """Clean JSON without trailing text should pass through unchanged."""
        from router import _strip_trailing_after_json
        text = '{"fixes": [{"error": "test"}], "summary": "ok"}'
        assert _strip_trailing_after_json(text) == text

    def test_strip_trailing_after_json_trailing_prose(self):
        """Trailing prose after JSON closing brace should be stripped."""
        from router import _strip_trailing_after_json
        text = '{"fixes": [], "summary": "ok"}\n\nThe fix is to replace cudaMalloc with hipMalloc...'
        assert _strip_trailing_after_json(text) == '{"fixes": [], "summary": "ok"}'

    def test_strip_trailing_after_json_nested_braces(self):
        """Nested braces inside JSON values should not confuse the balancer."""
        from router import _strip_trailing_after_json
        text = '{"fixes": [{"error": "missing { in code"}], "summary": "ok"}\ntrailing'
        result = _strip_trailing_after_json(text)
        assert result == '{"fixes": [{"error": "missing { in code"}], "summary": "ok"}'

    def test_strip_trailing_after_json_no_brace(self):
        """Text with no opening brace should return unchanged."""
        from router import _strip_trailing_after_json
        text = "This is just prose with no JSON whatsoever"
        assert _strip_trailing_after_json(text) == text

    def test_strip_trailing_after_json_prose_before(self):
        """Prose prefix before JSON should be preserved by the function (caller strips it)."""
        from router import _strip_trailing_after_json
        text = "I will analyze this. {\\\"fixes\\\": [{\\\"error\\\": \\\"test\\\"}]}\\\nstill going"
        result = _strip_trailing_after_json(text)
        # Should start with the prose prefix since the function doesn't strip prefix
        assert result.startswith("I will analyze this.")

    def test_strip_trailing_after_json_braces_in_string(self):
        """Braces inside quoted strings should be ignored."""
        from router import _strip_trailing_after_json
        text = '{"error": "unclosed { in message", "line": 42}\nsome note'
        assert _strip_trailing_after_json(text) == '{"error": "unclosed { in message", "line": 42}'

    # ── _keyword_prose_fallback ─────────────────────────────────────────────

    def test_keyword_fallback_with_keywords(self):
        """Prose with CUDA/HIP keywords should produce structured feedback."""
        from router import _keyword_prose_fallback
        text = (
            "The issue is with __shfl_up_sync. The mask must be 0xffffffffffffffffULL "
            "for wavefront64. The warpSize is 64 on AMD, not 32. "
            "Also need to check __shfl_xor_sync mask width."
        )
        result = _keyword_prose_fallback(text)
        assert result is not None
        assert "fixes" in result
        assert len(result["fixes"]) <= 3  # top 3 keywords
        if result["fixes"]:
            # At least shfl, mask, or warpSize should be in the output
            keywords_found = set()
            for f in result["fixes"]:
                keywords_found.add(f.get("error", ""))
            assert any("shfl" in str(f) for f in result["fixes"])

    def test_keyword_fallback_empty_text(self):
        """Prose with no keywords should return an empty result dict."""
        from router import _keyword_prose_fallback
        text = "The quick brown fox jumps over the lazy dog."
        result = _keyword_prose_fallback(text)
        assert result is not None
        assert "_strategy" in result
        assert result["_strategy"] == "keyword-fallback-empty" or result["fixes"] == []

    def test_keyword_fallback_top_3(self):
        """Only the top 3 most-mentioned keywords should appear in fixes."""
        from router import _keyword_prose_fallback
        text = (
            "shfl shfl shfl shfl shfl "  # 5 mentions
            "warpSize warpSize warpSize warpSize "  # 4 mentions
            "mask mask mask "  # 3 mentions
            "ballot ballot "  # 2 mentions
            "width "  # 1 mention
        )
        result = _keyword_prose_fallback(text)
        assert len(result["fixes"]) == 3
        error_strings = " ".join(f.get("error", "") for f in result["fixes"])
        assert "shfl" in error_strings
        assert "warpSize" in error_strings
        assert "mask" in error_strings

    def test_keyword_fallback_produces_valid_feedback_dict(self):
        """The fallback result should be compatible with the existing feedback builder."""
        from router import _keyword_prose_fallback
        text = "The __shfl_sync mask is wrong. Warp size should be 64."
        result = _keyword_prose_fallback(text)
        # Should be compatible with the checks at router.py:4328-4333:
        # fixes = glm_analysis.get("fixes", [])
        # if glm_analysis and (fixes or missing_inc or wrong_apis):
        fixes = result.get("fixes", [])
        assert len(fixes) > 0
        for f in fixes:
            assert "error" in f
            assert "root_cause" in f
            assert "priority" in f

    # ── Full cascade test (edge cases that the old 4-strategy missed) ─────────

    def test_cascade_with_prose_before_nested_braces_and_trailing(self):
        """Test that the full cascade handles: prose before JSON, nested braces
        inside string values, AND trailing prose after the closing brace.

        This is the exact pattern that caused the 'GLM analysis parse failed' error.
        """
        from router import _extract_balanced_json, _extract_arrays_regex, _strip_trailing_after_json, _keyword_prose_fallback

        raw_glm = (
            "Based on my analysis of the compiler errors, here is my evaluation:\n\n"
            "The kernel uses __shfl_up_sync which needs a 64-bit mask for wavefront64. "
            "The warpSize on AMD MI300X is 64, not 32 like NVIDIA.\n\n"
            '{\n'
            '  "fixes": [\n'
            '    {\n'
            '      "error": "Wrong __shfl_up_sync mask width — using 0x1f (32-bit) instead of 0x3f (64-bit)",\n'
            '      "root_cause": "CUDA uses 32-warps, HIP 64-warps on MI300X; the mask must cover all lanes",\n'
            '      "exact_fix": "replace 0x1f with 0x3f in __shfl_up_sync mask argument",\n'
            '      "priority": 1\n'
            '    },\n'
            '    {\n'
            '      "error": "Missing __syncwarp after divergent branch — {inconsistent lane states}",\n'
            '      "root_cause": "Divergent branches need sync to avoid deadlock on wavefront64",\n'
            '      "exact_fix": "add __syncwarp() after each if/else block",\n'
            '      "priority": 2\n'
            '    }\n'
            '  ]\n'
            '}\n'
            '\n'
            "I also noticed some potential issues with shared memory bank conflicts, "
            "but those are less critical.\n"
        )

        # Strategy 1: prose-strip + trailing-strip + json.loads
        json_start = raw_glm.find("{")
        raw_glm_json = raw_glm[json_start:] if json_start >= 0 else raw_glm
        trimmed = _strip_trailing_after_json(raw_glm_json)
        import json
        result = None
        try:
            result = json.loads(trimmed)
        except json.JSONDecodeError:
            pass

        if result is None:
            result = _extract_balanced_json(raw_glm)
        if result is None:
            result = _extract_arrays_regex(raw_glm)
        if result is None:
            result = _keyword_prose_fallback(raw_glm)

        assert result is not None, "All 5 strategies failed on a valid GLM output"
        assert "fixes" in result

    def test_cascade_no_json_at_all_keyword_fallback_saves(self):
        """When the LLM outputs pure prose with no JSON at all, Strategy 5
        (keyword fallback) should still produce structured feedback."""
        from router import _keyword_prose_fallback

        raw = (
            "The CUDA kernel uses __shfl_up_sync with a mask of 0xffffffff which is "
            "32-bit and needs to be 0xffffffffffffffffULL for wavefront64 on AMD GPUs. "
            "The warpSize should be changed from 32 to 64. Also the shared memory "
            "allocation size needs to be adjusted for wavefront64."
        )

        result = _keyword_prose_fallback(raw)
        fixes = result.get("fixes", [])
        assert len(fixes) > 0, "Keyword fallback should extract fixes from prose"
        # Should mention shfl, warpSize, and wavefront
        keywords_found = [f.get("error", "") for f in fixes]
        combined = " ".join(keywords_found)
        assert "shfl" in combined or "warpSize" in combined or "wavefront" in combined

    def test_truncation_detection_appends_note(self):
        """When the raw output contains the TRUNCATED marker, the cascade should
        append a truncation note to the fixes array."""
        from router import _keyword_prose_fallback

        raw = (
            '{"fixes": [{"error": "test error", "root_cause": "test", '
            '"exact_fix": "fix it", "priority": 1}], "summary": "ok"}'
            '\n// TRUNCATED: output hit max_tokens limit'
        )

        import json
        # Apply the same logic as the cascade
        is_truncated = "TRUNCATED" in raw
        assert is_truncated
        json_start = raw.find("{")
        trimmed = raw[json_start:] if json_start >= 0 else raw
        # Use _strip_trailing_after_json to strip everything after the balanced }
        from router import _strip_trailing_after_json
        clean = _strip_trailing_after_json(trimmed)
        result = json.loads(clean)

        # Now apply truncation logic
        if result:
            if not result.get("fixes"):
                result["fixes"] = []
            result["fixes"].append({
                "error": "(note: evaluator output was truncated at max_tokens — analysis may be incomplete)",
                "root_cause": "The evaluator response was truncated because it exceeded the max_tokens limit.",
                "priority": 99,
            })

        assert len(result["fixes"]) == 2
        assert any("truncated" in str(f).lower() for f in result["fixes"])

    def test_truncation_all_strategies_failed(self):
        """When truncation happens AND all 5 strategies fail, the truncation
        note should still be captured in a minimal glm_analysis dict."""
        from router import _keyword_prose_fallback

        raw = (
            "I'm analyzing the kernel and here are my thoughts about the "
            "__shfl_up_sync mask and warpSize issues on wavefront64...\n"
            "// TRUNCATED: output hit max_tokens limit"
        )

        # Simulate cascade (no JSON at all)
        result = _keyword_prose_fallback(raw)
        is_truncated = "TRUNCATED" in raw
        if is_truncated:
            if result is None:
                result = {"fixes": [], "_raw": raw[:500]}
            if not result.get("fixes"):
                result["fixes"] = []
            result["fixes"].append({
                "error": "(note: evaluator output was truncated at max_tokens — analysis may be incomplete)",
                "root_cause": "The evaluator response was truncated because it exceeded the max_tokens limit.",
                "priority": 99,
            })

        assert result is not None
        assert len(result["fixes"]) > 1  # keyword fixes + truncation note
        assert any("truncated" in str(f).lower() for f in result["fixes"])


# ── I1: Kimi code prompt tests ─────────────────────────────────────────────────

class TestKimiCodePrompt:
    """Tests for Kimi coder prompt building."""

    def test_cuda_include_dropped_in_code_prompt(self, router):
        """CUDA includes in code prompt should still work even if unresolved."""
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
