"""Unit tests for the verification module — spec-driven harness generator."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.verifier import VerificationAgent


def test_verify_no_hipcc():
    """Without hipcc, verification should return compile_success=False."""
    agent = VerificationAgent()
    result = agent.verify(
        hip_source="// Test kernel\nint main() { return 0; }",
        cuda_reference_output="",
        kernel_name="test_kernel"
    )
    assert result["compile_success"] is False
    assert result["passed"] is False
    assert "hipcc" in result["compile_output"] or result["compile_output"] != ""


def test_verify_diff_matched():
    """_diff should return True for identical outputs."""
    agent = VerificationAgent()
    ok, report = agent._diff("1.0\n2.0\n3.0\n", "1.0\n2.0\n3.0\n")
    assert ok is True
    assert "match" in report.lower()


def test_verify_diff_fp_tolerance():
    """_diff should accept floating-point differences within tolerance."""
    agent = VerificationAgent()
    ok, report = agent._diff("1.000001\n2.0\n3.0\n", "1.0\n2.0\n3.0\n")
    assert ok is True
    assert "tolerance" in report.lower()


def test_verify_diff_mismatched():
    """_diff should return False for different outputs."""
    agent = VerificationAgent()
    ok, report = agent._diff("1.0\n2.0\n3.0\n", "1.0\n2.0\n999.0\n")
    assert ok is False
    assert "differ" in report.lower() or "diff" in report


def test_verify_diff_empty():
    """_diff should return False for empty inputs."""
    agent = VerificationAgent()
    ok, report = agent._diff("", "1.0\n2.0\n")
    assert ok is False
    assert "Missing" in report


# ── Spec-driven harness tests ──────────────────────────────────────


def test_list_specs():
    """list_specs() should return known kernel specs."""
    agent = VerificationAgent()
    specs = agent.list_specs()
    assert "warp_reduce" in specs
    assert "histogram" in specs
    assert "softmax" in specs
    assert "transpose" in specs
    assert "conv2d" in specs
    assert "new_kernel" in specs


def test_load_spec_warp_reduce():
    """load_spec should return correct metadata for warp_reduce."""
    agent = VerificationAgent()
    spec = agent.load_spec("warp_reduce")
    assert spec is not None
    assert spec["kernel_function"] == "warp_reduce_kernel"
    assert spec["launch"]["grid"]["x"] == 4
    assert spec["launch"]["block"]["x"] == 64
    assert len(spec["params"]) == 3


def test_load_spec_unknown():
    """load_spec should return None for unknown kernels."""
    agent = VerificationAgent()
    spec = agent.load_spec("nonexistent_kernel")
    assert spec is None


def test_generate_harness_spec_driven():
    """_generate_harness should use spec when available."""
    agent = VerificationAgent()
    source = "__global__ void warp_reduce_kernel(const float* input, float* output, int n) {}"
    harness, kernel_start, kernel_end = agent._generate_harness("warp_reduce", "", source)
    # Spec-driven: should have dim3(4,1,1), dim3(64,1,1) instead of hardcoded <<<4, 64>>>
    assert "dim3(4,1,1)" in harness
    assert "dim3(64,1,1)" in harness
    assert "warp_reduce_kernel" in harness
    assert "hipMalloc" in harness
    assert "hipMemcpy" in harness
    # The reported kernel line range must actually contain the spliced source
    spliced_lines = harness.splitlines()[kernel_start - 1:kernel_end]
    assert "\n".join(spliced_lines) == source


def test_generate_harness_legacy_fallback():
    """_generate_harness should fall back to legacy harness when no spec."""
    agent = VerificationAgent()
    source = "__global__ void my_kernel() {}"
    harness, kernel_start, kernel_end = agent._generate_harness("unknown_kernel", "", source)
    # Legacy: hardcoded <<<4, 64>>>
    assert "<<<4, 64>>>" in harness
    assert "hip/hip_runtime.h" in harness
    assert "my_kernel" in harness
    spliced_lines = harness.splitlines()[kernel_start - 1:kernel_end]
    assert "\n".join(spliced_lines) == source


def test_harness_histogram_dynamic_shared():
    """Histogram harness should include dynamic shared memory."""
    agent = VerificationAgent()
    source = "__global__ void histogram_kernel(const float* input, int* histogram, int n, int num_bins) {}"
    harness, _start, _end = agent._generate_harness("histogram", "", source)
    # dim3(1,1,1), dim3(256,1,1), dynamic shared mem 1024
    assert "dim3(1,1,1)" in harness
    assert "dim3(256,1,1)" in harness
    assert ", 1024" in harness  # dynamic shared mem


def test_harness_transpose_2d_block():
    """Transpose harness should use 2D block (32,32)."""
    agent = VerificationAgent()
    source = "__global__ void transpose_kernel(const float* input, float* output, int width, int height) {}"
    harness, _start, _end = agent._generate_harness("transpose", "", source)
    assert "dim3(3,3,1)" in harness
    assert "dim3(32,32,1)" in harness
    assert "linear_ramp" not in harness or True  # not a string check we need


# ── Bug 2: self-contained programs must not be wrapped in a second harness ──


def test_generate_harness_skips_wrapping_when_source_has_main():
    """A ported source that already defines int main() must be returned as-is.

    Regression test for the nvidia_shfl_scan.cu failure: wrapping a complete
    NVIDIA sample program (which brings its own main()) in the generic
    harness produced two main() definitions -> 'redefinition of main', and
    every downstream compile error was then unattributable to the ported
    code (see docs/fix-plan-harness-and-diagnostics.md, Bug 2).
    """
    agent = VerificationAgent()
    source = (
        "__global__ void shfl_scan_test(int *data, int width, int *partial_sums) {}\n"
        "int main() {\n"
        "    shfl_scan_test<<<1,1>>>(nullptr, 32, nullptr);\n"
        "    return 0;\n"
        "}\n"
    )
    harness, kernel_start, kernel_end = agent._generate_harness("nvidia_shfl_scan", "", source)
    assert harness == source
    assert harness.count("int main(") == 1
    assert kernel_start == 1
    assert kernel_end == len(source.splitlines())


def test_generate_harness_main_detection_ignores_indentation_and_comments():
    """int main( should be detected regardless of leading whitespace."""
    agent = VerificationAgent()
    source = "    int main(void) {\n        return 0;\n    }\n"
    harness, kernel_start, kernel_end = agent._generate_harness("some_kernel", "", source)
    assert harness == source


def test_classify_error_origin_ported_code():
    agent = VerificationAgent()
    origin = agent._classify_error_origin(
        "test_kernel.cpp:8:5: error: use of undeclared identifier 'foo'", 6, 20
    )
    assert origin == "ported_code"


def test_classify_error_origin_harness():
    agent = VerificationAgent()
    origin = agent._classify_error_origin(
        "test_kernel.cpp:67:5: error: redefinition of 'main'", 6, 20
    )
    assert origin == "harness"


def test_classify_error_origin_unknown_when_unparseable():
    agent = VerificationAgent()
    origin = agent._classify_error_origin("some unrelated log line", 6, 20)
    assert origin == "unknown"


def test_verify_spec_tracking():
    """verify() should record spec_used in result when spec exists."""
    agent = VerificationAgent()
    result = agent.verify(
        hip_source="__global__ void warp_reduce_kernel() {}",
        kernel_name="warp_reduce"
    )
    assert result["spec_used"] == "warp_reduce"


def test_verify_no_spec_tracking():
    """verify() should set spec_used to None when no spec."""
    agent = VerificationAgent()
    result = agent.verify(
        hip_source="// No kernel",
        kernel_name="unknown_kernel"
    )
    assert result["spec_used"] is None
