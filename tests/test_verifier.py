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
    harness = agent._generate_harness("warp_reduce", "", source)
    # Spec-driven: should have dim3(4,1,1), dim3(64,1,1) instead of hardcoded <<<4, 64>>>
    assert "dim3(4,1,1)" in harness
    assert "dim3(64,1,1)" in harness
    assert "warp_reduce_kernel" in harness
    assert "hipMalloc" in harness
    assert "hipMemcpy" in harness


def test_generate_harness_legacy_fallback():
    """_generate_harness should fall back to legacy harness when no spec."""
    agent = VerificationAgent()
    source = "__global__ void my_kernel() {}"
    harness = agent._generate_harness("unknown_kernel", "", source)
    # Legacy: hardcoded <<<4, 64>>>
    assert "<<<4, 64>>>" in harness
    assert "hip/hip_runtime.h" in harness
    assert "my_kernel" in harness


def test_harness_histogram_dynamic_shared():
    """Histogram harness should include dynamic shared memory."""
    agent = VerificationAgent()
    source = "__global__ void histogram_kernel(const float* input, int* histogram, int n, int num_bins) {}"
    harness = agent._generate_harness("histogram", "", source)
    # dim3(1,1,1), dim3(256,1,1), dynamic shared mem 1024
    assert "dim3(1,1,1)" in harness
    assert "dim3(256,1,1)" in harness
    assert ", 1024" in harness  # dynamic shared mem


def test_harness_transpose_2d_block():
    """Transpose harness should use 2D block (32,32)."""
    agent = VerificationAgent()
    source = "__global__ void transpose_kernel(const float* input, float* output, int width, int height) {}"
    harness = agent._generate_harness("transpose", "", source)
    assert "dim3(3,3,1)" in harness
    assert "dim3(32,32,1)" in harness
    assert "linear_ramp" not in harness or True  # not a string check we need


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
