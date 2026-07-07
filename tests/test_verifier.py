"""Unit tests for the verification module."""
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
    assert "hipcc not found locally" in result["compile_output"] or \
           "hipcc" in result["compile_output"] or \
           result["compile_output"] != ""


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


def test_generate_harness():
    """Test harness should be valid C++ with hip includes."""
    agent = VerificationAgent()
    harness = agent._generate_harness("my_kernel", "", "__global__ void warp_reduce_kernel() {}")
    assert "hip/hip_runtime.h" in harness
    assert "warp_reduce_kernel" in harness
    assert "hipMalloc" in harness
    assert "hipMemcpy" in harness
