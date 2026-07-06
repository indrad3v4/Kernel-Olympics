"""Unit tests for the porting agent module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from porting_agent.agent import PortingAgent


def test_template_port_no_api_key():
    """Without API key, porting should use template fallback."""
    agent = PortingAgent(api_key="test")  # Triggers template fallback
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    result = agent.port_kernel(warp_source)
    assert result["confidence"] >= 80
    assert "ported_code" in result
    assert "changes" in result
    assert len(result["changes"]) > 0


def test_template_port_adds_header():
    """Template porting should add wavefront awareness comment."""
    agent = PortingAgent(api_key="test")
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    result = agent.port_kernel(warp_source)
    assert "ROCm/HIP port" in result["ported_code"]
    assert "wavefront" in result["ported_code"] or "wavefront" in str(result["changes"])


def test_template_port_with_cached_pattern():
    """With cached pattern, template port confidence should increase."""
    agent = PortingAgent(api_key="test")
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    result = agent.port_kernel(
        warp_source,
        cached_pattern={
            "confidence": 0.90,
            "id": "abc123",
            "verified_fix": "// Verified fix for warp reduce\n__shared__ float shared[64];\n..."
        }
    )
    assert result["confidence"] >= 90  # 85 base + 5 from cached = 90
    assert "Applied cached pattern" in str(result["changes"])


def test_template_port_minimal_code():
    """Safe code should still get a template porting header."""
    agent = PortingAgent(api_key="test")
    safe_code = "__global__ void safe(float* a, float* b) { *a = *b; }"
    result = agent.port_kernel(safe_code)
    assert result["confidence"] == 85
    assert result["ported_code"] is not None


def test_porting_result_format():
    """Porting result should have all required fields."""
    agent = PortingAgent(api_key="test")
    result = agent.port_kernel("__global__ void k(float* a) { *a = 1.0f; }")
    assert "ported_code" in result
    assert "confidence" in result
    assert "changes" in result
    assert "explanation" in result
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["changes"], list)
    assert isinstance(result["explanation"], str)
