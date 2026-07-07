"""Unit tests for the risk classifier module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_classifier.classifier import RiskClassifier, DANGER_PATTERNS


def test_classifier_init():
    '''Classifier should initialize with 9 patterns loaded (5 original + 4 new).'''
    c = RiskClassifier()
    assert len(c.patterns) == 9
    assert len(c.pattern_counters) == 9
    assert c.total_scans == 0


def test_classify_green_kernel():
    """Safe kernel (no warp patterns) should return green."""
    code = """
__global__ void safe_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    c = RiskClassifier()
    result = c.classify(code, "safe.cu")
    assert result["risk_level"] == "green"
    assert len(result["findings"]) == 0


def test_classify_red_kernel_warp():
    """warp_reduce.cu should be classified RED."""
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    c = RiskClassifier()
    result = c.classify(warp_source, "warp_reduce.cu")
    assert result["risk_level"] == "red", f"Expected RED, got {result['risk_level']}"
    assert len(result["findings"]) >= 3
    assert result['total_patterns_checked'] == 9


def test_classify_yellow_kernel_transpose():
    """transpose.cu should be classified YELLOW (medium + low severity, no high)."""
    trans_source = open("sample_kernels/cuda/transpose.cu").read()
    c = RiskClassifier()
    result = c.classify(trans_source, "transpose.cu")
    assert result["risk_level"] == "yellow", f"Expected YELLOW, got {result['risk_level']}"
    assert len(result["findings"]) >= 2


def test_classify_red_kernel_histogram():
    """histogram.cu should be classified RED."""
    hist_source = open("sample_kernels/cuda/histogram.cu").read()
    c = RiskClassifier()
    result = c.classify(hist_source, "histogram.cu")
    assert result["risk_level"] == "red", f"Expected RED, got {result['risk_level']}"
    # Should catch: WARP_SIZE=32, 0x1f mask, __shfl_xor_sync
    assert len(result["findings"]) >= 3


def test_conv2d_kernel_classify_red():
    """conv2d.cu should be classified RED (shfl_xor, warp mask, #define TILE_SIZE 32)."""
    conv_source = open("sample_kernels/cuda/conv2d.cu").read()
    c = RiskClassifier()
    result = c.classify(conv_source, "conv2d.cu")
    assert result["risk_level"] == "red", f"Expected RED, got {result['risk_level']}"
    # Should catch: __shfl_xor_sync (#define TILE_SIZE is not a pattern check, but __shfl_xor is)
    assert len(result["findings"]) >= 3


def test_findings_have_context():
    """Each finding should include surrounding source context."""
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    c = RiskClassifier()
    result = c.classify(warp_source, "warp_reduce.cu")
    for f in result["findings"]:
        assert f["line"] > 0
        assert f["pattern"] in [p[0] for p in DANGER_PATTERNS]
        assert len(f["context"]) > 0
        assert f["severity"] in ("high", "medium", "low")
        assert f["description"] is not None


def test_severity_mapping():
    '''High: shfl_down_sync, shfl_xor_sync, match_all_sync. Medium: warp_size, shared_mem, all_any_sync, activemask, warp_lane_shift.'''
    c = RiskClassifier()
    assert c._severity('shfl_down_sync') == 'high'
    assert c._severity('shfl_xor_sync') == 'high'
    assert c._severity('match_all_sync') == 'high'
    assert c._severity('warp_size_constant') == 'medium'
    assert c._severity('shared_mem_warp_tiling') == 'medium'
    assert c._severity('all_any_sync') == 'medium'
    assert c._severity('activemask') == 'medium'
    assert c._severity('warp_lane_shift') == 'medium'
    assert c._severity('syncwarp') == 'low'


def test_pattern_counter_tracking():
    """Pattern counters should track total matches across calls."""
    c = RiskClassifier()
    c.reset_counters()
    warp = open("sample_kernels/cuda/warp_reduce.cu").read()
    trans = open("sample_kernels/cuda/transpose.cu").read()

    c.classify(warp, "warp_reduce.cu")
    c.classify(trans, "transpose.cu")

    # warp_size_constant should be matched in at least warp_reduce
    assert c.pattern_counters["warp_size_constant"] >= 1
    # shared_mem_warp_tiling matched in warp_reduce
    assert c.pattern_counters["shared_mem_warp_tiling"] >= 1
    assert c.total_scans == 2


def test_classify_batch():
    """Batch classify should return results for all files."""
    sources = {
        "sample_kernels/cuda/warp_reduce.cu": open("sample_kernels/cuda/warp_reduce.cu").read(),
        "sample_kernels/cuda/transpose.cu": open("sample_kernels/cuda/transpose.cu").read(),
    }
    c = RiskClassifier()
    c.reset_counters()
    results = c.classify_batch(sources)
    assert len(results) == 2
    levels = {r["risk_level"] for r in results}
    assert "red" in levels  # warp_reduce is RED


def test_reset_counters():
    """reset_counters should clear state."""
    c = RiskClassifier()
    warp = open("sample_kernels/cuda/warp_reduce.cu").read()
    c.classify(warp, "warp_reduce.cu")
    assert c.total_scans > 0
    c.reset_counters()
    assert c.total_scans == 0
    assert all(v == 0 for v in c.pattern_counters.values())
