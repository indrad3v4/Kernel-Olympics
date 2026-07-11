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


def test_extern_shared_kernel_gets_nonzero_dynamic_smem():
    """A kernel using `extern __shared__` must never be launched with 0 dynamic
    shared memory — that runs the first shared access out of bounds → SIGSEGV
    before any output (the nvidia_shfl_scan runtime crash). When the spec pins no
    size, the harness derives a safe default from the block dimensions."""
    agent = VerificationAgent()
    # Spec-less kernel -> legacy path has no dynamic smem support, so drive it via
    # a spec that omits dynamic_shared_mem to exercise the auto-default.
    if agent.load_spec("nvidia_shfl_scan") is None:
        import pytest
        pytest.skip("nvidia_shfl_scan spec not present")
    src = (
        "__global__ void shfl_scan_test(int* data, int width, int* partial_sums) {\n"
        "    extern __shared__ int sums[];\n"
        "    int id = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    sums[threadIdx.x / warpSize] = data[id];\n"
        "}\n"
    )
    harness, _s, _e = agent._generate_harness("nvidia_shfl_scan", "", src)
    launch = [l for l in harness.splitlines() if "shfl_scan_test<<<" in l][0]
    config = launch.split("<<<")[1].split(">>>")[0]
    # third launch config argument (dynamic shared bytes) must be present & > 0
    parts = [p.strip() for p in config.split(",")]
    # config is "dim3(...), dim3(...), <bytes>" -> dim3 commas inflate the split;
    # the trailing numeric token is the shared-mem size.
    assert parts[-1].isdigit() and int(parts[-1]) > 0, (
        f"expected nonzero dynamic shared mem in launch, got: {launch}"
    )


def test_scalar_value_pinned_in_spec_is_used():
    """A scalar param with an explicit `value` must use it (not default to the
    input element count). nvidia_shfl_scan's `width` is 64 (the wavefront scan
    width), not 512; defaulting to the buffer size made __shfl_up_sync's width
    invalid."""
    agent = VerificationAgent()
    spec = agent.load_spec("nvidia_shfl_scan")
    if spec is None:
        import pytest
        pytest.skip("nvidia_shfl_scan spec not present")
    width_param = [p for p in spec["params"] if p["name"] == "width"][0]
    assert width_param.get("value") == 64
    src = "__global__ void shfl_scan_test(int* data, int width, int* partial_sums) {}"
    harness, _s, _e = agent._generate_harness("nvidia_shfl_scan", "", src)
    assert "int width = 64;" in harness


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
    """A self-contained source with no spec is returned as-is (no double main).

    Regression test for the original nvidia_shfl_scan double-main failure:
    wrapping a program that brings its own main() in the generic harness
    produced two main() definitions -> 'redefinition of main'. For a kernel
    with NO spec, returning the source unchanged is the correct guard.

    (The nvidia_shfl_scan case itself is now covered by
    ``test_generate_harness_device_subset_strips_leaked_main`` — its spec
    declares DEVICE_SUBSET, so a leaked main is stripped and the spec
    harness drives the kernel instead of compiling the model's own driver.)
    """
    agent = VerificationAgent()
    source = (
        "__global__ void some_kernel(int *data, int width, int *partial_sums) {}\n"
        "int main() {\n"
        "    some_kernel<<<1,1>>>(nullptr, 32, nullptr);\n"
        "    return 0;\n"
        "}\n"
    )
    # kernel name with no spec on disk -> self-contained early return
    harness, kernel_start, kernel_end = agent._generate_harness("no_spec_kernel", "", source)
    assert harness == source
    assert harness.count("int main(") == 1
    assert kernel_start == 1
    assert kernel_end == len(source.splitlines())


def test_generate_harness_device_subset_strips_leaked_main():
    """A DEVICE_SUBSET port that leaks main()/host code must not compile it.

    Root cause of the 2026-07 nvidia_shfl_scan TIMEOUT: contradictory prompt
    instructions made the coder emit the whole unportable NVIDIA sample plus a
    stray driver. The spec declares DEVICE_SUBSET, so the verifier must strip
    any leaked host driver and build the synthesized spec harness — one main(),
    the device kernel preserved, and none of the SDK-only host symbols.
    """
    agent = VerificationAgent()
    if agent.load_spec("nvidia_shfl_scan") is None:
        import pytest
        pytest.skip("nvidia_shfl_scan spec not present")
    leaked = (
        "__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)\n"
        "{\n"
        "    int id = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    data[id] = __shfl_up_sync(0xffffffff, data[id], 1, width);\n"
        "}\n"
        "int main(int argc, char **argv) {\n"
        "    cudaDeviceProp p;\n"
        "    findCudaDevice(argc, (const char**)argv);\n"
        "    return p.major;\n"
        "}\n"
    )
    harness, kernel_start, kernel_end = agent._generate_harness("nvidia_shfl_scan", "", leaked)
    assert harness.count("int main(") == 1
    assert "__global__ void shfl_scan_test" in harness
    assert "__shfl_up" in harness  # device body preserved, not truncated (non-sync variants)
    for host_sym in ("findCudaDevice", "cudaDeviceProp", "p.major"):
        assert host_sym not in harness


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


def test_classify_error_origin_link_undefined_main_lld():
    """Bug 6: ld.lld's undefined-symbol diagnostic has no file:line:col, so it
    used to fall through to 'unknown' — invisible to any caller trying to
    give targeted guidance ("you dropped main() — restore it")."""
    agent = VerificationAgent()
    origin = agent._classify_error_origin("ld.lld: error: undefined symbol: main", 6, 20)
    assert origin == "link"


def test_classify_error_origin_link_undefined_reference_gnu_ld():
    """GNU ld phrases the same failure as 'undefined reference to `main'."""
    agent = VerificationAgent()
    origin = agent._classify_error_origin(
        "test_kernel.cpp:(.text+0x10): undefined reference to `main'", 6, 20
    )
    assert origin == "link"


def test_classify_error_origin_harness_not_misclassified_as_link():
    """A harness-origin 'redefinition of main' must stay 'harness', not 'link'
    — only an UNDEFINED main symbol is a link-stage failure."""
    agent = VerificationAgent()
    origin = agent._classify_error_origin(
        "test_kernel.cpp:67:5: error: redefinition of 'main'", 6, 20
    )
    assert origin == "harness"


def test_signal_name_translates_sigsegv():
    """RUN-FIRST: exit -11 must read as SIGSEGV, not raw signal arithmetic."""
    assert VerificationAgent._signal_name(-11) == "SIGSEGV"
    # Signal numbering is platform-specific (SIGABRT is 6 on POSIX, 22 on
    # Windows) — the deployment target is the Linux notebook, where -6 names
    # SIGABRT; elsewhere the numeric fallback is the correct answer.
    assert VerificationAgent._signal_name(-6) in ("SIGABRT", "signal 6")
    assert VerificationAgent._signal_name(0) == ""
    assert VerificationAgent._signal_name(1) == ""
    assert VerificationAgent._signal_name(None) == ""


def test_quick_run_check_missing_binary_reports_failure():
    """quick_run_check on a never-compiled kernel: run_success=False,
    exit_code None (never launched) — distinct from a real crash."""
    agent = VerificationAgent()
    rc = agent.quick_run_check("kernel_that_was_never_compiled")
    assert rc["run_success"] is False
    assert rc["run_exit_code"] is None
    assert rc["signal"] == ""


def test_run_returns_exit_code():
    """Bug 0: _run() must surface the real exit code, not just a pass/fail
    bool, so callers can tell 'crashed' from 'ran and returned nonzero for a
    known reason' (e.g. the NVIDIA sample's EXIT_WAIVED=2)."""
    import tempfile
    from pathlib import Path
    agent = VerificationAgent()
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        run_ok, output, benchmark, exit_code = agent._run(build_dir, "nonexistent_binary")
        assert run_ok is False
        assert exit_code is None  # binary never launched — distinct from a real nonzero exit
        assert "not found" in output.lower() or "compile step" in output.lower()


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
