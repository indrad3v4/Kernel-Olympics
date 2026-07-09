"""Tests for CUDAKernelParser — spec auto-generation from CUDA source.

Tests the TRIZ #13 (Inversion) loop: parse source BEFORE compile, so the
harness is always correct on the first try instead of guessing and failing.
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from verification.spec_parser import (
    parse_kernel_signatures,
    generate_spec_from_source,
    save_spec,
    auto_generate_spec,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_KERNEL = """
#include <cuda_runtime.h>
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""

MULTI_KERNEL = """
__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL) {
    int tid = threadIdx.x;
}

__global__ void uniform_add(int *data, int *partial_sums, int len) {
    for (int i = 0; i < len; i++) data[i] += partial_sums[i];
}

int main(int argc, char *argv[]) {
    return 0;
}
"""

DEFAULT_VALUES = """
__global__ void kernel_with_defaults(float* output, int n = 256, float init = 0.0f) {
    int idx = threadIdx.x;
    output[idx] = init;
}
"""

NO_GLOBAL = """
#include <stdio.h>
int main() { printf("hello\\n"); return 0; }
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestParseKernelSignatures:

    def test_parses_simple_kernel(self):
        sigs = parse_kernel_signatures(SIMPLE_KERNEL)
        assert len(sigs) == 1
        assert sigs[0]["kernel_function"] == "vector_add"
        assert len(sigs[0]["params"]) == 4
        assert sigs[0]["params"][0]["name"] == "a"
        assert sigs[0]["params"][0]["direction"] == "in"

    def test_parses_multiple_kernels(self):
        sigs = parse_kernel_signatures(MULTI_KERNEL)
        assert len(sigs) == 2
        assert sigs[0]["kernel_function"] == "shfl_scan_test"
        assert sigs[1]["kernel_function"] == "uniform_add"

    def test_strips_block_comments(self):
        src = "/* comment */ __global__ void foo(int x) {}"
        sigs = parse_kernel_signatures(src)
        assert len(sigs) == 1

    def test_handles_default_values(self):
        sigs = parse_kernel_signatures(DEFAULT_VALUES)
        assert len(sigs) == 1
        # Default values should be stripped from param names
        assert sigs[0]["params"][1]["name"] == "n"
        assert sigs[0]["params"][1]["type"] == "int"

    def test_returns_empty_for_no_global(self):
        sigs = parse_kernel_signatures(NO_GLOBAL)
        assert len(sigs) == 0

    def test_infers_pointer_directions(self):
        sigs = parse_kernel_signatures(SIMPLE_KERNEL)
        assert sigs[0]["params"][0]["direction"] == "in"   # a
        assert sigs[0]["params"][1]["direction"] == "in"   # b
        assert sigs[0]["params"][2]["direction"] == "out"  # c
        assert sigs[0]["params"][3]["direction"] == "scalar"  # n

    def test_parses_int_pointer_args(self):
        """nvidia_shfl_scan has int* args — verify they parse correctly."""
        sigs = parse_kernel_signatures(MULTI_KERNEL)
        assert sigs[0]["params"][0]["type"] == "int*"
        assert sigs[0]["params"][0]["name"] == "data"
        assert sigs[0]["params"][1]["name"] == "width"
        assert sigs[0]["params"][2]["name"] == "partial_sums"


class TestGenerateSpecFromSource:

    def test_generates_complete_spec(self):
        spec = generate_spec_from_source("vector_add", SIMPLE_KERNEL)
        assert spec is not None
        assert spec["kernel_name"] == "vector_add"
        assert spec["kernel_function"] == "vector_add"
        assert len(spec["params"]) == 4
        assert "launch" in spec
        assert "grid" in spec["launch"]
        assert "block" in spec["launch"]

    def test_spec_has_size_expr_on_pointers(self):
        spec = generate_spec_from_source("vector_add", SIMPLE_KERNEL)
        ptr_params = [p for p in spec["params"] if "*" in p["type"]]
        assert all(p.get("size_expr") for p in ptr_params)

    def test_detects_self_contained_kernel(self):
        spec = generate_spec_from_source("shfl_scan", MULTI_KERNEL)
        assert spec is not None
        assert spec.get("self_contained") is True

    def test_returns_none_for_no_kernel(self):
        assert generate_spec_from_source("empty", NO_GLOBAL) is None


class TestAutoGenerateSpec:

    def test_saves_to_spec_dir(self, tmp_path):
        # Temporarily redirect _SPEC_DIR to a temp path
        from verification import spec_parser
        orig_dir = spec_parser._SPEC_DIR
        try:
            test_dir = tmp_path / "specs"
            spec_parser._SPEC_DIR = test_dir
            spec = auto_generate_spec("test_kernel", SIMPLE_KERNEL)
            assert spec is not None
            assert (test_dir / "test_kernel.json").exists()
        finally:
            spec_parser._SPEC_DIR = orig_dir

    def test_auto_generates_nvidia_shfl_scan_spec(self, tmp_path):
        """End-to-end: the actual kernel that triggered the harness abort."""
        from verification import spec_parser
        orig_dir = spec_parser._SPEC_DIR
        try:
            spec_parser._SPEC_DIR = tmp_path / "specs_tmp"
            with open("sample_kernels/cuda/nvidia_shfl_scan.cu") as f:
                src = f.read()
            spec = auto_generate_spec("nvidia_shfl_scan", src)
            assert spec is not None
            assert spec["kernel_function"] == "shfl_scan_test"
            assert len(spec["params"]) == 3
            assert spec["params"][0]["type"] == "int*"
            assert spec["params"][1]["type"] == "int"
            assert spec["params"][2]["type"] == "int*"
        finally:
            spec_parser._SPEC_DIR = orig_dir
