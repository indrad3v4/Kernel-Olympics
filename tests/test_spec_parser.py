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

INT_OUTPUT_KERNEL = """
__global__ void copy_kernel(const int* in, int* out, int n) {
    int idx = threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}
"""

COMMENTED_MAIN = """
__global__ void k(float* a) {}
// TODO: port int main() from original file
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

    def test_omits_fabricated_fields_when_self_contained(self):
        """Bug 4: launch/input_setup/output_readback are dead config once
        _generate_harness() (verifier.py) sees self_contained=True — it
        returns the ported source unwrapped and never consults them. Leaving
        a hardcoded (float*,float*,int) guess on disk for a kernel whose real
        params are (int*,int,int*) is misleading, not harmless."""
        spec = generate_spec_from_source("shfl_scan", MULTI_KERNEL)
        assert spec is not None
        assert spec.get("self_contained") is True
        assert "launch" not in spec
        assert "input_setup" not in spec
        assert "output_readback" not in spec

    def test_infers_int_element_type_from_output_param(self):
        """Bug 4: output_readback.element_type must come from the kernel's
        own output-direction param, not a hardcoded 'float' — a spec
        claiming float for an all-int* kernel contradicts its own params."""
        spec = generate_spec_from_source("copy_kernel", INT_OUTPUT_KERNEL)
        assert spec is not None
        assert spec["output_readback"]["element_type"] == "int"
        assert spec["output_readback"]["format"] == "int_per_line"

    def test_float_kernel_still_gets_float_readback(self):
        spec = generate_spec_from_source("vector_add", SIMPLE_KERNEL)
        assert spec["output_readback"]["element_type"] == "float"
        assert spec["output_readback"]["format"] == "float_per_line"

    def test_self_contained_regex_is_anchored_not_a_comment_false_positive(self):
        """Bug 4: the old unanchored re.search(r'int\\s+main\\s*\\(', source)
        matched the literal substring 'int main(' anywhere, including inside
        a comment. Anchoring with ^ + MULTILINE (matching the two call sites
        in verifier.py that already anchor this check) fixes the false
        positive."""
        spec = generate_spec_from_source("k", COMMENTED_MAIN)
        assert spec is not None
        assert spec.get("self_contained") is not True
        assert "launch" in spec  # fabricated-but-still-present since NOT self-contained

    def test_generated_spec_is_marked_auto_generated(self):
        spec = generate_spec_from_source("vector_add", SIMPLE_KERNEL)
        assert spec["auto_generated"] is True


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


class TestSaveSpecOverwriteGuard:
    """Bug 5: _auto_gen_spec() ran unconditionally at the top of every
    route() call with no existence check — silently destroying any
    hand-tuned spec (conv2d.json, softmax.json, new_kernel.json carry real
    reference_output paths that make the verifier's diff step meaningful)
    on the kernel's very next run."""

    def test_refuses_to_overwrite_hand_written_spec(self, tmp_path):
        from verification import spec_parser
        orig_dir = spec_parser._SPEC_DIR
        try:
            spec_parser._SPEC_DIR = tmp_path / "specs"
            spec_parser._SPEC_DIR.mkdir(parents=True)
            hand_written = {"kernel_name": "conv2d",
                             "reference_output": "sample_kernels/reference/conv2d_output.txt"}
            path = spec_parser._SPEC_DIR / "conv2d.json"
            path.write_text(json.dumps(hand_written), encoding="utf-8")

            fresh_spec = {"kernel_name": "conv2d", "auto_generated": True, "params": []}
            saved_path, written = spec_parser.save_spec("conv2d", fresh_spec)

            assert written is False
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            assert on_disk == hand_written  # untouched
        finally:
            spec_parser._SPEC_DIR = orig_dir

    def test_overwrites_previously_auto_generated_spec(self, tmp_path):
        from verification import spec_parser
        orig_dir = spec_parser._SPEC_DIR
        try:
            spec_parser._SPEC_DIR = tmp_path / "specs"
            spec_parser._SPEC_DIR.mkdir(parents=True)
            old_auto = {"kernel_name": "foo", "auto_generated": True, "params": []}
            path = spec_parser._SPEC_DIR / "foo.json"
            path.write_text(json.dumps(old_auto), encoding="utf-8")

            new_spec = {"kernel_name": "foo", "auto_generated": True, "params": [{"name": "x"}]}
            saved_path, written = spec_parser.save_spec("foo", new_spec)

            assert written is True
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            assert on_disk["params"] == [{"name": "x"}]
        finally:
            spec_parser._SPEC_DIR = orig_dir

    def test_auto_generate_spec_marks_persisted_false_when_blocked(self, tmp_path):
        from verification import spec_parser
        orig_dir = spec_parser._SPEC_DIR
        try:
            spec_parser._SPEC_DIR = tmp_path / "specs"
            spec_parser._SPEC_DIR.mkdir(parents=True)
            hand_written = {"kernel_name": "vector_add", "reference_output": "x.txt"}
            path = spec_parser._SPEC_DIR / "vector_add.json"
            path.write_text(json.dumps(hand_written), encoding="utf-8")

            spec = auto_generate_spec("vector_add", SIMPLE_KERNEL)
            assert spec is not None
            assert spec["_persisted"] is False
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            assert on_disk == hand_written  # untouched
        finally:
            spec_parser._SPEC_DIR = orig_dir
