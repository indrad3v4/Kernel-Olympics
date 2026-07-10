"""Tests for the symbol inventory and CUDA→HIP symbol diff."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.symbols import extract_symbols, diff_symbols, SymbolTable


CUDA = """#include <cuda_runtime.h>
#define WARP_SIZE 32
__device__ float warpReduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__global__ void reduce_kernel(float* in, float* out, int n) {
    out[blockIdx.x] = warpReduce(in[threadIdx.x]);
}
int main() { return 0; }
"""

HIP = (CUDA.replace("cuda_runtime.h", "hip/hip_runtime.h")
           .replace("0xffffffff", "0xffffffffffffffffULL")
           .replace("#define WARP_SIZE 32", "#define WAVEFRONT_SIZE 64"))


class TestExtractSymbols:
    def test_extracts_kernels_helpers_functions_macros_includes(self):
        t = extract_symbols(CUDA)
        assert t.kernels == ["reduce_kernel"]
        assert t.helpers == ["warpReduce"]
        assert "main" in t.functions
        assert t.macros["WARP_SIZE"] == "32"
        assert "<cuda_runtime.h>" in t.includes

    def test_a_name_belongs_to_exactly_one_kind(self):
        """A __global__ definition is a kernel and never also a plain function."""
        t = extract_symbols(CUDA)
        assert "reduce_kernel" not in t.functions
        assert "warpReduce" not in t.functions

    def test_symbols_in_comments_are_not_symbols(self):
        code = "// __global__ void ghost_kernel(int* a) {\n__global__ void real(int* a) {}\n"
        assert extract_symbols(code).kernels == ["real"]

    def test_symbols_in_string_literals_are_not_symbols(self):
        code = 'const char* s = "__global__ void fake(int* a) {";\n__global__ void real(int* a) {}\n'
        assert extract_symbols(code).kernels == ["real"]

    def test_control_flow_keywords_are_not_functions(self):
        code = "void f(){ if (x) { } for (;;) { } while (1) { } }\n"
        funcs = extract_symbols(code).functions
        assert "f" in funcs
        assert not ({"if", "for", "while"} & set(funcs))

    def test_prototypes_define_no_symbols(self):
        assert extract_symbols("__global__ void k(float* a);\n").kernels == []

    def test_quoted_includes_survive_intact(self):
        """Regression: an include's payload lives inside a string literal.

        Reading directives from literal-stripped text yielded `"        "`.
        """
        t = extract_symbols('#include <hip/hip_runtime.h>\n#include "shfl_integral.cuh"\n')
        assert t.includes == ['"shfl_integral.cuh"', '<hip/hip_runtime.h>']

    def test_macro_bodies_keep_their_string_literals(self):
        assert extract_symbols('#define MSG "hi"\n').macros == {"MSG": '"hi"'}

    def test_macro_body_drops_a_trailing_comment(self):
        assert extract_symbols("#define TILE 32  // wavefront\n").macros == {"TILE": "32"}

    def test_a_commented_out_include_is_not_an_include(self):
        assert extract_symbols('// #include "ghost.cuh"\n').includes == []

    def test_empty_source_yields_an_empty_table(self):
        t = extract_symbols("")
        assert t == SymbolTable()
        assert t.all_names() == set()

    def test_output_is_sorted_and_deduplicated(self):
        code = "__global__ void b(){}\n__global__ void a(){}\n__global__ void a(){}\n"
        assert extract_symbols(code).kernels == ["a", "b"]


class TestDiffSymbols:
    def test_identical_sources_produce_an_empty_diff(self):
        d = diff_symbols(CUDA, CUDA)
        assert d["removed_symbols"] == [] and d["added_symbols"] == []
        assert d["summary"]["kernels_dropped"] == 0

    def test_dropped_kernel_is_reported_as_missing(self):
        stripped = "\n".join(l for l in HIP.splitlines()
                             if "reduce_kernel" not in l and "out[blockIdx" not in l)
        d = diff_symbols(CUDA, stripped)
        assert "reduce_kernel" in d["kernels"]["missing"]
        assert d["summary"]["kernels_dropped"] == 1

    def test_dropped_helper_is_reported(self):
        no_helper = HIP.replace(
            "__device__ float warpReduce(float v) {", "__device__ float other(float v) {")
        d = diff_symbols(CUDA, no_helper)
        assert "warpReduce" in d["helpers"]["missing"]

    def test_preserved_symbols_are_listed(self):
        d = diff_symbols(CUDA, HIP)
        assert "reduce_kernel" in d["kernels"]["preserved"]
        assert "main" in d["functions"]["preserved"]

    def test_macro_change_is_reported(self):
        d = diff_symbols(CUDA, HIP)
        assert "WARP_SIZE" in d["macros"]["removed"]
        assert "WAVEFRONT_SIZE" in d["macros"]["added"]

    def test_changed_macro_value_is_reported(self):
        a = "#define TILE 32\n"
        b = "#define TILE 64\n"
        changed = diff_symbols(a, b)["macros"]["changed"]
        assert changed == [{"name": "TILE", "original": "32", "generated": "64"}]

    def test_include_swap_is_reported(self):
        d = diff_symbols(CUDA, HIP)
        assert "<cuda_runtime.h>" in d["includes"]["removed"]
        assert "<hip/hip_runtime.h>" in d["includes"]["added"]

    def test_rename_candidate_is_a_hypothesis_not_a_substitution(self):
        renamed = CUDA.replace("warpReduce", "wavefrontReduce")
        d = diff_symbols(CUDA, renamed)
        # The rename is proposed…
        assert any(c["from"] == "warpReduce" and c["to"] == "wavefrontReduce"
                   for c in d["rename_candidates"])
        # …AND the raw diff still shows both sides, so a reader who ignores the
        # hypothesis is never misled about what actually changed.
        assert "warpReduce" in d["removed_symbols"]
        assert "wavefrontReduce" in d["added_symbols"]

    def test_unrelated_names_are_not_proposed_as_renames(self):
        a = "__device__ int alpha(){ return 0; }\n"
        b = "__device__ int zzzzzz(){ return 0; }\n"
        assert diff_symbols(a, b)["rename_candidates"] == []

    def test_each_symbol_is_consumed_by_at_most_one_rename(self):
        a = "__device__ int warpScan(){return 0;}\n__device__ int warpSum(){return 0;}\n"
        b = "__device__ int waveScan(){return 0;}\n__device__ int waveSum(){return 0;}\n"
        pairs = diff_symbols(a, b)["rename_candidates"]
        assert len({p["from"] for p in pairs}) == len(pairs)
        assert len({p["to"] for p in pairs}) == len(pairs)

    def test_diff_is_byte_identical_across_runs(self):
        """Determinism is what makes one debug session diffable against another."""
        a = json.dumps(diff_symbols(CUDA, HIP), sort_keys=True)
        b = json.dumps(diff_symbols(CUDA, HIP), sort_keys=True)
        assert a == b

    def test_empty_inputs_do_not_raise(self):
        d = diff_symbols("", "")
        assert d["summary"]["kernels_dropped"] == 0
