"""Tests for pre-compilation static analysis.

Every rule here is advisory. The tests therefore check two things in equal
measure: that a rule fires on the defect it names, and that it does NOT fire on
valid code that merely resembles it. A false positive sends a repair prompt
chasing a phantom, which is worse than a missed finding.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.static_analysis import analyze, Finding, StaticAnalysisReport


HIP = """#include <hip/hip_runtime.h>
#define WAVEFRONT_SIZE 64
__device__ float waveReduce(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffffffffffULL, v, o);
    return v;
}
__global__ void reduce_kernel(float* in, float* out, int n) {
    out[blockIdx.x] = waveReduce(in[threadIdx.x]);
}
int main() { return 0; }
"""


def _rules(code):
    return {f.rule for f in analyze(code).findings}


class TestInvalidHipConstructs:
    def test_detects_residual_cuda_symbol(self):
        assert "residual-cuda-symbol" in _rules("void f(){ cudaMalloc(&p, 4); }")

    def test_detects_cuda_header(self):
        assert "cuda-header" in _rules('#include <cuda_runtime.h>\n')

    def test_detects_cuh_header(self):
        """Regression: a quoted include's payload is a string literal, and the
        literal-stripping pass blanked it before this rule could see it."""
        assert "cuda-header" in _rules('#include "shfl_integral_image.cuh"\n')

    def test_a_commented_out_cuda_header_is_not_flagged(self):
        assert "cuda-header" not in _rules('// #include "ghost.cuh"\n')

    def test_cuda_header_finding_reports_the_right_line(self):
        code = '#include <hip/hip_runtime.h>\n#include "x.cuh"\n'
        f = next(x for x in analyze(code).findings if x.rule == "cuda-header")
        assert f.line == 2

    def test_detects_syncwarp(self):
        assert "invalid-hip-construct" in _rules("__device__ void f(){ __syncwarp(); }")

    def test_detects_activemask(self):
        assert "invalid-hip-construct" in _rules("__device__ int f(){ return __activemask(); }")

    def test_detects_global_returning_non_void(self):
        assert "global-non-void" in _rules("__global__ int k(float* a){ return 1; }")

    def test_global_returning_void_is_clean(self):
        assert "global-non-void" not in _rules("__global__ void k(float* a){ }")

    def test_hip_symbols_are_not_flagged(self):
        assert "residual-cuda-symbol" not in _rules("void f(){ hipMalloc(&p, 4); }")


class TestDuplicateDefinitions:
    def test_detects_duplicate_definition(self):
        code = "__global__ void k(float* a){ }\n__global__ void k(float* a){ }\n"
        f = [x for x in analyze(code).findings if x.rule == "duplicate-definition"]
        assert len(f) == 1 and f[0].severity == "error"

    def test_duplicate_is_anchored_to_the_second_definition(self):
        code = "__global__ void k(float* a){ }\n__global__ void k(float* a){ }\n"
        f = next(x for x in analyze(code).findings if x.rule == "duplicate-definition")
        assert f.line == 2

    def test_two_different_functions_are_not_duplicates(self):
        code = "__global__ void a(float* x){ }\n__global__ void b(float* x){ }\n"
        assert "duplicate-definition" not in _rules(code)


class TestWavefrontHazards:
    def test_detects_32_bit_shuffle_mask(self):
        assert "warp-mask-32" in _rules(
            "__device__ int f(int v){ return __shfl_down_sync(0xffffffff, v, 1); }")

    def test_64_bit_mask_is_clean(self):
        assert "warp-mask-32" not in _rules(
            "__device__ int f(int v){ return __shfl_down_sync(0xffffffffffffffffULL, v, 1); }")

    def test_detects_warp_size_macro(self):
        assert "warp-size-32" in _rules("#define WARP_SIZE 32\n")

    def test_detects_warp_size_constant(self):
        assert "warp-size-32" in _rules("const int warp_size = 32;\n")

    def test_hazards_are_warnings_not_errors(self):
        """These compile cleanly. Calling them errors would be a lie."""
        r = analyze("#define WARP_SIZE 32\n__device__ int f(int v){ "
                    "return __shfl_down_sync(0xffffffff, v, 1); }")
        assert all(f.severity == "warning" for f in r.findings
                   if f.rule in {"warp-size-32", "warp-mask-32"})


class TestUnreachableCode:
    def test_detects_unreachable_code(self):
        code = "__global__ void k(float* a){\n    return;\n    a[0] = 1;\n}\n"
        assert "unreachable-code" in _rules(code)

    def test_guarded_return_is_not_unreachable(self):
        code = "__global__ void k(float* a){\n    if (a[0] > 0) return;\n    a[1] = 2;\n}\n"
        assert "unreachable-code" not in _rules(code)

    def test_return_before_closing_brace_is_not_unreachable(self):
        code = "__global__ void k(float* a){\n    a[0] = 1;\n    return;\n}\n"
        assert "unreachable-code" not in _rules(code)

    def test_a_label_after_a_jump_is_reachable(self):
        code = "void f(int x){\n switch(x){\n case 1:\n  break;\n case 2:\n  return;\n }\n}\n"
        assert "unreachable-code" not in _rules(code)


class TestMalformedDeclarations:
    def test_detects_a_truncated_declaration(self):
        code = "__global__ void k(float* a)\n"
        assert "malformed-declaration" in _rules(code)

    def test_brace_on_the_next_line_is_idiomatic_not_malformed(self):
        code = "__global__ void k(float* a)\n{\n}\n"
        assert "malformed-declaration" not in _rules(code)

    def test_a_prototype_is_not_malformed(self):
        assert "malformed-declaration" not in _rules("__global__ void k(float* a);\n")

    def test_a_wrapped_parameter_list_is_not_malformed(self):
        code = "__global__ void k(float* a,\n                  float* b) {\n}\n"
        assert "malformed-declaration" not in _rules(code)


class TestReportShape:
    def test_comments_and_strings_never_trigger_rules(self):
        code = ('// TODO: replace cudaMalloc with hipMalloc\n'
                'void f(){ printf("cudaMalloc failed"); }\n')
        assert "residual-cuda-symbol" not in _rules(code)

    def test_clean_hip_source_yields_no_errors(self):
        assert analyze(HIP).counts()["error"] == 0

    def test_empty_source_is_handled(self):
        r = analyze("")
        assert r.findings == [] and r.lines_analyzed == 0

    def test_whitespace_only_source_is_handled(self):
        assert analyze("   \n\n  ").findings == []

    def test_findings_are_sorted_by_line(self):
        code = ("__global__ int bad(float* a){ return 1; }\n"
                "#include <cuda_runtime.h>\n"
                "__device__ void h(){ __syncwarp(); }\n")
        lines = [f.line for f in analyze(code).findings]
        assert lines == sorted(lines)

    def test_counts_agree_with_findings(self):
        r = analyze("#include <cuda_runtime.h>\n#define WARP_SIZE 32\n")
        c = r.counts()
        assert c["total"] == len(r.findings) == c["error"] + c["warning"] + c["info"]

    def test_report_is_json_serializable(self):
        json.dumps(analyze(HIP).to_dict())

    def test_analysis_is_byte_identical_across_runs(self):
        """Determinism is what makes one debug session diffable against another."""
        code = "#include <cuda_runtime.h>\n__global__ int k(){ return 1; }\n"
        a = json.dumps(analyze(code).to_dict(), sort_keys=True)
        b = json.dumps(analyze(code).to_dict(), sort_keys=True)
        assert a == b

    def test_finding_carries_evidence_from_the_source_line(self):
        f = next(x for x in analyze('#include <cuda_runtime.h>\n').findings
                 if x.rule == "cuda-header")
        assert "cuda_runtime.h" in f.evidence

    def test_a_raising_rule_degrades_to_an_info_finding(self, monkeypatch):
        """A bug in one rule must never take down a debug dump."""
        import verification.static_analysis as sa

        def boom(*a, **kw):
            raise RuntimeError("rule exploded")

        monkeypatch.setattr(sa, "_find_duplicate_definitions", boom)
        r = sa.analyze(HIP)
        assert any(f.rule == "analyzer-error" and f.severity == "info"
                   for f in r.findings)
