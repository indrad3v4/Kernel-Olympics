"""Unit tests for the pre-compile structural validator."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.structural import validate_structure, extract_top_level_functions


SRC = '''
#include <cuda_runtime.h>
__device__ float helper(float x) { return x * 2.0f; }
__global__ void warp_reduce_kernel(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    float v = in[tid];
    v += __shfl_down_sync(0xffffffff, v, 16);
    if (tid == 0) out[0] = helper(v);
}
int main() {
    float *d; hipMalloc(&d, 4);
    printf("done\\n");
    return 0;
}
'''

GOOD = (SRC.replace("cuda_runtime.h", "hip/hip_runtime.h")
           .replace("__shfl_down_sync(0xffffffff, v, 16)", "__shfl_down(v, 16)"))


def test_valid_port_is_accepted():
    r = validate_structure(SRC, GOOD)
    assert r.ok
    assert r.score == 1.0
    assert not r.errors


def test_identity_port_is_accepted():
    assert validate_structure(SRC, SRC).ok


def test_unbalanced_brace_is_rejected():
    r = validate_structure(SRC, GOOD[: GOOD.rindex("return 0;")])
    assert not r.ok
    assert any("unbalanced braces" in e for e in r.errors)


def test_truncation_marker_is_rejected():
    r = validate_structure(SRC, GOOD.replace("int main()", "// ... rest of code\nint main()"))
    assert not r.ok
    assert any("truncation marker" in e for e in r.errors)


def test_duplicate_definition_is_rejected():
    r = validate_structure(SRC, GOOD + "\n" + GOOD)
    assert not r.ok
    assert any("duplicate definitions" in e for e in r.errors)


def test_severely_truncated_output_is_rejected():
    r = validate_structure(SRC, "int main(){return 0;}")
    assert not r.ok
    assert any("truncated" in e for e in r.errors)


def test_empty_generation_is_rejected():
    r = validate_structure(SRC, "")
    assert not r.ok
    assert "empty generation" in r.errors


def test_dropped_main_warns_but_does_not_reject():
    r = validate_structure(SRC, GOOD[: GOOD.index("int main()")])
    assert r.ok, "dropped main() must not hard-reject"
    assert "main" in r.missing_symbols
    assert any("symbols dropped" in w for w in r.warnings)


def test_dropped_helper_is_reported():
    no_helper = GOOD.replace("__device__ float helper(float x) { return x * 2.0f; }", "")
    assert "helper" in validate_structure(SRC, no_helper).missing_symbols


def test_brace_inside_string_or_char_literal_is_not_counted():
    trap = GOOD.replace('printf("done\\n");', "char c = '{'; printf(\"}\\n\");")
    assert validate_structure(SRC, trap).ok


def test_brace_inside_comment_is_not_counted():
    trap = GOOD.replace('printf("done\\n");', '// closing } brace in comment\n    printf("ok\\n");')
    assert validate_structure(SRC, trap).ok


def test_comparison_and_shift_operators_are_not_templates():
    trap = GOOD.replace("int tid = threadIdx.x;",
                        "int tid = threadIdx.x; int z = 1 << 3; bool b = tid < n && n > 2;")
    assert validate_structure(SRC, trap).ok


def test_extracts_top_level_definitions():
    assert set(extract_top_level_functions(SRC)) == {"helper", "warp_reduce_kernel", "main"}


def test_control_flow_is_not_mistaken_for_a_function():
    code = "int main() { if (x) { y(); } for (int i=0;i<3;i++) { z(); } return 0; }"
    names = extract_top_level_functions(code)
    assert "if" not in names and "for" not in names


def test_prototype_is_not_a_definition():
    assert "foo" not in extract_top_level_functions("void foo(int a);")
