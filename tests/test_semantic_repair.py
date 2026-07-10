"""Unit tests for the Semantic Translation Repair Engine.

The engine converts hipcc/clang diagnostics into minimal, deterministic source
patches recovered from the ORIGINAL CUDA source — never regenerating the file,
never inventing code. These tests pin the behaviours the mission requires:
diagnostic parsing, symbol resolution from CUDA, minimal additive patches,
cause classification, confidence tiers, scope detection, recompile-guarded
acceptance/rejection (no regression), and byte-for-byte determinism.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.semantic_repair import (
    SemanticRepairEngine, CudaSourceIndex, HipUnit, parse_diagnostics,
    Diagnostic, CONF_HIGH, CONF_MEDIUM, CONF_LOW,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

CUDA = """#include <cuda_runtime.h>
#define TILE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct Params { int n; float scale; };

typedef unsigned long long u64;

__device__ float warp_reduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void kern(const float* in, float* out, Params p) {
    int i = blockIdx.x * TILE + threadIdx.x;
    float v = (i < p.n) ? in[i] * p.scale : 0.0f;
    out[blockIdx.x] = warp_reduce(v);
}
"""

# A HIP port that dropped TILE, Params and warp_reduce during translation.
HIP_MISSING = """#include <hip/hip_runtime.h>
#define WAVEFRONT_SIZE 64

__global__ void kern(const float* in, float* out, Params p) {
    int i = blockIdx.x * TILE + threadIdx.x;
    float v = (i < p.n) ? in[i] * p.scale : 0.0f;
    out[blockIdx.x] = warp_reduce(v);
}
"""

DIAGS = [
    "kern.hip.cpp:4:41: error: unknown type name 'Params'",
    "kern.hip.cpp:5:29: error: use of undeclared identifier 'TILE'",
    "kern.hip.cpp:7:23: error: use of undeclared identifier 'warp_reduce'",
]


# ── Diagnostic parsing ──────────────────────────────────────────────────────

def test_parse_undeclared_identifier():
    d = parse_diagnostics(["f.cpp:5:1: error: use of undeclared identifier 'foo'"])
    assert len(d) == 1
    assert d[0].kind == "undeclared"
    assert d[0].symbol == "foo"
    assert d[0].line == 5


def test_parse_did_you_mean_suggestion():
    d = parse_diagnostics(
        ["f.cpp:5:1: error: use of undeclared identifier 'hipMalloc'; did you mean 'hipFree'?"])
    assert d[0].symbol == "hipMalloc"
    assert d[0].suggestion == "hipFree"


def test_parse_unknown_type():
    d = parse_diagnostics(["f.cpp:1:1: error: unknown type name 'Params'"])
    assert d[0].kind == "unknown-type"
    assert d[0].symbol == "Params"


def test_parse_missing_include():
    d = parse_diagnostics(["f.cpp:1:10: fatal error: 'cuda_runtime.h' file not found"])
    assert d[0].kind == "missing-include"
    assert d[0].symbol == "cuda_runtime.h"


def test_parse_no_member():
    d = parse_diagnostics(["f.cpp:9:5: error: no member named 'scale' in 'Params'"])
    assert d[0].kind == "no-member"
    assert d[0].symbol == "scale"
    assert d[0].owner == "Params"


def test_parse_skips_warnings_and_notes():
    d = parse_diagnostics([
        "f.cpp:1:1: warning: unused variable 'x'",
        "f.cpp:2:2: note: expanded from macro",
        "f.cpp:3:3: error: use of undeclared identifier 'y'",
    ])
    assert len(d) == 1
    assert d[0].symbol == "y"


def test_diagnostic_key_is_location_independent():
    a = parse_diagnostics(["f.cpp:5:1: error: use of undeclared identifier 'foo'"])[0]
    b = parse_diagnostics(["f.cpp:99:7: error: use of undeclared identifier 'foo'"])[0]
    assert a.key == b.key  # learning cache stability


# ── CUDA source index ───────────────────────────────────────────────────────

def test_index_finds_macro_type_helper():
    idx = CudaSourceIndex(CUDA)
    assert idx.lookup("TILE").kind == "macro"
    assert idx.lookup("CEIL_DIV").kind == "macro"
    assert idx.lookup("Params").kind == "type"
    assert idx.lookup("u64").kind == "type"
    assert idx.lookup("warp_reduce").kind == "helper"
    assert idx.lookup("kern").kind == "kernel"


def test_index_macro_text_is_verbatim():
    idx = CudaSourceIndex(CUDA)
    assert idx.lookup("TILE").text.strip() == "#define TILE 32"
    assert "CEIL_DIV(a, b)" in idx.lookup("CEIL_DIV").text


def test_index_helper_body_is_brace_matched():
    idx = CudaSourceIndex(CUDA)
    text = idx.lookup("warp_reduce").text
    assert text.startswith("__device__ float warp_reduce")
    assert text.rstrip().endswith("}")
    assert text.count("{") == text.count("}")


def test_index_unknown_symbol_returns_none():
    assert CudaSourceIndex(CUDA).lookup("does_not_exist") is None


# ── HIP unit analysis ───────────────────────────────────────────────────────

def test_hipunit_parameter_use_is_not_a_definition():
    # The bug this pins: `__global__ void kern(..., Params p)` USES Params; it
    # must not be read as defining it.
    unit = HipUnit(HIP_MISSING)
    assert not unit.defines("Params")
    assert not unit.defines("warp_reduce")


def test_hipunit_recognizes_real_definitions():
    unit = HipUnit(CUDA)
    assert unit.defines("warp_reduce")
    assert unit.defines("Params")
    assert unit.defines("TILE")


# ── Resolution + restoration (additive pass, no compiler) ───────────────────

def test_restores_all_dropped_symbols():
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS)
    assert res.changed
    restored = {p.symbol for p in res.accepted_patches}
    assert restored == {"Params", "TILE", "warp_reduce"}
    # Every restoration lands before the kernel that uses it.
    head = res.patched_code.split("__global__ void kern")[0]
    assert "#define TILE 32" in head
    assert "struct Params" in head
    assert "warp_reduce" in head


def test_restoration_is_verbatim_from_cuda():
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS)
    assert "__shfl_down_sync(0xffffffff, v, 16)" not in res.patched_code  # sanity
    assert "for (int o = 16; o > 0; o >>= 1)" in res.patched_code  # exact CUDA body


def test_cause_classification():
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS)
    causes = {p.symbol: p.root_cause for p in res.accepted_patches}
    # warp_reduce/Params are in the symbol-diff removed set → removed-during-extraction;
    # TILE is a macro (not tracked as a symbol) → omitted. Both are valid "lost".
    assert causes["warp_reduce"] in ("removed-during-extraction", "omitted")
    assert causes["TILE"] == "omitted"


def test_confidence_tiers():
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS)
    for p in res.accepted_patches:
        assert p.confidence == CONF_HIGH  # restorations are high-confidence
        assert p.confidence_score >= 0.9


def test_no_change_when_nothing_missing():
    # HIP already contains everything → no diagnostics resolve → no change.
    res = SemanticRepairEngine(CUDA, CUDA).repair(DIAGS)
    assert not res.changed
    assert res.patched_code == CUDA


def test_does_not_duplicate_existing_definition():
    # If the port already defines TILE, restoring it must be a no-op (no
    # redefinition, which hipcc treats as a hard error).
    hip = HIP_MISSING.replace("#define WAVEFRONT_SIZE 64",
                              "#define WAVEFRONT_SIZE 64\n#define TILE 32")
    res = SemanticRepairEngine(CUDA, hip).repair(
        ["f.cpp:1:1: error: use of undeclared identifier 'TILE'"])
    assert res.patched_code.count("#define TILE 32") == 1


# ── Includes ────────────────────────────────────────────────────────────────

def test_missing_cuda_header_maps_to_hip():
    hip = "__global__ void k(float* a) { int i = threadIdx.x; a[i] = 0; }\n"
    res = SemanticRepairEngine(CUDA, hip).repair(
        ["k.cpp:1:10: fatal error: 'cuda_runtime.h' file not found"])
    assert res.changed
    assert "#include <hip/hip_runtime.h>" in res.patched_code


# ── Determinism ─────────────────────────────────────────────────────────────

def test_repair_is_deterministic():
    a = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS).patched_code
    b = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS).patched_code
    assert a == b


def test_diagnostic_order_does_not_affect_output():
    a = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS).patched_code
    b = SemanticRepairEngine(CUDA, HIP_MISSING).repair(list(reversed(DIAGS))).patched_code
    assert a == b


# ── Recompile-guarded loop (Phase 6) ────────────────────────────────────────

def _fake_recompiler(cuda_symbols_needed):
    """A recompiler whose errors vanish as each named symbol becomes defined."""
    from verification.semantic_repair import (
        _defines_function, _defines_macro, _defines_type)

    def recompile(src):
        errs = []
        for name, kind in cuda_symbols_needed:
            present = {"macro": _defines_macro, "func": _defines_function,
                       "type": _defines_type}[kind](src, name)
            if not present:
                errs.append(f"t.cpp:1:1: error: use of undeclared identifier '{name}'")
        return (len(errs) == 0, errs)

    return recompile


def test_recompile_guided_reaches_clean_compile():
    recompile = _fake_recompiler([("TILE", "macro"), ("warp_reduce", "func")])
    _, initial = recompile(HIP_MISSING)
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(initial, recompile=recompile)
    assert res.errors_after == 0
    assert res.changed
    assert all(p.accepted for p in res.patches)


def test_no_regression_patch_rejected():
    # A recompiler that always reports MORE errors after any edit → every patch
    # must be rejected and the code left unchanged.
    def worse(_src):
        return (False, ["a:1:1: error: x", "b:2:2: error: y",
                        "c:3:3: error: z", "d:4:4: error: w"])
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(DIAGS, recompile=worse)
    assert not res.changed
    assert all(not p.accepted for p in res.patches)


# ── Scope repair (Phase 4) ──────────────────────────────────────────────────

def test_detects_device_builtin_at_host_scope():
    hip = ("#include <hip/hip_runtime.h>\n"
           "int host_fn() { int i = threadIdx.x; return i; }\n")
    res = SemanticRepairEngine("", hip).repair([])
    scope = [f for f in res.unresolved if f["kind"] == "wrong-scope"]
    assert any(f["symbol"] == "threadIdx" for f in scope)
    assert scope[0]["confidence"] == CONF_LOW  # never auto-applied


def test_device_builtin_inside_kernel_is_fine():
    hip = ("#include <hip/hip_runtime.h>\n"
           "__global__ void k(float* a) { a[threadIdx.x] = blockIdx.x; }\n")
    res = SemanticRepairEngine("", hip).repair([])
    assert not any(f["kind"] == "wrong-scope" for f in res.unresolved)


# ── Learning cache (Phase 9) ────────────────────────────────────────────────

def test_learning_cache_records_successful_strategy():
    cache = {}
    SemanticRepairEngine(CUDA, HIP_MISSING, cache=cache).repair(DIAGS)
    assert cache.get("undeclared:TILE") == "restore-macro"
    assert cache.get("unknown-type:Params") == "restore-type"


def test_cache_shared_across_engines():
    cache = {}
    SemanticRepairEngine(CUDA, HIP_MISSING, cache=cache).repair(DIAGS)
    # A second engine (different kernel) receives the same cache instance.
    eng2 = SemanticRepairEngine(CUDA, HIP_MISSING, cache=cache)
    assert eng2.cache is cache
    assert "undeclared:TILE" in eng2.cache


# ── Unresolved reporting (never guesses) ────────────────────────────────────

def test_unresolvable_symbol_left_for_llm():
    # A symbol that exists in neither the HIP unit nor the CUDA source cannot be
    # deterministically recovered — it must be reported, not fabricated.
    res = SemanticRepairEngine(CUDA, HIP_MISSING).repair(
        ["f.cpp:1:1: error: use of undeclared identifier 'totally_unknown_xyz'"])
    assert not res.changed
    assert any(u["symbol"] == "totally_unknown_xyz" for u in res.unresolved)
