"""Structural gate: hipcc must not run on generations that are broken at the text level.

Before this gate, a 60s hipcc call was spent every iteration to rediscover defects
a text-level check catches in <1ms — an unbalanced brace, a truncation marker, a
gutted body. The compiler then reports parser errors ("expected member name",
"expected unqualified-id") whose root cause is that shape, so every refinement
faithfully reproduces the same errors and the budget drains.

These tests pin the contract for that gate at three layers:

  1. ``_postprocess_port`` computes a ``ValidationResult`` alongside the code,
     for every entry point (initial port, refine, refine-retry).
  2. The route() loop treats a not-ok result as a compile failure with synthetic
     error strings — the hipcc call is skipped and the report goes into the
     next refine's feedback.
  3. ``_build_kimi_refine_prompt`` names the specific defects and dropped
     symbols so the coder acts on the shape of its own output, not on a
     downstream parser error.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from router import ModelRouter
from verification.structural import ValidationResult, validate_structure


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


ORIGINAL_CUDA = """\
#include <cuda_runtime.h>
__global__ void k(float* x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) x[tid] = x[tid] * 2.0f;
}

void helper_a(float* x) { (void)x; }
void helper_b(float* x) { (void)x; }
"""


# ── _postprocess_port returns a ValidationResult ────────────────────────────

class TestPostprocessPortStructuralReturn:
    def test_returns_four_tuple_with_structural_result(self, router):
        raw = "```cpp\n" + ORIGINAL_CUDA.replace("cuda_runtime.h",
                                                  "hip/hip_runtime.h") + "```"
        _code, _changelog, _restored, structural = router._postprocess_port(
            raw, ORIGINAL_CUDA)
        assert isinstance(structural, ValidationResult)

    def test_valid_port_gets_ok_true(self, router):
        valid = ORIGINAL_CUDA.replace("cuda_runtime.h", "hip/hip_runtime.h")
        raw = f"```hip\n{valid}\n```"
        _, _, _, structural = router._postprocess_port(raw, ORIGINAL_CUDA)
        assert structural.ok is True


# ── Broken generations that MUST be rejected before hipcc ──────────────────

class TestStructuralRejects:
    def test_unbalanced_braces_reject(self, router):
        """A body that ends before its closing brace is the textbook 'parser
        will explode with cascading errors' shape. hipcc will produce a dozen
        useless diagnostics; this catches it in <1ms."""
        broken = "```cpp\n#include <hip/hip_runtime.h>\n" \
                 "__global__ void k(float* x) {\n" \
                 "    x[threadIdx.x] = 1.0f;\n" \
                 "```"  # missing closing '}'
        _, _, _, structural = router._postprocess_port(broken, ORIGINAL_CUDA)
        assert structural.ok is False
        assert any("unbalanced braces" in e for e in structural.errors)

    def test_truncation_marker_reject(self, router):
        """A '// ... rest of code' marker is text the coder gave up mid-file.
        hipcc reads it as a syntax error at a comment, which is nonsense."""
        broken = ("```hip\n"
                  "#include <hip/hip_runtime.h>\n"
                  "__global__ void k(float* x) {\n"
                  "    x[0] = 1.0f;\n"
                  "}\n"
                  "// ... rest of code omitted\n"
                  "```")
        _, _, _, structural = router._postprocess_port(broken, ORIGINAL_CUDA)
        assert structural.ok is False
        assert any("truncation marker" in e for e in structural.errors)

    def test_severely_truncated_generation_reject(self, router):
        """A generation an order of magnitude shorter than the source is
        almost certainly cut off mid-response — reject before hipcc."""
        long_src = ORIGINAL_CUDA + "\n".join(
            f"void filler_{i}() {{ /* pad */ }}" for i in range(50))
        broken = "```hip\n#include <hip/hip_runtime.h>\n```"
        _, _, _, structural = router._postprocess_port(broken, long_src)
        assert structural.ok is False


# ── DEVICE_SUBSET ports are sized against the device region, not the file ──

# A full NVIDIA-sample-shaped source: a large host driver + a couple of kernels.
# A DEVICE_SUBSET port correctly emits ONLY the kernels, so it is a small
# fraction of the whole file — which must NOT be read as truncation.
_HOST_HEAVY_SOURCE = (
    "#include <cuda_runtime.h>\n"
    "#include <helper_cuda.h>\n"
    + "\n".join(f"void host_helper_{i}(int* p) {{ (void)p; /* pad */ }}"
                for i in range(40))
    + "\n"
    "__global__ void scan_kernel(int* data, int width, int* sums) {\n"
    "    extern __shared__ int s[];\n"
    "    int id = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    s[threadIdx.x] = data[id];\n"
    "    __syncthreads();\n"
    "    data[id] = s[threadIdx.x];\n"
    "}\n"
    "int main() { return 0; }\n"
)

_DEVICE_ONLY_PORT = (
    "__global__ void scan_kernel(int* data, int width, int* sums) {\n"
    "    extern __shared__ int s[];\n"
    "    int id = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    s[threadIdx.x] = data[id];\n"
    "    __syncthreads();\n"
    "    data[id] = s[threadIdx.x];\n"
    "}\n"
)


class TestDeviceSubsetSizing:
    def test_device_subset_port_not_rejected_as_truncated(self):
        """A correct device-only port is a small fraction of a host-heavy file;
        under DEVICE_SUBSET it must be sized against the source's device region,
        not the whole file, or it is falsely rejected as truncated and never
        reaches hipcc (the nvidia_shfl_scan iteration-exhaustion regression)."""
        r = validate_structure(_HOST_HEAVY_SOURCE, _DEVICE_ONLY_PORT,
                               port_mode="DEVICE_SUBSET")
        assert r.ok is True, r.reason()

    def test_same_port_rejected_without_device_subset(self):
        """Without the port_mode hint the whole-file baseline still applies —
        this documents the exact behavior the fix changes."""
        r = validate_structure(_HOST_HEAVY_SOURCE, _DEVICE_ONLY_PORT)
        assert r.ok is False

    def test_device_subset_still_rejects_real_truncation(self):
        """The truncation signal is preserved: a near-empty port is still below
        25% of the device-region baseline."""
        tiny = "#include <hip/hip_runtime.h>\n"
        r = validate_structure(_HOST_HEAVY_SOURCE, tiny, port_mode="DEVICE_SUBSET")
        assert r.ok is False

    def test_device_subset_does_not_flag_dropped_host_helpers(self):
        """Dropped host helpers are correct in DEVICE_SUBSET mode and must not
        appear as missing symbols (which would push the coder to re-add them)."""
        r = validate_structure(_HOST_HEAVY_SOURCE, _DEVICE_ONLY_PORT,
                               port_mode="DEVICE_SUBSET")
        assert not any("host_helper" in s for s in r.missing_symbols)
        assert "main" not in r.missing_symbols


# ── Symbol preservation surfaces as a warning, not a hard reject ───────────

class TestStructuralSymbolPreservation:
    def test_dropped_helper_appears_in_missing_symbols(self, router):
        """A missing helper is strong evidence of a bad port but a regex
        extractor is not trustworthy enough to gate a compile on. It rides
        into the refine prompt as targeted feedback instead."""
        port_without_helper = ("#include <hip/hip_runtime.h>\n"
                               "__global__ void k(float* x, int n) {\n"
                               "    int tid = threadIdx.x;\n"
                               "    if (tid < n) x[tid] = x[tid] * 2.0f;\n"
                               "}\n"
                               "void helper_a(float* x) { (void)x; }\n"
                               "// helper_b dropped\n")
        _, _, _, structural = router._postprocess_port(
            port_without_helper, ORIGINAL_CUDA)
        assert "helper_b" in structural.missing_symbols


# ── Refine prompt carries the report so the coder can act on it ────────────

class TestRefinePromptCarriesStructuralReport:
    def test_prompt_lists_dropped_symbols(self, router):
        report = {
            "ok": False,
            "reason": "REJECTED: unbalanced braces (depth +1)",
            "missing_symbols": ["helper_b", "helper_a"],
            "warnings": [],
            "errors": ["unbalanced braces (depth +1) — truncated before the final '}'"],
        }
        prompt = router._build_kimi_refine_prompt(
            kernel_source=ORIGINAL_CUDA,
            previous_code="// prev",
            feedback="prior feedback",
            patterns=[],
            structural_report=report,
        )
        assert "STRUCTURAL ERRORS" in prompt
        assert "unbalanced braces" in prompt
        assert "SYMBOLS DROPPED" in prompt
        assert "helper_b" in prompt and "helper_a" in prompt

    def test_prompt_omits_structural_section_when_report_is_none(self, router):
        prompt = router._build_kimi_refine_prompt(
            kernel_source=ORIGINAL_CUDA,
            previous_code="// prev",
            feedback="prior feedback",
            patterns=[],
            structural_report=None,
        )
        assert "STRUCTURAL ERRORS" not in prompt
        assert "SYMBOLS DROPPED" not in prompt

    def test_prompt_omits_structural_section_when_report_is_ok(self, router):
        """An ok report has no errors and no missing symbols to name — do not
        emit an empty header section that would prime the coder to hunt for
        problems that are not there."""
        report = {
            "ok": True, "reason": "structurally valid",
            "missing_symbols": [], "warnings": [], "errors": [],
        }
        prompt = router._build_kimi_refine_prompt(
            kernel_source=ORIGINAL_CUDA,
            previous_code="// prev",
            feedback="prior feedback",
            patterns=[],
            structural_report=report,
        )
        assert "STRUCTURAL ERRORS" not in prompt
        assert "SYMBOLS DROPPED" not in prompt


# ── The gate never takes itself down ───────────────────────────────────────

class TestGateFailsSafe:
    def test_structural_check_error_does_not_break_postprocess(self, router, monkeypatch):
        """A bug in validate_structure must never propagate — a broken gate
        is worse than a permissive one because it blocks every port."""
        import router as router_mod

        def _raise(*_a, **_kw):
            raise RuntimeError("simulated bug in structural check")

        monkeypatch.setattr(router_mod, "_validate_structure", _raise)
        # Should NOT raise; should return an ok=True fallback so the loop
        # continues to hipcc as it did before this file existed.
        _, _, _, structural = router._postprocess_port(
            "```hip\n#include <hip/hip_runtime.h>\nint main(){}\n```",
            ORIGINAL_CUDA)
        assert structural.ok is True
