"""Tests for the PortMode contract (WHOLE_PROGRAM vs DEVICE_SUBSET).

Root cause under test: two consumers used to answer two different questions
off the same ``self_contained`` flag. The coder prompt asked "does this source
have a main()?" (yes -> reproduce it); the verifier's harness generator asked
"should I expect the port to supply its own main()?" (also yes, because
self_contained said so). Neither asked "CAN the port supply a working main()
at all?" For a full NVIDIA sample whose main() depends on a local header this
repo never vendored (``nvidia_shfl_scan.cu`` -> ``shfl_integral_image.cuh``),
the answer is no — and the pipeline ended up with a translation unit that had
neither the original driver (the coder was told to drop the call it can't
satisfy) nor a synthesized harness (the verifier assumed one wasn't needed).

``determine_port_mode()`` is the single decision every consumer below now
shares. These tests pin: the decision itself, that router and verifier read
it identically, that a WHOLE_PROGRAM kernel's behavior is completely
unchanged, and that DEVICE_SUBSET actually converges (compiles) rather than
merely existing as an architectural label.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from verification.spec_parser import (
    determine_port_mode, is_self_contained, unresolved_local_headers,
    generate_spec_from_source, PORT_MODE_WHOLE_PROGRAM, PORT_MODE_DEVICE_SUBSET,
)
from verification.verifier import VerificationAgent
import router as router_mod
from router import ModelRouter


# ── Fixtures ─────────────────────────────────────────────────────────────────

NVIDIA_SHFL_SCAN_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'sample_kernels', 'cuda', 'nvidia_shfl_scan.cu')

with open(NVIDIA_SHFL_SCAN_PATH, encoding='utf-8') as _f:
    NVIDIA_SHFL_SCAN_CUDA = _f.read()

# A self-contained program whose dependencies ALL resolve — must stay
# WHOLE_PROGRAM. Distinguishes "self-contained" from "self-contained AND
# unresolvable" as the actual trigger condition.
SELF_CONTAINED_RESOLVABLE = """
#include <cuda_runtime.h>
__global__ void add_one(int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}
int main() {
    return 0;
}
"""

# A bare kernel snippet — no main() at all. Must stay WHOLE_PROGRAM (there is
# no "whole program" vs "device subset" distinction without a driver to split).
BARE_KERNEL = """
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""

# A self-contained program with an unresolved LOCAL header but where main()
# does NOT call anything from it — still DEVICE_SUBSET, since the decision is
# about the header being unresolvable, not about what main() happens to call
# (see determine_port_mode's docstring for why this is deliberately the
# simpler, more conservative rule).
SELF_CONTAINED_UNRESOLVABLE_HEADER = """
#include <cuda_runtime.h>
#include "totally_unvendored_helper.cuh"
__global__ void k(int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}
int main() {
    return 0;
}
"""

# A correct, wavefront64-aware DEVICE_SUBSET port of nvidia_shfl_scan.cu's two
# device kernels — the reference fixture referenced in the mission doc's A.5.
# __shfl_up_sync's mask is 64-bit (wavefront64, not CUDA's 32-lane warp); width
# is left parametrized (the harness supplies 64, a full-wavefront scan).
NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT = """#include <hip/hip_runtime.h>
#define WAVEFRONT_SIZE 64

__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    int warp_id = threadIdx.x / warpSize;

    int value = data[id];

#pragma unroll
    for (int i = 1; i <= width; i *= 2) {
        unsigned long long mask = 0xffffffffffffffffULL;
        int n = __shfl_up_sync(mask, value, i, width);
        if (lane_id >= i)
            value += n;
    }

    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        int warp_sum = sums[lane_id];
        unsigned long long mask2 = (1ULL << (blockDim.x / warpSize)) - 1;
        for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
            int n = __shfl_up_sync(mask2, warp_sum, i, (blockDim.x / warpSize));
            if (lane_id >= i)
                warp_sum += n;
        }
        sums[lane_id] = warp_sum;
    }
    __syncthreads();

    int blockSum = 0;
    if (warp_id > 0) {
        blockSum = sums[warp_id - 1];
    }
    value += blockSum;
    data[id] = value;

    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = value;
    }
}

__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (id > len)
        return;
    if (threadIdx.x == 0) {
        buf = partial_sums[blockIdx.x];
    }
    __syncthreads();
    data[id] += buf;
}
"""


# ── The portability decision itself ────────────────────────────────────────

class TestDeterminePortMode:
    def test_nvidia_shfl_scan_is_device_subset(self):
        assert determine_port_mode(NVIDIA_SHFL_SCAN_CUDA) == PORT_MODE_DEVICE_SUBSET

    def test_nvidia_shfl_scan_has_the_unresolved_header(self):
        assert unresolved_local_headers(NVIDIA_SHFL_SCAN_CUDA) == ['shfl_integral_image.cuh']

    def test_self_contained_with_resolvable_deps_stays_whole_program(self):
        assert is_self_contained(SELF_CONTAINED_RESOLVABLE)
        assert unresolved_local_headers(SELF_CONTAINED_RESOLVABLE) == []
        assert determine_port_mode(SELF_CONTAINED_RESOLVABLE) == PORT_MODE_WHOLE_PROGRAM

    def test_bare_kernel_snippet_is_whole_program(self):
        """No main() to split -- WHOLE_PROGRAM is the only sensible answer."""
        assert not is_self_contained(BARE_KERNEL)
        assert determine_port_mode(BARE_KERNEL) == PORT_MODE_WHOLE_PROGRAM

    def test_unresolved_header_alone_without_self_contained_is_whole_program(self):
        """DEVICE_SUBSET only exists to split a program that HAS a main() to
        split. A bare kernel with a stray unresolved include is still just
        WHOLE_PROGRAM (there's no driver to separate it from)."""
        src = '#include "ghost.cuh"\n__global__ void k(int* a){}\n'
        assert unresolved_local_headers(src) == ['ghost.cuh']
        assert not is_self_contained(src)
        assert determine_port_mode(src) == PORT_MODE_WHOLE_PROGRAM

    def test_self_contained_with_unresolvable_header_is_device_subset(self):
        assert determine_port_mode(SELF_CONTAINED_UNRESOLVABLE_HEADER) == PORT_MODE_DEVICE_SUBSET

    def test_a_vendored_local_header_does_not_trigger_device_subset(self):
        """A local header this repo DOES vendor must not spuriously trip the
        DEVICE_SUBSET path -- only a genuinely unresolvable one does."""
        # transpose.cu (or any sample) is vendored under sample_kernels/, so a
        # program that includes it by name should resolve.
        src = (
            '#include <cuda_runtime.h>\n'
            '#include "warp_reduce.cu"\n'  # exists under sample_kernels/cuda/
            '__global__ void k(int* a){}\n'
            'int main(){ return 0; }\n'
        )
        assert unresolved_local_headers(src) == []
        assert determine_port_mode(src) == PORT_MODE_WHOLE_PROGRAM


class TestRouterDelegatesToSpecParser:
    """The decision must be computed in exactly one place. These pin that
    ModelRouter's static/class methods are thin delegations, not independent
    reimplementations that could silently drift."""

    def test_router_unresolved_local_headers_matches_spec_parser(self):
        assert (ModelRouter._unresolved_local_headers(NVIDIA_SHFL_SCAN_CUDA)
                == unresolved_local_headers(NVIDIA_SHFL_SCAN_CUDA))

    def test_router_determine_port_mode_matches_spec_parser(self):
        assert (ModelRouter._determine_port_mode(NVIDIA_SHFL_SCAN_CUDA)
                == determine_port_mode(NVIDIA_SHFL_SCAN_CUDA))
        assert (ModelRouter._determine_port_mode(SELF_CONTAINED_RESOLVABLE)
                == determine_port_mode(SELF_CONTAINED_RESOLVABLE)
                == PORT_MODE_WHOLE_PROGRAM)


# ── Spec generation persists port_mode without breaking legacy behavior ────

class TestSpecGenerationPersistsPortMode:
    def test_auto_generated_nvidia_shfl_scan_spec_is_device_subset(self):
        spec = generate_spec_from_source("nvidia_shfl_scan", NVIDIA_SHFL_SCAN_CUDA)
        assert spec["port_mode"] == PORT_MODE_DEVICE_SUBSET
        assert spec["self_contained"] is True
        # Unlike the legacy WHOLE_PROGRAM self-contained path, DEVICE_SUBSET
        # DOES need harness config -- the verifier will synthesize one.
        assert "launch" in spec
        assert "output_readback" in spec

    def test_whole_program_self_contained_still_omits_harness_config(self):
        """Regression pin: the pre-existing WHOLE_PROGRAM behavior (launch/
        input_setup/output_readback are dead config once _generate_harness
        returns the source as-is) must be completely unaffected."""
        spec = generate_spec_from_source("k", SELF_CONTAINED_RESOLVABLE)
        assert spec["port_mode"] == PORT_MODE_WHOLE_PROGRAM
        assert spec["self_contained"] is True
        assert "launch" not in spec
        assert "input_setup" not in spec
        assert "output_readback" not in spec

    def test_bare_kernel_spec_is_whole_program_and_unaffected(self):
        spec = generate_spec_from_source("vector_add", BARE_KERNEL)
        assert spec["port_mode"] == PORT_MODE_WHOLE_PROGRAM
        assert "self_contained" not in spec
        assert "launch" in spec


# ── The committed nvidia_shfl_scan.json spec is correct ────────────────────

class TestNvidiaShflScanSpecFile:
    """The hand-written spec on disk -- fixed as part of this task."""

    @pytest.fixture
    def spec(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'src', 'verification', 'specs', 'nvidia_shfl_scan.json')
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def test_port_mode_is_device_subset(self, spec):
        assert spec["port_mode"] == PORT_MODE_DEVICE_SUBSET

    def test_readback_type_is_int_not_float(self, spec):
        """The bug: this kernel operates on int*, but the spec claimed float."""
        assert spec["output_readback"]["element_type"] == "int"
        assert spec["output_readback"]["format"] == "int_per_line"

    def test_kernel_function_matches_the_device_signature(self, spec):
        assert spec["kernel_function"] == "shfl_scan_test"
        types = [p["type"] for p in spec["params"]]
        assert types == ["int*", "int", "int*"]

    def test_launch_config_present(self, spec):
        assert spec["launch"]["grid"]["x"] >= 1
        assert spec["launch"]["block"]["x"] % 64 == 0, \
            "block size should be a multiple of 64 (wavefront) for a clean nWarps split"

    def test_dynamic_shared_mem_matches_wavefront64_warp_count(self, spec):
        block_x = spec["launch"]["block"]["x"]
        n_warps = block_x // 64
        assert spec["dynamic_shared_mem"] == n_warps * 4  # sizeof(int)

    def test_width_override_is_64_not_32(self, spec):
        """The whole porting job: width must become a wavefront64 quantity,
        not stay at CUDA's warp-32 value."""
        assert spec["kernel_args_override"] == "64"

    def test_spec_is_still_valid_json_and_loadable(self):
        agent = VerificationAgent()
        spec = agent.load_spec("nvidia_shfl_scan")
        assert spec is not None
        assert spec["port_mode"] == PORT_MODE_DEVICE_SUBSET


# ── _ensure_main_preserved is mode-driven and authoritative ────────────────

class TestEnsureMainPreservedIsModeDriven:
    def test_device_subset_never_restores_regardless_of_dependency_analysis(self):
        """The decline must not be an emergent side effect: even a ported
        text that WOULD otherwise pass _unsatisfied_main_calls (e.g. it kept
        no main() and the driver's calls all happen to resolve) must still
        not get main() restored in DEVICE_SUBSET mode."""
        ported_no_main = "__global__ void k(int* a){ a[0]=1; }\n"
        code, restored = ModelRouter._ensure_main_preserved(
            ported_no_main, SELF_CONTAINED_RESOLVABLE,
            port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert restored is False
        assert code == ported_no_main  # untouched

    def test_whole_program_mode_still_restores_as_before(self):
        """Regression pin: default port_mode (WHOLE_PROGRAM) must behave
        exactly as _ensure_main_preserved did before this change existed."""
        ported_no_main = "__global__ void add_one(int* data, int n){ }\n"
        code, restored = ModelRouter._ensure_main_preserved(
            ported_no_main, SELF_CONTAINED_RESOLVABLE)  # port_mode defaults
        assert restored is True
        assert "int main(" in code


# ── _extract_device_subset ─────────────────────────────────────────────────

class TestExtractDeviceSubset:
    def test_extracts_exactly_the_two_device_kernels(self):
        subset = ModelRouter._extract_device_subset(NVIDIA_SHFL_SCAN_CUDA)
        assert "__global__ void shfl_scan_test" in subset
        assert "__global__ void uniform_add" in subset

    def test_drops_the_host_driver_entirely(self):
        subset = ModelRouter._extract_device_subset(NVIDIA_SHFL_SCAN_CUDA)
        for dropped in ("int main(", "shuffle_simple_test", "shuffle_integral_image_test",
                        "CPUverify", "checkCudaErrors", "findCudaDevice"):
            assert dropped not in subset, f"{dropped!r} should have been dropped"

    def test_is_dramatically_smaller_than_the_full_source(self):
        """The actual fix for 'the model tries to reproduce the whole program
        and runs out of budget partway through': less to reproduce."""
        subset = ModelRouter._extract_device_subset(NVIDIA_SHFL_SCAN_CUDA)
        full_lines = len(NVIDIA_SHFL_SCAN_CUDA.splitlines())
        subset_lines = len(subset.splitlines())
        assert subset_lines < full_lines * 0.4, \
            f"{subset_lines} lines is not a meaningful reduction from {full_lines}"

    def test_device_kernel_bodies_are_not_truncated(self):
        """Brace-matched extraction: the LAST line of each kernel must survive."""
        subset = ModelRouter._extract_device_subset(NVIDIA_SHFL_SCAN_CUDA)
        assert "partial_sums[blockIdx.x] = value;" in subset  # end of shfl_scan_test
        assert "data[id] += buf;" in subset  # end of uniform_add

    def test_falls_back_to_full_source_when_no_device_definition_found(self):
        host_only = "int main(){ return 0; }\n"
        assert ModelRouter._extract_device_subset(host_only) == host_only


class TestDeviceSymbolNames:
    def test_returns_exactly_the_device_kernels(self):
        names = ModelRouter._device_symbol_names(NVIDIA_SHFL_SCAN_CUDA)
        assert names == {"shfl_scan_test", "uniform_add"}

    def test_excludes_host_functions(self):
        names = ModelRouter._device_symbol_names(NVIDIA_SHFL_SCAN_CUDA)
        assert "main" not in names
        assert "shuffle_simple_test" not in names
        assert "shuffle_integral_image_test" not in names
        assert "CPUverify" not in names
        assert "iDivUp" not in names


# ── _postprocess_port: missing_symbols scoped in DEVICE_SUBSET mode ────────

class TestPostprocessPortSymbolScoping:
    """This is the fix that lets Part A actually converge, not just exist
    architecturally: without it the refine prompt would tell the coder to
    'restore' main()/shuffle_simple_test/etc -- directly contradicting the
    DEVICE_SUBSET instruction to drop them."""

    def test_device_subset_does_not_flag_dropped_host_functions_as_missing(self):
        r = ModelRouter(api_key="test")
        raw = "```hip\n" + NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT + "```\n"
        _, _, _, structural = r._postprocess_port(
            raw, NVIDIA_SHFL_SCAN_CUDA, port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        for host_fn in ("main", "shuffle_simple_test", "shuffle_integral_image_test",
                        "CPUverify", "verifyDataRowSums", "iDivUp"):
            assert host_fn not in structural.missing_symbols

    def test_device_subset_would_still_flag_a_dropped_kernel(self):
        """The filter narrows to device names -- it must not silently swallow
        a REAL defect (the coder also dropping one of the two kernels)."""
        r = ModelRouter(api_key="test")
        only_one_kernel = (
            "#include <hip/hip_runtime.h>\n"
            "__global__ void shfl_scan_test(int *data, int width, int *partial_sums){ }\n"
        )
        raw = "```hip\n" + only_one_kernel + "```\n"
        _, _, _, structural = r._postprocess_port(
            raw, NVIDIA_SHFL_SCAN_CUDA, port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert "uniform_add" in structural.missing_symbols

    def test_whole_program_mode_is_unaffected_by_the_filter(self):
        """Regression pin: default port_mode must not change existing
        missing-symbol reporting for any other kernel."""
        r = ModelRouter(api_key="test")
        raw = "```hip\n__global__ void add_one(int *data, int n){ }\n```\n"
        _, _, _, structural = r._postprocess_port(raw, SELF_CONTAINED_RESOLVABLE)
        # main() was dropped and WHOLE_PROGRAM mode restores it via
        # _ensure_main_preserved, so it should NOT show up as missing either
        # -- but for a DIFFERENT reason (it got restored, not filtered).
        assert "main" not in structural.missing_symbols


# ── Coder/refine prompts: DEVICE_SUBSET instructions ───────────────────────

class TestCoderPromptDeviceSubset:
    def test_device_subset_prompt_shows_only_the_device_subset(self):
        r = ModelRouter(api_key="test")
        p = r._build_kimi_code_prompt(
            NVIDIA_SHFL_SCAN_CUDA, [], port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert "int main(" not in p
        assert "shuffle_simple_test" not in p
        assert "__global__ void shfl_scan_test" in p

    def test_device_subset_prompt_forbids_writing_main(self):
        r = ModelRouter(api_key="test")
        p = r._build_kimi_code_prompt(
            NVIDIA_SHFL_SCAN_CUDA, [], port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert "DEVICE_SUBSET" in p
        assert "Do NOT write a main()" in p

    def test_device_subset_prompt_does_not_contain_the_whole_program_instruction(self):
        r = ModelRouter(api_key="test")
        p = r._build_kimi_code_prompt(
            NVIDIA_SHFL_SCAN_CUDA, [], port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert "Do NOT strip main() — it drives the full test." not in p

    def test_whole_program_prompt_is_byte_identical_to_before_this_feature(self):
        """The strongest regression pin available: default port_mode must
        produce EXACTLY the prompt text the pipeline produced before
        DEVICE_SUBSET existed."""
        r = ModelRouter(api_key="test")
        with_default = r._build_kimi_code_prompt(NVIDIA_SHFL_SCAN_CUDA, [])
        with_explicit = r._build_kimi_code_prompt(
            NVIDIA_SHFL_SCAN_CUDA, [], port_mode=router_mod.PORT_MODE_WHOLE_PROGRAM)
        assert with_default == with_explicit
        assert "Do NOT strip main() — it drives the full test." in with_default
        assert "DEVICE_SUBSET" not in with_default

    def test_refine_prompt_device_subset_forbids_main(self):
        r = ModelRouter(api_key="test")
        p = r._build_kimi_refine_prompt(
            NVIDIA_SHFL_SCAN_CUDA, "previous ported code", "fix this", [],
            port_mode=router_mod.PORT_MODE_DEVICE_SUBSET)
        assert "DEVICE_SUBSET" in p
        assert "Do NOT add a main()" in p

    def test_refine_prompt_whole_program_unchanged(self):
        r = ModelRouter(api_key="test")
        with_default = r._build_kimi_refine_prompt(
            NVIDIA_SHFL_SCAN_CUDA, "previous code", "feedback", [])
        with_explicit = r._build_kimi_refine_prompt(
            NVIDIA_SHFL_SCAN_CUDA, "previous code", "feedback", [],
            port_mode=router_mod.PORT_MODE_WHOLE_PROGRAM)
        assert with_default == with_explicit


# ── Verifier: harness synthesis is authoritative under DEVICE_SUBSET ───────

class TestVerifierHarnessSynthesis:
    @pytest.fixture
    def agent(self):
        return VerificationAgent()

    def test_device_subset_synthesizes_a_harness_not_return_as_is(self, agent):
        harness, k_start, k_end = agent._generate_harness(
            "nvidia_shfl_scan", "", NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT)
        # A synthesized harness contains a main() OUTSIDE the ported kernel's
        # line range -- the "return as-is" path would have no main() at all
        # (this reference port correctly has none) and would report the whole
        # file as the kernel range.
        assert "int main() {" in harness
        assert k_end < len(harness.splitlines())
        main_line = next(i for i, l in enumerate(harness.splitlines(), 1)
                         if "int main() {" in l)
        assert main_line > k_end, "main() must be synthesized AFTER the ported code, not inside it"

    def test_harness_launches_with_wavefront64_width(self, agent):
        harness, _, _ = agent._generate_harness(
            "nvidia_shfl_scan", "", NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT)
        assert "shfl_scan_test<<<dim3(4,1,1), dim3(256,1,1), 16>>>" in harness

    def test_device_subset_overrides_even_a_ported_main_present_by_mistake(self, agent):
        """Authoritative: port_mode wins even if the ported text disobeyed
        its instructions and included a main() anyway (e.g. a stale restore,
        or a coder that ignored the prompt)."""
        disobedient = (NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT
                       + "\nint main() { return 0; }\n")
        harness, k_start, k_end = agent._generate_harness(
            "nvidia_shfl_scan", "", disobedient)
        # The "return ported_kernel_source as-is" shortcut always starts the
        # kernel range at harness line 1 (see _generate_harness). Taking the
        # synthesized-harness path instead means a preamble precedes it, so
        # k_start > 1 -- and the harness text itself differs from the raw,
        # disobedient input rather than echoing it verbatim.
        assert k_start > 1
        assert harness != disobedient

    def test_kernel_without_port_mode_field_is_unaffected(self, agent, tmp_path, monkeypatch):
        """Regression pin: any spec written before this contract existed
        (self_contained: true, no port_mode key) must behave exactly as
        before -- return the ported source as-is."""
        from verification import verifier as verifier_mod
        spec_dir = tmp_path / "specs"
        spec_dir.mkdir()
        (spec_dir / "legacy_kernel.json").write_text(
            json.dumps({"kernel_name": "legacy_kernel", "self_contained": True}),
            encoding="utf-8")
        monkeypatch.setattr(verifier_mod, "_SPEC_DIR", spec_dir)
        agent2 = VerificationAgent()
        ported = "int main() { return 0; }\n"
        harness, k_start, k_end = agent2._generate_harness("legacy_kernel", "", ported)
        assert harness == ported
        assert k_end == len(ported.splitlines())


# ── Integration: the reference port actually compiles ──────────────────────

_HIPCC_AVAILABLE = VerificationAgent()._hipcc_available


@pytest.mark.skipif(not _HIPCC_AVAILABLE,
                    reason="hipcc not available in this environment -- "
                           "run on the AMD GPU box to exercise this path")
class TestRealCompile:
    """Gated behind hipcc availability per the mission doc's A.4 item 3: keep
    the path runnable locally on the GPU box, but don't fail CI without one."""

    def test_reference_device_subset_port_compiles_clean(self, tmp_path):
        agent = VerificationAgent()
        agent.build_dir = tmp_path
        result = agent.quick_compile_check(
            NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT,
            kernel_name="nvidia_shfl_scan")
        assert result["compile_success"], result.get("errors")

    def test_reference_device_subset_port_runs_without_crashing(self, tmp_path):
        agent = VerificationAgent()
        agent.build_dir = tmp_path
        cc = agent.quick_compile_check(
            NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT,
            kernel_name="nvidia_shfl_scan")
        assert cc["compile_success"], cc.get("errors")
        rc = agent.quick_run_check("nvidia_shfl_scan")
        assert rc["run_success"], rc.get("run_output")


# ── End-to-end: route() no longer self-sabotages on this kernel ───────────

from router import AgentResult  # noqa: E402


def _fake_call_model(coder_output):
    """A _call_model_impl that returns *coder_output* for the coder role and
    trivially succeeds for planner/evaluator, mirroring
    tests/test_debug_mode_integration.py's fake-model style."""
    def impl(self, model_key, prompt, system_prompt="", prefill="",
             max_seconds=None, max_tokens_override=None):
        if model_key == "deepseek":
            return AgentResult("deepseek", True, "Plan: port the two device kernels.", 0.8, 80, 2000.0)
        if model_key == "glm":
            return AgentResult("glm", True, coder_output, 0.7, 900, 8000.0)
        if model_key == "kimi27":
            return AgentResult("kimi27", True, '{"pass":true,"issues":[]}', 0.9, 200, 1000.0)
        return AgentResult(model_key, False, "", 0.0)
    return impl


class _FakeVerifier:
    """Simulates a hipcc that accepts the reference DEVICE_SUBSET port and
    rejects anything containing a host driver -- standing in for a real
    compiler on a box without hipcc, so the LOOP's behavior (not the real
    compiler's) is what gets proven here."""

    def attach_debug_session(self, s): pass
    def detach_debug_session(self): pass

    def quick_compile_check(self, src, kernel_name="k"):
        if "int main(" in src or "shuffle_simple_test" in src:
            return {"compile_success": False,
                    "compile_output": "error: undefined reference to findCudaDevice",
                    "errors": ["test.cpp:1:1: error: undefined reference to findCudaDevice"],
                    "error_origins": ["ported_code"], "error_context": [],
                    "all_harness_origin": False}
        return {"compile_success": True, "compile_output": "", "errors": [],
                "error_origins": [], "error_context": [], "all_harness_origin": False}

    def quick_run_check(self, kernel_name):
        return {"run_success": True, "run_output": "256\n256\n256\n256\n",
                "run_exit_code": 0, "signal": "", "benchmark_us": 5.0}


class TestEndToEndConvergence:
    """The actual claim behind Part A: not just that DEVICE_SUBSET exists as a
    label, but that route() reaches a passing compile on this kernel — and
    does not spend the budget on structural rejects and a doomed main()
    restore the way the original 156.6s FAILED run did."""

    def test_a_correct_device_subset_port_converges_on_the_first_generation(self, monkeypatch):
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_call_model("```hip\n"
                                             + NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT
                                             + "```\n"))
        r = ModelRouter(api_key="fake")
        result = r.route(NVIDIA_SHFL_SCAN_CUDA, [{"pattern": "shfl_up_sync"}],
                         max_iterations=3, verifier=_FakeVerifier(),
                         kernel_name="nvidia_shfl_scan", max_seconds=0,
                         fast_path=False)
        assert result["port_mode"] == PORT_MODE_DEVICE_SUBSET
        assert result["compile_passed"] is True
        assert result["iterations_used"] <= 1, (
            "a correct first generation should not need a refine iteration; "
            f"changes={result['changes']}")
        assert "int main(" not in result["ported_code"]

    def test_main_not_restored_note_reflects_device_subset_not_a_blocker(self, monkeypatch):
        """The coder dropping main() in DEVICE_SUBSET mode is CORRECT behavior,
        not a defect masked by dependency analysis -- the changelog must say so."""
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_call_model("```hip\n"
                                             + NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT
                                             + "```\n"))
        r = ModelRouter(api_key="fake")
        result = r.route(NVIDIA_SHFL_SCAN_CUDA, [], max_iterations=2,
                         verifier=_FakeVerifier(), kernel_name="nvidia_shfl_scan",
                         max_seconds=0, fast_path=False)
        device_subset_notes = [c for c in result["changes"] if "DEVICE_SUBSET" in c]
        assert device_subset_notes, "expected a [port-mode] or [main] note naming DEVICE_SUBSET"
