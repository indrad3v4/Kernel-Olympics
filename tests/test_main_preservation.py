"""Tests for the PR #13 post-mortem fixes (prompts/to-fable-post-pr13-gap-analysis.md).

The 2026-07-09 run spent all 180s and produced nothing because a self-contained
program's ``main()`` was dropped by the coder, and every phase downstream treated
the resulting ``undefined symbol: main`` as a code defect to reason about.

  P0  main() preservation          → TestExtractMain, TestEnsureMainPreserved
  P0  linker-error short-circuit   → TestLinkerClassifiers, TestRouteLinkerShortCircuit
  P1  wavefront-delta plan prompt  → TestDeltaPlanPrompt
  P1  GLM self_contained context   → TestGlmSelfContainedContext
  P2  budget-aware refine dispatch → TestRefineBudgetGuard
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

REPO = Path(__file__).resolve().parents[1]

from router import (
    ModelRouter, AgentResult, PLAN_DELTA_MAX_TOKENS,
    COMPILE_RESERVE_SECONDS, MIN_LLM_TIMEOUT_SECONDS,
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """No test in this file may reach the network."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


# A complete program: kernel + driver, with a main() that has nested braces,
# a brace inside a string literal, and a brace inside a comment — the three
# things a naive "scan to the next }" extractor gets wrong.
SELF_CONTAINED_CUDA = """
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    input[tid] = __shfl_up_sync(0xffffffff, input[tid], 1);
}

int main(int argc, char** argv) {
    float* d = nullptr;
    cudaMalloc(&d, 256 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        if (i % 2) { printf("brace in string: {\\n"); }
    }
    // a lone } in a comment
    cudaFree(d);
    return 0;
}
"""

# The coder's failure mode: kernels only, driver dropped.
PORT_WITHOUT_MAIN = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    input[tid] = __shfl_up(input[tid], 1);
}
"""

BARE_KERNEL_CUDA = """
#include <cuda_runtime.h>
__global__ void k(float* x) { int i = threadIdx.x; x[i] = 0.f; }
"""

LINKER_ERRS = [
    "ld.lld: error: undefined symbol: main",
    "clang++: error: linker command failed with exit code 1",
]


# ── P0: main() extraction ────────────────────────────────────────────────

class TestExtractMain:
    def test_extracts_a_brace_balanced_driver(self):
        m = ModelRouter._extract_main(SELF_CONTAINED_CUDA)
        assert m.startswith("int main(")
        assert m.endswith("}")
        assert m.count("{") == m.count("}")
        # the whole body came along, not just up to the first nested close
        assert "cudaFree(d);" in m
        assert "return 0;" in m

    def test_braces_in_strings_and_comments_do_not_terminate_it(self):
        m = ModelRouter._extract_main(SELF_CONTAINED_CUDA)
        assert 'brace in string: {' in m
        assert "a lone } in a comment" in m

    def test_returns_empty_for_a_bare_kernel(self):
        assert ModelRouter._extract_main(BARE_KERNEL_CUDA) == ""

    def test_ignores_a_forward_declaration(self):
        assert ModelRouter._extract_main("int main(int, char**);\n") == ""

    def test_returns_empty_on_unbalanced_source(self):
        assert ModelRouter._extract_main("int main() {\n  if (1) {\n") == ""


# ── P0: main() preservation ──────────────────────────────────────────────

class TestEnsureMainPreserved:
    def test_restores_a_dropped_driver(self):
        code, restored = ModelRouter._ensure_main_preserved(
            PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        assert restored is True
        assert ModelRouter._is_self_contained(code)
        assert "scan_kernel" in code, "the port's own kernel must survive"

    def test_restores_the_driver_verbatim_still_speaking_cuda(self):
        """The driver is appended RAW. _postprocess_port runs the mechanical pass
        over the combined file right afterwards, which both translates it and —
        because the file now has a main() — injects the helper shims it calls."""
        code, _ = ModelRouter._ensure_main_preserved(
            PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        tail = code.split("main() restored")[1]
        assert "cudaMalloc" in tail and "cudaFree" in tail

    def test_restore_does_not_duplicate_the_injected_preamble(self):
        """Hipifying an isolated main() would re-inject the helper shims and
        WAVEFRONT_SIZE. Appending those to a file that has them is a
        redefinition error — one link error traded for a dozen compile ones."""
        hipified, _ = ModelRouter._hipify_source(SELF_CONTAINED_CUDA)
        port = hipified.replace(ModelRouter._extract_main(hipified), "")
        code, restored = ModelRouter._ensure_main_preserved(port, SELF_CONTAINED_CUDA)
        assert restored is True
        assert code.count("#define WAVEFRONT_SIZE") == 1
        assert code.count("int main(") == 1

    def test_is_a_noop_when_the_port_kept_its_main(self):
        code, restored = ModelRouter._ensure_main_preserved(
            SELF_CONTAINED_CUDA, SELF_CONTAINED_CUDA)
        assert restored is False
        assert code == SELF_CONTAINED_CUDA

    def test_is_a_noop_for_a_bare_kernel_snippet(self):
        code, restored = ModelRouter._ensure_main_preserved(
            PORT_WITHOUT_MAIN, BARE_KERNEL_CUDA)
        assert restored is False
        assert code == PORT_WITHOUT_MAIN

    def test_is_idempotent(self):
        once, _ = ModelRouter._ensure_main_preserved(
            PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        twice, restored = ModelRouter._ensure_main_preserved(once, SELF_CONTAINED_CUDA)
        assert restored is False
        assert twice == once

    def test_empty_inputs_are_safe(self):
        assert ModelRouter._ensure_main_preserved("", SELF_CONTAINED_CUDA) == ("", False)
        assert ModelRouter._ensure_main_preserved(PORT_WITHOUT_MAIN, "") == (
            PORT_WITHOUT_MAIN, False)


class TestPostprocessPort:
    def test_applies_extract_fix_and_restore_in_one_step(self, router):
        raw = f"```cpp\n{PORT_WITHOUT_MAIN}\n```"
        code, changelog, restored, _structural = router._postprocess_port(raw, SELF_CONTAINED_CUDA)
        assert restored is True
        assert ModelRouter._is_self_contained(code)
        assert any("main() restored" in c for c in changelog)

    def test_reports_no_restore_when_none_was_needed(self, router):
        raw = f"```cpp\n{SELF_CONTAINED_CUDA}\n```"
        _, changelog, restored, _structural = router._postprocess_port(raw, SELF_CONTAINED_CUDA)
        assert restored is False
        assert not any("main() restored" in c for c in changelog)

    def test_restored_driver_gets_the_helper_shims_it_calls(self, router):
        """ORDER BUG. _fix_ported_code injects findCudaDevice / sdk*Timer /
        StopWatchInterface only when it sees a self-contained program. Restoring
        main() AFTER the mechanical pass means the file was not self-contained at
        fix time, so the shims were never injected and the driver we just
        reattached called findCudaDevice into a void."""
        code, _, restored, _structural = router._postprocess_port(
            PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        assert restored is True
        assert "static inline int findCudaDevice" in code
        assert "struct StopWatchInterface {" in code

    def test_restored_driver_is_hipified_by_the_mechanical_pass(self, router):
        code, _, _, _ = router._postprocess_port(PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        tail = code.split("main() restored")[1]
        assert "hipMalloc" in tail and "hipFree" in tail
        assert ModelRouter._residual_cuda_symbols(tail) == []

    def test_second_pass_does_not_duplicate_the_shim_block(self, router):
        """The re-injection guard used to key on the `// _verifier_helper_shims`
        marker comment. _extract_code's raw-code fallback starts at the first
        #include and discards every line above it, marker included — so the next
        pass re-injected the block and hipcc reported a redefinition of
        StopWatchInterface. The guard now keys on the definitions themselves."""
        once, _, _, _ = router._postprocess_port(PORT_WITHOUT_MAIN, SELF_CONTAINED_CUDA)
        twice, _, restored, _ = router._postprocess_port(once, SELF_CONTAINED_CUDA)
        assert restored is False
        assert twice.count("struct StopWatchInterface {") == 1
        assert twice.count("static inline int findCudaDevice") == 1
        assert twice.count("#define WAVEFRONT_SIZE") == 1
        assert twice.count("int main(") == 1


# ── The driver must not be restored into a file that cannot satisfy it ───

# A driver that depends on a helper the coder may legitimately drop, mirroring
# nvidia_shfl_scan.cu: shuffle_integral_image_test() needs shfl_integral_image.cuh,
# a header this repo never vendored, so the coder is told to drop that code path.
CUDA_WITH_HELPER = """
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k(float* x) { x[threadIdx.x] = 0.f; }

bool portable_test(int argc, char** argv) { return true; }

bool header_dependent_test() { return shfl_integral_image_kernel(); }

int main(int argc, char** argv) {
    int dev = findCudaDevice(argc, (const char**)argv);
    if (dev < 0) exit(EXIT_WAIVED);
    bool a = portable_test(argc, argv);
    bool b = header_dependent_test();
    return (a && b) ? 0 : 1;
}
"""


class TestRestoreDependencies:
    def test_declines_when_the_driver_calls_a_helper_the_port_dropped(self, router):
        """Reattaching a driver that calls a function nobody defines converts a
        compile problem into an unfixable link problem — and every refinement then
        fights the restore, which re-appends the same driver each iteration."""
        port = ("#include <hip/hip_runtime.h>\n"
                "__global__ void k(float* x) { x[threadIdx.x] = 0.f; }\n"
                "bool portable_test(int argc, char** argv) { return true; }\n")
        code, changelog, restored, _ = router._postprocess_port(port, CUDA_WITH_HELPER)
        assert restored is False
        assert "header_dependent_test" not in code
        assert any("NOT restored" in c and "header_dependent_test" in c
                   for c in changelog), changelog

    def test_restores_when_every_helper_the_driver_calls_survives(self, router):
        port = ("#include <hip/hip_runtime.h>\n"
                "__global__ void k(float* x) { x[threadIdx.x] = 0.f; }\n"
                "bool portable_test(int argc, char** argv) { return true; }\n"
                "bool header_dependent_test() { return true; }\n")
        code, _, restored, _ = router._postprocess_port(port, CUDA_WITH_HELPER)
        assert restored is True
        assert ModelRouter._is_self_contained(code)

    def test_unsatisfied_calls_ignores_runtime_and_library_symbols(self):
        """printf/exit/hipMalloc are not defined in the original .cu, so they can
        never be reported as dropped helpers."""
        main = ModelRouter._extract_main(CUDA_WITH_HELPER)
        port = ("bool portable_test(int a, char** b) { return true; }\n"
                "bool header_dependent_test() { return true; }\n")
        missing = ModelRouter._unsatisfied_main_calls(main, port, CUDA_WITH_HELPER)
        assert missing == []
        assert "printf" not in missing and "exit" not in missing
        assert "findCudaDevice" not in missing  # a shim symbol, not defined in the .cu

    def test_real_kernel_driver_needs_the_header_dependent_helper(self):
        """The regression this whole class exists for, on the real sample."""
        src = (REPO / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu").read_text(
            encoding="utf-8", errors="replace")
        main = ModelRouter._extract_main(src)
        kernel_only = ("#include <hip/hip_runtime.h>\n"
                       "__global__ void shfl_scan_test(int* d, int w, int* p) {}\n")
        missing = ModelRouter._unsatisfied_main_calls(main, kernel_only, src)
        assert "shuffle_integral_image_test" in missing
        assert "shuffle_simple_test" in missing


class TestDroppedPathConflict:
    def test_coder_is_told_how_to_reconcile_main_with_a_dropped_path(self, router):
        """"Keep main()" and "drop the header-dependent path" collide on one line:
        main() calls that path. Unresolved, the coder either keeps a call to a
        function it never defined, or deletes main(). Both are link errors."""
        src = (REPO / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu").read_text(
            encoding="utf-8", errors="replace")
        hipified, _ = router._hipify_source(src)
        p = router._build_kimi_code_prompt(src, [], preprocessed_source=hipified)
        assert "Do NOT strip main()" in p
        assert "RESOLVING THE CONFLICT" in p
        assert "do not delete main()" in p

    def test_no_conflict_note_when_every_header_resolves(self, router):
        p = router._build_kimi_code_prompt(SELF_CONTAINED_CUDA, [])
        assert "Do NOT strip main()" in p
        assert "RESOLVING THE CONFLICT" not in p


class TestHelperShims:
    def test_shim_defines_exit_waived(self):
        """EXIT_WAIVED is a helper_string.h macro, not a std one. _fix_ported_code
        strips helper_*.h, so every restored NVIDIA driver referenced it undeclared."""
        code, _ = ModelRouter._hipify_source(SELF_CONTAINED_CUDA)
        assert "#define EXIT_WAIVED" in code

    def test_real_sample_uses_exit_waived_and_the_shim_supplies_it(self):
        src = (REPO / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu").read_text(
            encoding="utf-8", errors="replace")
        assert "EXIT_WAIVED" in src, "sample no longer exercises this path"
        hipified, _ = ModelRouter._hipify_source(src)
        assert "#define EXIT_WAIVED" in hipified

    def test_shim_is_injected_exactly_once_across_repeated_passes(self):
        code, _ = ModelRouter._hipify_source(SELF_CONTAINED_CUDA)
        again, _ = ModelRouter._hipify_source(code)
        assert again.count("struct StopWatchInterface {") == 1
        assert again.count("#define EXIT_WAIVED") == 1


# ── Artifacts that are text, not C++ ─────────────────────────────────────

class TestSanitizeExtracted:
    def test_html_truncation_marker_never_reaches_the_compiler(self):
        """`<!-- TRUNCATED ... -->` was appended to the model's content and rode
        the extraction fallbacks into the file hipcc compiled. At C++ file scope
        `<` is `expected external declaration` (col 1) and `--` is
        `expected unqualified-id` (col 3) — two errors on one line that no model
        can fix, because no model wrote them."""
        out = ("#include <hip/hip_runtime.h>\n"
               "__global__ void k(float* x) {}\n"
               "<!-- TRUNCATED: output hit max_tokens limit -->")
        code = ModelRouter._extract_code(out)
        assert "<!--" not in code and "-->" not in code
        assert "__global__" in code

    def test_cpp_line_comment_truncation_marker_is_also_stripped(self):
        out = ("#include <hip/hip_runtime.h>\n"
               "__global__ void k(float* x) {}\n"
               "// TRUNCATED: output hit max_tokens limit")
        code = ModelRouter._extract_code(out)
        assert "TRUNCATED" not in code

    @pytest.mark.parametrize("tag", ["c++", "C++", "HIP", "cu", "", "cpp"])
    def test_any_fence_language_tag_is_handled(self, tag):
        """An allow-list of (cuda|hip|cpp|python) meant ```c++ fell through to the
        raw-code strategy, which swallowed the closing fence and the prose after."""
        out = (f"Here is the port:\n```{tag}\n"
               "#include <hip/hip_runtime.h>\n"
               "__global__ void k(float* x) {}\n"
               "```\nExplanation: I widened the shuffle mask to 0x3f.")
        code = ModelRouter._extract_code(out)
        assert "```" not in code
        assert "Explanation" not in code
        assert "__global__" in code

    def test_trailing_prose_after_a_stray_fence_is_dropped(self):
        code = ModelRouter._sanitize_extracted(
            "#include <hip/hip_runtime.h>\nint main(){return 0;}\n```\nAll done!")
        assert "```" not in code and "All done" not in code
        assert "int main()" in code

    def test_clean_code_is_left_alone(self):
        src = "#include <hip/hip_runtime.h>\nint main(){return 0;}"
        assert ModelRouter._sanitize_extracted(src) == src


# ── P0: linker-error classification ──────────────────────────────────────

class TestLinkerClassifiers:
    def test_the_real_run_errors_are_linker_only(self):
        # verifier tags only the first line "link"; its clang++ mate is "unknown".
        assert ModelRouter._is_linker_only(LINKER_ERRS, ["link", "unknown"]) is True
        assert ModelRouter._is_missing_main_error(LINKER_ERRS, ["link", "unknown"]) is True

    def test_a_compile_error_is_not_linker_only(self):
        errs = ["t.cpp:55:39: error: use of undeclared identifier 'n'"]
        assert ModelRouter._is_linker_only(errs, ["ported_code"]) is False

    def test_a_mixed_set_is_not_linker_only(self):
        errs = LINKER_ERRS[:1] + ["k.cpp:12:3: error: expected ';'"]
        assert ModelRouter._is_linker_only(errs, ["link", "ported_code"]) is False

    def test_an_undefined_symbol_other_than_main_is_not_missing_main(self):
        errs = ["ld.lld: error: undefined symbol: helperFn(int)",
                "clang++: error: linker command failed with exit code 1"]
        assert ModelRouter._is_linker_only(errs, ["unknown", "unknown"]) is True
        assert ModelRouter._is_missing_main_error(errs, ["unknown", "unknown"]) is False

    def test_no_errors_is_not_linker_only(self):
        assert ModelRouter._is_linker_only([], []) is False


class TestRouteLinkerShortCircuit:
    @patch.object(ModelRouter, '_call_model')
    def test_dropped_main_is_restored_before_the_first_compile(self, mock_call, router):
        """The driver is reattached mechanically the moment the coder returns, so
        hipcc never sees `undefined symbol: main` and no LLM phase is ever asked to
        reason about it. The 2026-07-09 run spent its whole budget on that question."""
        import tempfile
        from pathlib import Path
        build_dir = Path(tempfile.mkdtemp())

        def side_effect(model_key, *a, **k):
            if model_key == "glm":
                return AgentResult(model_key, True,
                                   f"```cpp\n{PORT_WITHOUT_MAIN}\n```", 0.5)
            if model_key == "kimi27":
                return AgentResult(model_key, True, '{"pass": true, "issues": []}', 0.9)
            return AgentResult(model_key, True, "plan", 0.5)
        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.build_dir = build_dir
        verifier._generate_harness.return_value = ("// mock harness", 0, 0)
        verifier._compile.return_value = (True, "", Path("/dev/null"))
        verifier._classify_error_origin.return_value = "harness"
        verifier._warn_if_legacy_harness.return_value = None
        # _run needs to succeed so the test's compile-pass is not reverted
        verifier._run.return_value = (True, "42\n", 0, 0)
        verifier._signal_name.return_value = ""

        result = router.route(SELF_CONTAINED_CUDA, [], max_iterations=4,
                              verifier=verifier, kernel_name="test_mainlink",
                              max_seconds=0, fast_path=False)

        assert result["compile_passed"] is True
        assert ModelRouter._is_self_contained(result["ported_code"])
        assert any("[main]" in c and "restored" in c for c in result["changes"])

        # Whatever was handed to _inloop_compile must contain main().
        compiled_src = (build_dir / "loop_test_mainlink" / "test_mainlink.hip.cpp")
        compiled = compiled_src.read_text(encoding="utf-8") if compiled_src.exists() else result["ported_code"]
        assert ModelRouter._is_self_contained(compiled)

        called = [c.args[0] for c in mock_call.call_args_list]
        assert called.count("glm") == 1, f"coder was re-invoked: {called}"
        assert called.count("deepseek") == 1, f"planner re-ran: {called}"

    @patch.object(ModelRouter, '_call_model')
    def test_linker_only_errors_skip_glm_and_replan(self, mock_call, router):
        """An undefined symbol that is NOT main still skips the analyst and the
        planner — neither can conjure a missing symbol — but the coder refines."""
        import tempfile
        from pathlib import Path
        other = ["ld.lld: error: undefined symbol: helperFn(int)",
                 "clang++: error: linker command failed with exit code 1"]
        build_dir = Path(tempfile.mkdtemp())

        def side_effect(model_key, *a, **k):
            if model_key == "glm":
                return AgentResult(model_key, True,
                                   f"```cpp\n{SELF_CONTAINED_CUDA}\n```", 0.5)
            return AgentResult(model_key, True, "plan", 0.5)
        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.build_dir = build_dir
        # Compile output contains linker errors that _inloop_compile parses
        verifier._generate_harness.return_value = ("// mock harness", 0, 0)
        verifier._compile.return_value = (False, "\n".join(other), Path("/dev/null"))
        # Linker errors are "unknown" origin (not harness, not ported_code)
        verifier._classify_error_origin.side_effect = lambda text, ks, ke: "unknown"
        verifier._warn_if_legacy_harness.return_value = None

        result = router.route(SELF_CONTAINED_CUDA, [], max_iterations=2,
                              verifier=verifier, kernel_name="test_linkonly",
                              max_seconds=0, fast_path=False)

        called = [c.args[0] for c in mock_call.call_args_list]
        assert "kimi27" not in called, f"Kimi analyst ran on a link error: {called}"
        assert called.count("deepseek") == 1, f"a re-plan fired: {called}"
        assert any("[linker]" in c for c in result["changes"])


# ── P1: wavefront-delta plan prompt ──────────────────────────────────────

class TestDeltaPlanPrompt:
    def test_delta_prompt_omits_the_cuda_original(self, router):
        hipified, _ = router._hipify_source(SELF_CONTAINED_CUDA)
        p = router._build_deepseek_plan_prompt(
            SELF_CONTAINED_CUDA, [], hipified_source=hipified)
        assert "```cuda" not in p
        assert "```hip" in p
        assert "Do not plan it again" in p

    def test_delta_prompt_is_materially_shorter_on_a_real_kernel(self, router):
        """The saving is not embedding a 15k-char original the draft already
        translates. On a toy kernel the two prompts are comparable; the win only
        shows on the program that actually blew the budget."""
        real = REPO / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu"
        src = real.read_text(encoding="utf-8", errors="replace")
        hipified, _ = router._hipify_source(src)
        full = router._build_deepseek_plan_prompt(src, [])
        delta = router._build_deepseek_plan_prompt(src, [], hipified_source=hipified)
        assert len(delta) < len(full) * 0.6, (
            f"delta plan prompt is {len(delta)} chars vs full {len(full)}")

    def test_delta_prompt_keeps_the_wavefront_checklist(self, router):
        hipified, _ = router._hipify_source(SELF_CONTAINED_CUDA)
        p = router._build_deepseek_plan_prompt(
            SELF_CONTAINED_CUDA, [], hipified_source=hipified)
        assert "__shfl_*_sync" in p
        assert "blockDim/64" in p

    def test_no_hipified_source_keeps_the_full_prompt(self, router):
        p = router._build_deepseek_plan_prompt(SELF_CONTAINED_CUDA, [])
        assert "```cuda" in p
        assert "Do not plan it again" not in p

    @patch.object(ModelRouter, '_call_model')
    def test_route_caps_the_delta_plan_token_budget(self, mock_call, router):
        mock_call.return_value = AgentResult("deepseek", False, "", 0.0)
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}

        router.route(SELF_CONTAINED_CUDA, [], max_iterations=1, verifier=verifier,
                     kernel_name="test_deltacap", max_seconds=0, fast_path=False)

        plan_calls = [c for c in mock_call.call_args_list if c.args[0] == "deepseek"]
        assert plan_calls, "planner never ran"
        assert plan_calls[0].kwargs["max_tokens_override"] == PLAN_DELTA_MAX_TOKENS


# ── P1: GLM self_contained context ───────────────────────────────────────

class TestGlmSelfContainedContext:
    def test_flag_adds_the_restore_main_guidance(self, router):
        p = router._build_glm_error_analysis_prompt(
            PORT_WITHOUT_MAIN, LINKER_ERRS, 1, [], self_contained=True)
        assert "SELF-CONTAINED" in p
        assert "restore main() from the original" in p
        assert "never 'write a new main()'" in p

    def test_absent_flag_leaves_the_prompt_alone(self, router):
        p = router._build_glm_error_analysis_prompt(
            PORT_WITHOUT_MAIN, LINKER_ERRS, 1, [], self_contained=False)
        assert "SELF-CONTAINED" not in p


# ── P2: budget-aware refine dispatch ─────────────────────────────────────

class TestRefineBudgetGuard:
    @patch.object(ModelRouter, '_call_model')
    def test_refine_is_not_dispatched_without_room_to_finish(self, mock_call, router):
        """A Kimi call started with less than the compile reserve on the clock is
        killed in flight and returns nothing — strictly worse than not calling.

        The iteration-boundary check cannot catch this: it runs BEFORE the compile,
        the GLM analyst and the re-plan spend their share of the same clock. With
        _call_model mocked no real time passes, so the clock is driven explicitly.
        """
        clock = {"t": 0.0}

        def side_effect(model_key, *a, **k):
            clock["t"] += 20.0  # every LLM call burns 20s of the budget
            if model_key == "kimi27":
                return AgentResult(model_key, True,
                                   f"```cpp\n{SELF_CONTAINED_CUDA}\n```", 0.5)
            return AgentResult(model_key, True, "plan", 0.5)
        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.build_dir = Path("/tmp/fake_build")
        verifier._generate_harness.return_value = ("// mock harness", 0, 0)
        verifier._compile.return_value = (False, "t.cpp:1:1: error: boom", Path("/dev/null"))
        verifier._classify_error_origin.return_value = "ported_code"
        verifier._warn_if_legacy_harness.return_value = None

        # 100s budget: plan(20) + code(20) leaves 60 — enough to enter iteration 1.
        # Then analyst(20) + informed re-plan(20) leave 20, and a refine needs
        # MIN_LLM_TIMEOUT_SECONDS on top of the 25s compile reserve. It must not fire.
        with patch("router.time.monotonic", lambda: clock["t"]):
            result = router.route(SELF_CONTAINED_CUDA, [], max_iterations=3,
                                  verifier=verifier, kernel_name="test_refinebudget",
                                  max_seconds=100, fast_path=False)

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        assert any("[budget]" in c and "not enough for a refine" in c
                   for c in result["changes"]), result["changes"]

        # Exactly one coder call: the initial port. The refine was never dispatched.
        glm_calls = [c for c in mock_call.call_args_list if c.args[0] == "glm"]
        assert len(glm_calls) == 1, f"a refine was dispatched anyway: {len(glm_calls)} calls"

    @patch.object(ModelRouter, '_call_model')
    def test_refine_call_receives_a_max_seconds_cap(self, mock_call, router):
        def side_effect(model_key, *a, **k):
            if model_key == "glm":
                return AgentResult(model_key, True,
                                   f"```cpp\n{SELF_CONTAINED_CUDA}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)
        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.build_dir = Path("/tmp/fake_build")
        verifier._generate_harness.return_value = ("// mock harness", 0, 0)
        verifier._compile.return_value = (False, "t.cpp:1:1: error: boom", Path("/dev/null"))
        verifier._classify_error_origin.return_value = "ported_code"
        verifier._warn_if_legacy_harness.return_value = None

        router.route(SELF_CONTAINED_CUDA, [], max_iterations=2, verifier=verifier,
                     kernel_name="test_refinecap", max_seconds=180, fast_path=False)

        glm_calls = [c for c in mock_call.call_args_list if c.args[0] == "glm"]
        assert len(glm_calls) >= 2, "no refine happened"
        refine = glm_calls[1]
        assert refine.kwargs.get("max_seconds") is not None
        assert refine.kwargs["max_seconds"] <= 180 - COMPILE_RESERVE_SECONDS

    @patch.object(ModelRouter, '_call_model')
    def test_unlimited_budget_leaves_the_refine_uncapped(self, mock_call, router):
        def side_effect(model_key, *a, **k):
            if model_key == "glm":
                return AgentResult(model_key, True,
                                   f"```cpp\n{SELF_CONTAINED_CUDA}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)
        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.build_dir = Path("/tmp/fake_build")
        verifier._generate_harness.return_value = ("// mock harness", 0, 0)
        verifier._compile.return_value = (False, "t.cpp:1:1: error: boom", Path("/dev/null"))
        verifier._classify_error_origin.return_value = "ported_code"
        verifier._warn_if_legacy_harness.return_value = None

        router.route(SELF_CONTAINED_CUDA, [], max_iterations=2, verifier=verifier,
                     kernel_name="test_refineunl", max_seconds=0, fast_path=False)

        glm_calls = [c for c in mock_call.call_args_list if c.args[0] == "glm"]
        assert len(glm_calls) >= 2
        assert glm_calls[1].kwargs.get("max_seconds") is None
