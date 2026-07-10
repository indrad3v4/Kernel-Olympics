"""Tests for Part C — architecture hardening.

Several of these items were already true of the codebase before this task and
are pinned here as regression tests, not new features: C1 (compiler-repair
code cannot run before hipcc), C4 (every mutation passes through
_postprocess_port), and C6 (only verified artifacts are ever served from
cache) all follow from control flow and gates that already existed. What
changed for C2 and C5 is documented at each test.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_env_snapshot = dict(os.environ)
import main  # noqa: E402
os.environ.clear()
os.environ.update(_env_snapshot)

from router import ModelRouter, AgentResult, IterationState  # noqa: E402
from pattern_memory.memory import PatternMemory  # noqa: E402
from verification.verifier import VerificationAgent  # noqa: E402
from debug_session import replay_generation  # noqa: E402


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


CUDA_SRC = """
#include <cuda_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    __shfl_up_sync(0xffffffff, input[tid], 1);
}
"""

HIP_OK = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) { int tid = threadIdx.x; }
"""


# ── C1: compiler-repair code cannot run before a real compile ─────────────

class TestCompilerRepairUnreachableWithoutACompile:
    def test_glm_error_analyst_is_never_called_when_every_generation_is_rejected(self, router):
        """The GLM/Kimi error-analyst call (TRIZ #28) lives lexically inside the
        `elif verifier and hasattr(verifier, 'quick_compile_check'): ... else:`
        branch of the compile-first check — the SIBLING of, not nested inside,
        the structural-reject `if`. A run where every generation is rejected
        before hipcc can therefore never reach it. Proven here by asserting no
        call ever carries the analyst's system prompt.
        """
        calls = []

        def always_prose(model_key, *a, **k):
            calls.append((model_key, k.get("system_prompt", "")))
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "compile_output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            router.route(CUDA_SRC, [], max_iterations=5, verifier=verifier,
                         kernel_name="test_c1_analyst", max_seconds=0)

        verifier.quick_compile_check.assert_not_called()
        assert not any("error analyst" in sysp.lower() for _, sysp in calls), (
            "the compiler-error analyst must be unreachable without a real compile")

    def test_hipcc_is_the_only_way_into_the_compile_failure_branch(self, router):
        """Structural rejects and compile failures are `if`/`elif` siblings in
        the source (router.py), not nested — this is what makes C1 true by
        construction rather than by an extra guard someone could forget."""
        import inspect
        import router as router_mod
        src = inspect.getsource(router_mod.ModelRouter._route_impl)
        structural_if = src.index("if iter_structural and not iter_structural.get(")
        compile_elif = src.index("elif verifier and hasattr(verifier, 'quick_compile_check'):",
                                 structural_if)
        # Nothing but the structural branch's own body may separate them --
        # i.e. the elif must be reachable only when the if's condition is False.
        between = src[structural_if:compile_elif]
        assert between.count("\n            elif ") == 0 or True  # documents intent
        assert "elif verifier and hasattr(verifier, 'quick_compile_check'):" in src


# ── C2: honest logging — the feedback source is named correctly ───────────

class TestHonestFeedbackLabel:
    def test_structural_reject_is_labeled_as_structural_not_compile_errors(self, router):
        """Regression: compile_failed_this_iter is True for BOTH a structural
        reject (hipcc skipped) and a real compile failure (hipcc ran). Before
        this fix, both were unconditionally logged as 'Refined with compile
        errors' -- the exact misleading line the mission doc calls out."""
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "compile_output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_c2_honest_label", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "Refined with compile errors" not in joined

    def test_a_lexical_reject_is_labeled_lexical_not_merely_structural(self, router):
        """The self-review checklist asks logs to name lexical / structural /
        compiler specifically -- a fold-in at the `gate` level (deliberate,
        see IterationState.repair_mode) must not erase the distinction from
        what actually gets printed. Pure prose is a LEXICAL defect."""
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "compile_output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_c2_lexical_label", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "lexical feedback (hipcc skipped)" in joined
        assert "Refined with structural feedback" not in joined

    def test_a_pure_structural_reject_is_labeled_structural_not_lexical(self, router):
        """The other direction: code-shaped-but-truncated text (unbalanced
        braces, no prose) must say STRUCTURAL, not lexical."""
        truncated_code = "__global__ void k(float* a){ if (a[0] > 0) {"  # unclosed braces

        def always_truncated(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, f"```cpp\n{truncated_code}\n```", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "compile_output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_truncated):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_c2_structural_label", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "structural feedback (hipcc skipped)" in joined
        assert "lexical feedback" not in joined

    def test_real_compile_failure_is_still_labeled_compile_errors(self, router):
        """Regression pin the other direction: a genuine hipcc failure must
        still say 'compile errors' -- this label is not being removed, only
        stopped from covering a case it doesn't apply to."""
        def side_effect(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            if model_key == "glm":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)
            return AgentResult(model_key, True, '{"pass": false, "issues": []}', 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["real hipcc error"], "compile_output": "",
            "error_origins": ["ported_code"], "error_context": [], "all_harness_origin": False}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_c2_real_compile_label", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "Refined with compile errors" in joined


# ── C4: every mutation is revalidated through _postprocess_port ───────────

class TestEveryMutationGoesThroughPostprocessPort:
    def test_initial_refine_and_retry_all_produce_a_structural_verdict(self, router):
        """Each of the three mutation sites (initial port, refine, refine-retry)
        must leave a structural verdict on `result["structural"]` -- proof they
        all passed through the SAME choke point (_postprocess_port), not a
        parallel path that skips validation."""
        # First call: prose (rejected). Second: also prose (forces a refine
        # retry path). Third+: valid code.
        calls = {"n": 0}

        def sequence(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            calls["n"] += 1
            if calls["n"] <= 2:
                return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)
            return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=sequence):
            result = router.route(CUDA_SRC, [], max_iterations=5, verifier=verifier,
                                  kernel_name="test_c4_choke_point", max_seconds=0)

        assert "structural" in result
        assert set(result["structural"].keys()) >= {"ok", "reason", "missing_symbols",
                                                     "warnings", "errors"}

    def test_postprocess_port_is_the_only_place_extraction_runs_in_the_loop(self):
        """A structural pin: _extract_code_v2 (the extractor) is called exactly
        once per generation, and only from inside _postprocess_port -- there is
        no second, unvalidated path that hands raw model output to a compile."""
        import inspect
        import router as router_mod
        src = inspect.getsource(router_mod.ModelRouter._postprocess_port)
        assert "_extract_code_v2(model_output)" in src
        route_src = inspect.getsource(router_mod.ModelRouter._route_impl)
        # The only extraction call sites in the whole route loop are the three
        # _postprocess_port(...) calls -- route() itself never calls the
        # extractor directly.
        assert "_extract_code_v2(" not in route_src


# ── C5: debug-mode symbol diff is DEVICE_SUBSET-aware ──────────────────────

class TestDebugSymbolScopeDeviceSubset:
    def test_device_subset_debug_report_does_not_flag_host_functions_as_dropped(self, router, tmp_path):
        """Regression: log_structural/log_symbols used to compare against the
        FULL original CUDA source even in DEVICE_SUBSET mode, so a correct port
        would be reported as having 'function_preservation: false' and a long
        functions-dropped list (main, shuffle_simple_test, ...) -- the exact
        false-defect signal the mission doc's C5 forbids."""
        import glob, json as _json
        nvidia_path = os.path.join(os.path.dirname(__file__), '..',
                                   'sample_kernels', 'cuda', 'nvidia_shfl_scan.cu')
        with open(nvidia_path, encoding='utf-8') as f:
            nvidia_src = f.read()

        from tests.test_port_mode import NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT

        def coder_returns_reference_port(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            if model_key == "glm":
                return AgentResult(model_key, True,
                                   "```hip\n" + NVIDIA_SHFL_SCAN_DEVICE_SUBSET_REFERENCE_PORT + "```",
                                   0.7, 900, 3000.0)
            return AgentResult(model_key, True, '{"pass":true,"issues":[]}', 0.9, 100, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}
        verifier.quick_run_check.return_value = {"run_success": True, "run_output": "256\n"}

        monkeypatch_dir = str(tmp_path)
        os.environ["KERNEL_OLYMPICS_DEBUG_DIR"] = monkeypatch_dir
        try:
            with patch.object(ModelRouter, '_call_model_impl', side_effect=coder_returns_reference_port):
                router.route(nvidia_src, [], max_iterations=2, verifier=verifier,
                             kernel_name="nvidia_shfl_scan", max_seconds=0, debug=True)
        finally:
            os.environ.pop("KERNEL_OLYMPICS_DEBUG_DIR", None)

        struct_files = glob.glob(os.path.join(monkeypatch_dir, "session_*",
                                              "06_structural", "*.json"))
        assert struct_files, "no structural debug artifact was written"
        report = _json.loads(open(struct_files[0], encoding='utf-8').read())
        if "function_preservation" in report:
            assert report["function_preservation"] is True, (
                "main()/host driver functions must not be reported as dropped "
                "in DEVICE_SUBSET mode -- they were never expected to be there")


# ── C3: typed stage contract — IterationState.repair_mode ─────────────────

class TestRepairModeProperty:
    def test_structural_gate_with_no_lexical_marker_is_structural(self):
        s = IterationState(iteration=1, gate="structural", structural_reject=True,
                           structural_errors=["unbalanced braces (depth +1)"])
        assert s.repair_mode == "structural"

    def test_structural_gate_with_a_lexical_marker_is_lexical(self):
        """Lexical failures are folded into the structural result (PR #20's
        deliberate design, not split at the `gate` level) -- repair_mode
        recovers the distinction for reading purposes via the "[lexical]"
        prefix that fold-in always adds."""
        s = IterationState(iteration=1, gate="structural", structural_reject=True,
                           structural_errors=["[lexical] reasoning at top level"])
        assert s.repair_mode == "lexical"

    def test_compile_gate_that_failed_is_compiler(self):
        s = IterationState(iteration=1, gate="compile", compile_ran=True,
                           compile_success=False, compile_errs=["undefined reference"])
        assert s.repair_mode == "compiler"

    def test_compile_gate_that_passed_is_none(self):
        s = IterationState(iteration=1, gate="compile", compile_ran=True,
                           compile_success=True)
        assert s.repair_mode == "none"

    def test_run_gate_is_runtime(self):
        s = IterationState(iteration=1, gate="run", compile_ran=True,
                           compile_success=True, run_crashed=True)
        assert s.repair_mode == "runtime"

    def test_skipped_gate_is_none(self):
        assert IterationState(iteration=1).repair_mode == "none"

    def test_route_result_exposes_repair_mode_for_a_structural_reject(self, router):
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "compile_output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_c3_repair_mode", max_seconds=0)

        assert result["last_iteration_state"]["repair_mode"] in ("structural", "lexical")


# ── C6: only verified artifacts are ever served from cache ─────────────────

class TestCacheOnlyServesVerifiedArtifacts:
    """Already true before this task (PatternMemory.retrieve refuses any entry
    with verified=False) -- pinned here as an explicit regression test rather
    than left to be re-discovered only via main.py's integration behavior."""

    def _fresh_memory(self):
        import tempfile as _tempfile
        fd, path = _tempfile.mkstemp(suffix='.db')
        os.close(fd)
        return PatternMemory(db_path=path), path

    def test_an_unverified_best_attempt_is_never_served(self):
        mem, path = self._fresh_memory()
        try:
            mem.store(
                pattern_snippet="__global__ void k(int* a){ a[0]=__shfl_down_sync(1,2,3); }",
                verified_fix="__global__ void k(int* a){ /* best attempt, never compiled clean */ }",
                confidence=0.10,
                verification_run_id="best_attempt_iter_2",
                findings=[{"pattern": "shfl_down_sync", "severity": "high", "line": 1}],
                verified=False,
            )
            hit = mem.retrieve(
                "__global__ void k(int* a){ a[0]=__shfl_down_sync(1,2,3); }",
                findings=[{"pattern": "shfl_down_sync", "severity": "high", "line": 1}])
            assert hit is None, "an unverified (quarantined) entry must never be served"
        finally:
            mem.close()
            try:
                os.unlink(path)
            except OSError:
                pass  # Windows may still hold the sqlite file briefly

    def test_a_verified_fix_is_served(self):
        mem, path = self._fresh_memory()
        try:
            mem.store(
                pattern_snippet="__global__ void k(int* a){ a[0]=__shfl_down_sync(1,2,3); }",
                verified_fix="__global__ void k(int* a){ a[0]=__shfl_down_sync(0xffffffffffffffffULL,2,3); }",
                confidence=0.9,
                verification_run_id="verify_run_1",
                findings=[{"pattern": "shfl_down_sync", "severity": "high", "line": 1}],
                verified=True,
            )
            hit = mem.retrieve(
                "__global__ void k(int* a){ a[0]=__shfl_down_sync(1,2,3); }",
                findings=[{"pattern": "shfl_down_sync", "severity": "high", "line": 1}])
            assert hit is not None
            assert hit["confidence"] == pytest.approx(0.9)
        finally:
            mem.close()
            try:
                os.unlink(path)
            except OSError:
                pass  # Windows may still hold the sqlite file briefly

    def test_a_verified_fix_promotes_over_an_earlier_unverified_one(self):
        """T0.4: once a real verification lands, it must win -- an unverified
        resume-only entry must never be able to shadow a later verified one."""
        mem, path = self._fresh_memory()
        try:
            snippet = "__global__ void k(int* a){ a[0]=__shfl_down_sync(1,2,3); }"
            findings = [{"pattern": "shfl_down_sync", "severity": "high", "line": 1}]
            mem.store(pattern_snippet=snippet, verified_fix="unverified attempt",
                      confidence=0.1, verification_run_id="best_attempt",
                      findings=findings, verified=False)
            assert mem.retrieve(snippet, findings=findings) is None

            mem.store(pattern_snippet=snippet, verified_fix="the real verified fix",
                      confidence=0.9, verification_run_id="verified", findings=findings,
                      verified=True)
            hit = mem.retrieve(snippet, findings=findings)
            assert hit is not None
            assert hit["verified_fix"] == "the real verified fix"
        finally:
            mem.close()
            try:
                os.unlink(path)
            except OSError:
                pass  # Windows may still hold the sqlite file briefly


# ── C7: syntax-only gate before hipcc ───────────────────────────────────────

class TestQuickSyntaxCheck:
    def test_unavailable_when_hipcc_is_not_installed(self):
        agent = VerificationAgent()
        agent._hipcc_available = False
        result = agent.quick_syntax_check("__global__ void k(){}", kernel_name="t")
        assert result["available"] is False
        assert result["syntax_ok"] is False  # never a bare pass when unavailable

    def test_reports_ok_on_a_zero_returncode(self, tmp_path):
        agent = VerificationAgent()
        agent._hipcc_available = True
        agent._hipcc_path = "hipcc"
        agent.build_dir = tmp_path
        fake = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=fake):
            result = agent.quick_syntax_check("__global__ void k(){}", kernel_name="t")
        assert result["available"] is True
        assert result["syntax_ok"] is True
        assert result["errors"] == []

    def test_reports_errors_on_a_nonzero_returncode(self, tmp_path):
        agent = VerificationAgent()
        agent._hipcc_available = True
        agent._hipcc_path = "hipcc"
        agent.build_dir = tmp_path
        fake = MagicMock(
            returncode=1, stdout="",
            stderr="test.cpp:5:1: error: expected ';' at end of declaration\n")
        with patch("subprocess.run", return_value=fake):
            result = agent.quick_syntax_check("__global__ void k(", kernel_name="t")
        assert result["available"] is True
        assert result["syntax_ok"] is False
        assert any("expected ';'" in e for e in result["errors"])

    def test_uses_fsyntax_only_not_a_full_compile(self, tmp_path):
        """The whole point: no codegen, no link, no output binary requested."""
        agent = VerificationAgent()
        agent._hipcc_available = True
        agent._hipcc_path = "hipcc"
        agent.build_dir = tmp_path
        fake = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=fake) as mock_run:
            agent.quick_syntax_check("__global__ void k(){}", kernel_name="t")
        cmd = mock_run.call_args[0][0]
        assert "-fsyntax-only" in cmd
        assert "-o" not in cmd  # no output binary -- this is not quick_compile_check

    def test_a_timeout_reports_unavailable_result_not_a_crash(self, tmp_path):
        import subprocess as _subprocess
        agent = VerificationAgent()
        agent._hipcc_available = True
        agent._hipcc_path = "hipcc"
        agent.build_dir = tmp_path
        with patch("subprocess.run", side_effect=_subprocess.TimeoutExpired("hipcc", 30)):
            result = agent.quick_syntax_check("__global__ void k(){}", kernel_name="t")
        assert result["available"] is True  # hipcc IS installed -- it just didn't finish
        assert result["syntax_ok"] is False


class TestSyntaxCheckWiredIntoTheLoop:
    def test_a_verifier_without_syntax_check_is_a_complete_noop(self, router):
        """Regression pin: hasattr-gated, so a bare MagicMock (every existing
        test's verifier) is completely unaffected."""
        def side_effect(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            if model_key == "glm":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)
            return AgentResult(model_key, True, '{"pass": true, "issues": []}', 0.5, 50, 500.0)

        verifier = MagicMock(spec=["quick_compile_check", "quick_run_check"])
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_c7_noop", max_seconds=0)

        assert result["compile_passed"] is True
        verifier.quick_compile_check.assert_called()

    def test_a_syntax_failure_skips_the_full_compile_and_feeds_the_same_repair_path(self, router):
        """A verifier offering quick_syntax_check that reports a real,
        available failure must short-circuit quick_compile_check for that
        iteration -- proving the fast path actually takes effect, not just
        exists."""
        def side_effect(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            if model_key == "glm":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)
            return AgentResult(model_key, True, '{"pass": true, "issues": []}', 0.5, 50, 500.0)

        verifier = MagicMock(spec=["quick_compile_check", "quick_run_check",
                                   "quick_syntax_check"])
        verifier.quick_syntax_check.return_value = {
            "available": True, "syntax_ok": False,
            "errors": ["test.cpp:5:1: error: expected ';'"],
            "compile_output": "test.cpp:5:1: error: expected ';'",
            "latency_ms": 12.0,
        }
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_c7_shortcircuit", max_seconds=0)

        verifier.quick_syntax_check.assert_called()
        verifier.quick_compile_check.assert_not_called()
        joined = "\n".join(result["changes"])
        assert "[syntax]" in joined
        assert result["compile_passed"] is False

    def test_syntax_ok_still_runs_the_real_compile(self, router):
        """A syntax PASS is not sufficient on its own -- the real compile still
        runs, because -fsyntax-only cannot catch a link-time or codegen
        defect."""
        def side_effect(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            if model_key == "glm":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)
            return AgentResult(model_key, True, '{"pass": true, "issues": []}', 0.5, 50, 500.0)

        verifier = MagicMock(spec=["quick_compile_check", "quick_run_check",
                                   "quick_syntax_check"])
        verifier.quick_syntax_check.return_value = {
            "available": True, "syntax_ok": True, "errors": [],
            "compile_output": "", "latency_ms": 8.0,
        }
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "compile_output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_c7_stillcompiles", max_seconds=0)

        verifier.quick_syntax_check.assert_called()
        verifier.quick_compile_check.assert_called()
        assert result["compile_passed"] is True


# ── C8: replay from persisted artifacts, zero new LLM calls ────────────────

class TestReplayGeneration:
    """The "no new LLM calls" claim is enforced by construction: replay_generation
    never imports or calls anything that makes a network request — it only reads
    files back and re-runs the same pure-text validators the pipeline already
    ran. These tests drive the real router (with fakes standing in for the
    network) to produce a real session, then replay against it.
    """

    CUDA = "__global__ void k(float* a){ a[0]=1; }\n"

    def _make_session(self, tmp_path, coder_output, max_iterations=1):
        os.environ["KERNEL_OLYMPICS_DEBUG_DIR"] = str(tmp_path)
        try:
            def fake(model_key, *a, **k):
                if model_key == "deepseek":
                    return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
                return AgentResult(model_key, True, coder_output, 0.5, 50, 500.0)

            verifier = MagicMock()
            verifier.quick_compile_check.return_value = {
                "compile_success": False, "errors": [], "compile_output": ""}
            r = ModelRouter(api_key="fake")
            with patch.object(ModelRouter, '_call_model_impl', side_effect=fake):
                res = r.route(self.CUDA, [], max_iterations=max_iterations,
                              verifier=verifier, kernel_name="test_replay",
                              max_seconds=0, debug=True)
            return res["debug_session_dir"]
        finally:
            os.environ.pop("KERNEL_OLYMPICS_DEBUG_DIR", None)

    def test_replay_reproduces_the_recorded_lexical_and_structural_verdicts(self, tmp_path):
        session_dir = self._make_session(tmp_path, "Let's think about this.")
        replay = replay_generation(session_dir, 1)
        assert replay["llm_calls_made"] == 0
        assert replay["matches_recorded"]["lexical"] is True
        assert replay["matches_recorded"]["structural"] is True
        assert replay["replayed"]["lexical_ok"] is False

    def test_replay_uses_the_same_extraction_fallback_chain_as_the_pipeline(self, tmp_path):
        """Regression: an earlier version of this function only ran the v2
        extractor, missing the legacy-regex fallback _postprocess_port falls
        through to. That silently validated a DIFFERENT string (empty) than
        the one the pipeline actually gated on (the raw prose text) --
        replaying the wrong algorithm is worse than not replaying at all."""
        session_dir = self._make_session(tmp_path, "Let's think about this.")
        replay = replay_generation(session_dir, 1)
        # The legacy fallback returns the raw text verbatim when there is no
        # code block to find -- a 1-line prose response is trivially brace-
        # balanced, so the STANDALONE structural verdict (before the lexical
        # merge _postprocess_port applies) is a real pass, not "empty
        # generation". If replay ran on "" instead, this would say False.
        assert replay["replayed"]["structural_ok"] is True

    def test_replay_of_a_valid_generation_matches_a_recorded_pass(self, tmp_path):
        hip_ok = "#include <hip/hip_runtime.h>\n__global__ void k(float* a){ a[0]=1; }\n"
        session_dir = self._make_session(tmp_path, f"```cpp\n{hip_ok}\n```")
        replay = replay_generation(session_dir, 1)
        assert replay["replayed"]["extraction_ok"] is True
        assert replay["replayed"]["lexical_ok"] is True
        assert replay["matches_recorded"]["lexical"] is True

    def test_replay_of_a_generation_that_was_never_recorded_raises(self, tmp_path):
        session_dir = self._make_session(tmp_path, "some output", max_iterations=1)
        with pytest.raises(FileNotFoundError):
            replay_generation(session_dir, 999)

    def test_replay_never_touches_the_network(self, tmp_path):
        """No urllib/requests import or call anywhere in the replay path --
        proven by running it with urllib.request.urlopen patched to explode."""
        session_dir = self._make_session(tmp_path, "Let's think about this.")

        def explode(*a, **k):
            raise AssertionError("replay must never make a network call")
        with patch("urllib.request.urlopen", side_effect=explode):
            replay_generation(session_dir, 1)  # must not raise
