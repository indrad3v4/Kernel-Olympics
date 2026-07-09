"""Tests for the 3-minute-demo work (prompts/to-fable-system-roast-and-fix.md).

Covers the seven fixes:
  P0  hard wall-clock timeout          → TestDeadline, TestRouteTimeout
  P0  cache-first check                → TestCacheFirst
  P0  faster stagnation detection      → TestStagnationThresholds
  P1  two-layer SIGSEGV fix            → TestTwoLayerFix
  P1  banner single source of truth    → TestBannerSingleSource
  P2  prompt versioning                → TestPromptVersioning
"""
import io
import os
import sys
import contextlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# main.py auto-loads the repo's .env at import time via os.environ.setdefault().
# Importing it here would otherwise leak a real FIREWORKS_API_KEY into the shared
# pytest process — which makes ModelRouter(api_key="") pick the key up and issue
# live network calls. Snapshot/restore around the import, as test_main.py does.
_env_snapshot = dict(os.environ)
import main
os.environ.clear()
os.environ.update(_env_snapshot)

from router import (
    ModelRouter, AgentResult, Deadline, PipelineTimeoutError,
    PROMPT_VERSION, SYSTEM_PROMPTS, MAX_PIPELINE_SECONDS,
    STAGNATION_ABORT_THRESHOLD, MAX_REPLANS,
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """No test in this file may reach the network."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


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

HIP_OTHER = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) { int tid = threadIdx.x; /* v2 */ }
"""


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


# ── P0: Deadline primitive ───────────────────────────────────────────────

class TestDeadline:
    def test_unlimited_budget_never_expires(self):
        d = Deadline(0)
        assert d.unlimited
        assert not d.expired()
        assert d.remaining() is None

    def test_negative_budget_is_unlimited(self):
        assert Deadline(-1).unlimited

    def test_expired_once_budget_spent(self):
        d = Deadline(100)
        d._t0 -= 101  # pretend 101s elapsed
        assert d.expired()
        assert d.remaining() < 0

    def test_clamp_shrinks_timeout_to_remaining(self):
        d = Deadline(100)
        d._t0 -= 70  # 30s left
        assert d.clamp_timeout(180) == pytest.approx(30, abs=1)

    def test_clamp_never_returns_negative(self):
        d = Deadline(10)
        d._t0 -= 50
        assert d.clamp_timeout(180) == 0.0

    def test_clamp_is_noop_when_unlimited(self):
        assert Deadline(0).clamp_timeout(180) == 180

    def test_clamp_does_not_inflate_a_short_timeout(self):
        """A 30s local call must not grow to the full remaining budget."""
        d = Deadline(300)
        assert d.clamp_timeout(30) == 30

    def test_exhausted_when_remaining_is_below_the_call_floor(self):
        """Budget not yet expired, but too small to fit any LLM call.

        Found end-to-end: with 3s left and a 5s floor, expired() was False, so a
        run that had really timed out was reported as '[kimi27] Code generation
        FAILED' — a model failure — with time still on the clock.
        """
        d = Deadline(100)
        d._t0 -= 98  # 2s left, below the 5s floor
        assert not d.expired()
        assert d.exhausted()

    def test_not_exhausted_with_ample_budget(self):
        assert not Deadline(100).exhausted()

    def test_unlimited_is_never_exhausted(self):
        assert not Deadline(0).exhausted()


class TestCallModelBudget:
    def test_pipeline_timeout_error_is_a_runtime_error(self):
        assert issubclass(PipelineTimeoutError, RuntimeError)

    def test_expired_budget_short_circuits_before_any_request(self, router):
        """The whole point: no HTTP request is issued once the budget is gone."""
        router._deadline = Deadline(10)
        router._deadline._t0 -= 50
        with patch("urllib.request.urlopen") as mock_open:
            res = router._call_model("kimi27", "prompt")
        mock_open.assert_not_called()
        assert res.success is False

    def test_timeout_does_not_fall_through_to_the_local_endpoint(self, router):
        """A PipelineTimeoutError must end the call, not trigger a second doomed
        request against the local vLLM endpoint."""
        router._deadline = Deadline(180)
        router._deadline._t0 -= 178  # 2s left → below the floor, mid-call

        attempts = []

        def fake_urlopen(req, timeout=None):
            attempts.append(req.full_url)
            raise AssertionError("no request should be issued")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = router._call_model("kimi27", "prompt")

        assert res.success is False
        assert attempts == [], f"issued doomed requests: {attempts}"

    def test_kimi_retry_cannot_outlive_the_budget(self, router):
        """kimi27 has timeout=180 and retries at 2x. Under a 180s pipeline budget
        neither attempt may be issued with a timeout exceeding what remains."""
        router._deadline = Deadline(180)
        router._deadline._t0 -= 150  # 30s left
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            raise OSError("boom")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_model("kimi27", "prompt")

        assert seen, "expected at least one request attempt"
        assert all(t <= 31 for t in seen), f"timeout escaped the budget: {seen}"


# ── P0: route() honours the wall-clock budget ────────────────────────────

class TestRouteTimeout:
    def test_default_budget_is_180s(self):
        assert MAX_PIPELINE_SECONDS == 180

    @patch.object(ModelRouter, '_call_model')
    def test_route_aborts_and_returns_best_compiling_code(self, mock_call, router):
        """Budget expires mid-loop → stop, flag it, hand back the code that compiled."""
        mock_call.side_effect = lambda mk, *a, **k: AgentResult(
            mk, True, f"```cpp\n{HIP_OK}\n```" if mk == "kimi27" else '{"pass": true}', 0.5)

        verifier = MagicMock()
        # pre-loop passes, iter1 passes (freezes best attempt), then we starve the budget
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        real_deadline = {}
        orig_init = Deadline.__init__

        def spy_init(self, budget_s):
            orig_init(self, budget_s)
            real_deadline['d'] = self
            self._t0 -= (budget_s + 1)  # already expired on the first check

        with patch.object(Deadline, '__init__', spy_init):
            result = router.route(CUDA_SRC, [], max_iterations=10,
                                  verifier=verifier, kernel_name="test_timeout")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        assert result["iterations_used"] == 0
        assert any("Wall-clock limit" in c for c in result["changes"])

    @patch.object(ModelRouter, '_call_model')
    def test_timeout_hands_back_the_compiling_code_not_the_last_refine(
            self, mock_call, router):
        """The plan's actual P0 requirement: 'abort and return the best-compiling
        code so far'. iter1 compiles but SIGSEGVs (code A); Kimi refines into code
        B; the budget expires before B is ever compiled. The caller must receive
        A — the code known to build — not the unvalidated B.

        Compile-pass + clean-run breaks the loop as converged (see the 'keeping
        working code' branch), so a runtime crash is what drives a refine here.
        """
        code_A = HIP_OK
        code_B = HIP_OTHER

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                # 1st kimi call = initial port (A); 2nd = the refine (B)
                kimi_calls = [c for c in mock_call.call_args_list if c[0][0] == "kimi27"]
                code = code_A if len(kimi_calls) <= 1 else code_B
                return AgentResult(model_key, True, f"```cpp\n{code}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": false, "issues": ["shfl width"]}', 0.5)

        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.quick_compile_check.side_effect = [
            {"compile_success": False, "errors": ["e"], "output": ""},  # pre-loop
            {"compile_success": True, "errors": [], "output": ""},      # iter1 → A compiles
        ]
        # A compiles but crashes → the loop refines into B instead of converging
        verifier.quick_run_check.return_value = {
            "run_success": False, "signal": "SIGSEGV", "run_exit_code": -11, "run_output": ""}

        # Budget survives iteration 1's boundary check, dies before iteration 2's.
        calls = {"n": 0}

        def fake_exhausted(self):
            calls["n"] += 1
            return calls["n"] > 1

        with patch.object(Deadline, 'exhausted', fake_exhausted):
            result = router.route(CUDA_SRC, [], max_iterations=10, verifier=verifier,
                                  kernel_name="test_recover")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        # `ported_code` is post-processed by _fix_ported_code (it injects
        # WAVEFRONT_SIZE), so identify the lineage by B's unique marker rather
        # than by string equality with the raw source.
        assert "/* v2 */" not in result["ported_code"], \
            "returned the unvalidated refine instead of the code that compiled"
        assert result["ported_code"] == result["best_attempt_code"]
        assert result["compile_passed"] is True
        assert result["compile_errors"] == []
        assert any("Returned best compiling code" in c for c in result["changes"])

    @patch.object(ModelRouter, '_call_model')
    def test_max_seconds_zero_disables_the_budget(self, mock_call, router):
        mock_call.side_effect = lambda mk, *a, **k: AgentResult(
            mk, True, f"```cpp\n{HIP_OK}\n```" if mk == "kimi27" else '{"pass": true}', 0.5)
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                              kernel_name="test_nobudget", max_seconds=0)
        assert result["timed_out"] is False
        assert result.get("abort_reason") != "pipeline_timeout"

    def test_no_api_key_result_still_carries_budget_fields(self):
        r = ModelRouter(api_key="")
        assert r.api_key == "", "test would otherwise hit the network"
        result = r.route(CUDA_SRC, [])
        assert result["timed_out"] is False
        assert result["prompt_version"] == PROMPT_VERSION

    @patch.object(ModelRouter, '_call_model')
    def test_timeout_during_initial_codegen_is_reported_as_a_timeout(self, mock_call, router):
        """Regression: the budget can die on the very first Kimi call, before the
        loop. That used to return via the 'code generation FAILED' path with
        timed_out=False — a timeout disguised as a model failure."""
        # DeepSeek plans fine; Kimi returns the failed AgentResult that _call_model
        # produces when the budget is gone.
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, False, "", 0.0)
            return AgentResult(model_key, True, "a plan", 0.5)

        mock_call.side_effect = side_effect

        orig_init = Deadline.__init__

        def spy_init(self, budget_s):
            orig_init(self, budget_s)
            self._t0 -= (budget_s - 1)  # 1s left: not expired, but below the floor

        with patch.object(Deadline, '__init__', spy_init):
            result = router.route(CUDA_SRC, [], max_iterations=5, kernel_name="test_early")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        assert not result["ported_code"]
        assert any("initial code generation" in c for c in result["changes"])
        assert not any("Code generation FAILED" in c for c in result["changes"]), \
            "a budget timeout must not be reported as a model failure"


# ── P0: cache-first check (already present — pinned against regression) ──

class TestCacheFirst:
    def test_retrieve_is_called_before_route_and_short_circuits_it(self, tmp_path):
        """A verified cache hit must skip the LLM pipeline entirely.

        This is the ~0.1s-instead-of-15-min path. It already worked; this test
        exists so a refactor cannot quietly move retrieve() after route().
        """
        ko = main.KernelOlympics.__new__(main.KernelOlympics)  # no __init__ side effects
        ko.memory = MagicMock()
        ko.memory.retrieve.return_value = {
            "id": "pat-1", "verified_fix": HIP_OK, "confidence": 0.95, "llm_time_s": 12.0,
        }
        ko.router = MagicMock()

        cached = ko.memory.retrieve(CUDA_SRC, findings=[])
        assert cached is not None
        ko.router.route.assert_not_called()

    def test_memory_retrieve_refuses_unverified_entries(self, tmp_path):
        """The plan proposed guarding on cached['verified']; retrieve() already does."""
        from pattern_memory.memory import PatternMemory
        mem = PatternMemory(db_path=str(tmp_path / "m.db"))
        findings = [{"pattern": "shfl_sync", "line": 1, "severity": "high"}]
        mem.store(CUDA_SRC, HIP_OK, confidence=0.9, verified=False, findings=findings)
        assert mem.retrieve(CUDA_SRC, findings=findings) is None


# ── P0: stagnation thresholds ────────────────────────────────────────────

class TestStagnationThresholds:
    def test_thresholds_fire_early_enough_for_a_3min_demo(self):
        assert STAGNATION_ABORT_THRESHOLD == 2
        assert MAX_REPLANS == 2

    def test_abort_is_reachable_within_the_iteration_budget(self):
        """Old gate needed replan_count >= max_iterations//2 (=5 at default 10),
        but re-plans were capped well below that, so it took ~6 iterations.
        The new gate must be satisfiable at MAX_REPLANS."""
        assert MAX_REPLANS >= 1
        assert STAGNATION_ABORT_THRESHOLD <= MAX_REPLANS + 1

    @patch.object(ModelRouter, '_call_model')
    def test_one_replan_per_iteration_at_most(self, mock_call, router):
        """A stagnant iteration used to fire two DeepSeek re-plans (~160s)."""
        calls = []

        def side_effect(model_key, *a, **k):
            calls.append(model_key)
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        mock_call.side_effect = side_effect
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["error: undeclared foo"], "output": ""}

        router.route(CUDA_SRC, [], max_iterations=3, verifier=verifier,
                     kernel_name="test_replan", max_seconds=0)

        deepseek_calls = [c for c in calls if c == "deepseek"]
        # 1 initial plan + at most MAX_REPLANS re-plans
        assert len(deepseek_calls) <= 1 + MAX_REPLANS, f"too many DeepSeek calls: {calls}"


# ── P1: two-layer SIGSEGV fix ────────────────────────────────────────────

class TestTwoLayerFix:
    def test_refine_prompt_freezes_the_base_kernel(self, router):
        p = router._build_kimi_refine_prompt(
            CUDA_SRC, HIP_OTHER, "crashed with SIGSEGV", [],
            frozen_base_code=HIP_OK)
        assert "Layer 1" in p
        assert "do NOT rewrite" in p.lower() or "Do NOT rewrite" in p
        assert "DISCARDED" in p
        assert HIP_OK.strip()[:40] in p

    def test_refine_prompt_omits_layer1_when_no_frozen_base(self, router):
        p = router._build_kimi_refine_prompt(
            CUDA_SRC, HIP_OTHER, "compile errors", [], frozen_base_code="")
        assert "Layer 1" not in p

    @patch.object(ModelRouter, '_call_model')
    def test_layer2_that_breaks_the_build_is_discarded(self, mock_call, router):
        """iter1 compiles + crashes → Kimi rewrites → iter2 fails to compile.
        The rewrite must be thrown away and the frozen base returned."""
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                # first call = initial port (compiles), second = the bad rewrite
                code = HIP_OK if mock_call.call_count <= 2 else HIP_OTHER
                return AgentResult(model_key, True, f"```cpp\n{code}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": false, "issues": ["shfl width"]}', 0.5)

        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.quick_compile_check.side_effect = [
            {"compile_success": True, "errors": [], "output": ""},   # pre-loop
            {"compile_success": True, "errors": [], "output": ""},   # iter1 → freeze Layer 1
            {"compile_success": False, "errors": ["error: a", "error: b"],
             "output": ""},                                          # iter2 → Layer 2 broke it
        ]
        # compiles but SIGSEGVs
        verifier.quick_run_check.return_value = {
            "run_success": False, "signal": "SIGSEGV", "run_exit_code": -11, "run_output": ""}

        result = router.route(CUDA_SRC, [], max_iterations=3, verifier=verifier,
                              kernel_name="test_twolayer", max_seconds=0)

        assert result["abort_reason"] == "layer2_rejected"
        assert any("two-layer" in c for c in result["changes"])
        # the returned code is the frozen, compiling base — not the broken rewrite
        assert result["compile_passed"] is True
        assert result["compile_errors"] == []


# ── P1: banner single source of truth ────────────────────────────────────

class TestBannerSingleSource:
    def test_render_banner_exists_and_is_the_only_emitter(self):
        assert callable(main._render_banner)
        src = Path(main.__file__).read_text(encoding="utf-8")
        # The banner literal may appear in BANNER_TEXT and nowhere else as a print()
        assert src.count("╔═ Kernel Olympics ═") == 1, \
            "banner literal duplicated — route it through _render_banner()"

    def test_display_prints_exactly_one_banner(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.Display()
        assert buf.getvalue().count(main.BANNER_TEXT) == 1

    def test_silent_display_prints_no_banner(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.Display(silent=True)
        assert buf.getvalue().count(main.BANNER_TEXT) == 0

    def test_pipeline_construction_emits_a_single_banner(self):
        """T2.1 regression: KernelOlympics built Display twice."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.KernelOlympics()
        assert buf.getvalue().count(main.BANNER_TEXT) == 1

    def test_silent_pipeline_emits_no_banner(self):
        """The duplicate Display() also dropped silent=silent."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.KernelOlympics(silent=True)
        assert buf.getvalue().count(main.BANNER_TEXT) == 0


# ── P2: prompt versioning ────────────────────────────────────────────────

class TestPromptVersioning:
    def test_prompt_version_is_semver_tagged(self):
        assert PROMPT_VERSION.startswith("v")
        major, minor, patch_ = PROMPT_VERSION[1:].split(".")
        assert all(p.isdigit() for p in (major, minor, patch_))

    def test_every_system_prompt_carries_the_version(self):
        assert SYSTEM_PROMPTS, "no system prompts defined"
        for name, text in SYSTEM_PROMPTS.items():
            assert f"[prompt {PROMPT_VERSION}]" in text, \
                f"system prompt {name!r} is missing its version tag"

    def test_route_reports_the_prompt_version(self):
        r = ModelRouter(api_key="")
        assert r.api_key == "", "test would otherwise hit the network"
        assert r.route(CUDA_SRC, [])["prompt_version"] == PROMPT_VERSION

    def test_changelog_json_matches_the_code_version(self):
        repo = Path(__file__).resolve().parents[1]
        data = json.loads((repo / "data" / "prompt_changelog.json").read_text(encoding="utf-8"))
        assert data["current"] == PROMPT_VERSION
        versions = [v["version"] for v in data["versions"]]
        assert PROMPT_VERSION in versions

    def test_changelog_md_documents_the_current_version(self):
        repo = Path(__file__).resolve().parents[1]
        md = (repo / "prompts" / "CHANGELOG.md").read_text(encoding="utf-8")
        assert PROMPT_VERSION in md
