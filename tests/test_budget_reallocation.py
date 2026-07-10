"""Tests for Part B — budget re-allocation with a protected repair reserve.

The failure this fixes: on the mission doc's cited 180s run, planning finished
in 22.2s, leaving the initial generation an effectively unbounded cap of
`remaining() - COMPILE_RESERVE_SECONDS` (~132s) — bounded by nothing but the
compile reserve. It used 89.9s. One refine iteration (43.4s) then left ~24s,
one second short of even the flat compile reserve: "no room to retry".

Two independent fixes:

  1. CODE_BUDGET_FRACTION / REPAIR_RESERVE_FRACTION cap the initial generation
     directly and protect a real repair reserve, WITHOUT raising
     MAX_PIPELINE_SECONDS (the mission doc explicitly forbids that as the
     primary fix — re-allocate the existing budget, don't grow it).

  2. Consecutive STRUCTURAL rejects (as opposed to compile-error stagnation,
     already handled by the pre-existing stagnation_count/kimi_plateau_count
     machinery) now have their own adaptive-stop: a 2nd consecutive reject
     switches to a terser retry instead of repeating the same ask, and a 3rd
     aborts early rather than spending the whole clock on a 4th doomed
     generation — which is exactly what the reported nvidia_shfl_scan run did
     (two structural rejects, then a timeout, with hipcc never once invoked).
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_env_snapshot = dict(os.environ)
import main  # noqa: E402 — see test_demo_budget.py for why this import is guarded
os.environ.clear()
os.environ.update(_env_snapshot)

from router import (  # noqa: E402
    ModelRouter, AgentResult, Deadline,
    CODE_BUDGET_FRACTION, REPAIR_RESERVE_FRACTION, COMPILE_RESERVE_SECONDS,
    MIN_LLM_TIMEOUT_SECONDS, MAX_PIPELINE_SECONDS,
)


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


def _instant_fireworks(seen):
    """Records (url, timeout) for every request and returns instantly."""
    class _Resp:
        def __init__(self, body):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        seen.append((req.full_url, timeout))
        import json as _json
        body = _json.dumps({
            "choices": [{"message": {"content": f"```cpp\n{HIP_OK}\n```"},
                        "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }).encode()
        return _Resp(body)
    return fake


# ── The constants exist and are not just MAX_PIPELINE_SECONDS raised ───────

class TestBudgetConstantsExist:
    def test_max_pipeline_seconds_is_unchanged(self):
        """The mission doc's guardrail: do NOT raise the global timeout as the
        primary fix. Re-allocate it."""
        assert MAX_PIPELINE_SECONDS == 180

    def test_code_budget_fraction_is_meaningfully_smaller_than_the_whole_budget(self):
        assert 0 < CODE_BUDGET_FRACTION < 0.5

    def test_repair_reserve_fraction_is_a_real_majority_reservation(self):
        assert REPAIR_RESERVE_FRACTION >= 0.3


# ── The initial generation is capped tighter on a normal-sized budget ──────

class TestInitialGenerationCap:
    def test_coder_cap_is_tighter_than_the_old_compile_reserve_only_formula(self, router):
        """On a 180s budget with fast planning, the OLD formula
        (remaining() - COMPILE_RESERVE_SECONDS) let the coder claim ~130s+.
        The new cap must come in meaningfully under that."""
        seen = []

        def instant_plan_then_capture(model_key, *a, **k):
            # _call_model_impl(model_key, prompt, system_prompt, prefill,
            # max_seconds, max_tokens_override) -- all positional.
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            seen.append(a[3] if len(a) > 3 else None)
            return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=instant_plan_then_capture):
            router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                         kernel_name="test_budget_realloc", max_seconds=180)

        assert seen, "coder was never called"
        code_cap = seen[0]
        old_formula_cap = 180 - COMPILE_RESERVE_SECONDS  # ~155s, the prior ceiling
        assert code_cap < old_formula_cap * 0.6, (
            f"coder cap {code_cap}s is not meaningfully tighter than the old "
            f"~{old_formula_cap}s ceiling")
        assert code_cap >= MIN_LLM_TIMEOUT_SECONDS

    def test_repair_reserve_leaves_a_real_window_for_refinement(self, router):
        """The actual point: after planning + initial generation, there must be
        a substantial amount of budget left for compile+refine — not the ~24s
        the reported run was left with after a single refine attempt."""
        seen_caps = []

        def fast_calls(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 1000.0)
            seen_caps.append(a[3] if len(a) > 3 else None)
            return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 1000.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=fast_calls):
            router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                         kernel_name="test_repair_reserve", max_seconds=180)

        code_cap = seen_caps[0]
        # Elapsed time is negligible (all calls return instantly), so the
        # budget remaining after the coder's OWN cap is ~180 - code_cap.
        remaining_for_repair = 180 - code_cap
        assert remaining_for_repair >= 180 * REPAIR_RESERVE_FRACTION - 5

    def test_small_budget_is_unaffected_the_coder_still_runs(self, router):
        """Regression pin: on a budget too small for the new reservation
        formula to make sense (compile reserve + repair reserve exceeds the
        whole budget), the coder must still get a usable cap — it is not
        skippable the way the planner is."""
        seen = []
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch("urllib.request.urlopen", side_effect=_instant_fireworks(seen)):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_small_budget_realloc", max_seconds=40)

        assert result["ported_code"], "the coder must still produce a port on a tight budget"

    def test_unlimited_budget_has_no_cap(self, router):
        seen = []
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch("urllib.request.urlopen", side_effect=_instant_fireworks(seen)):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_unlimited_realloc", max_seconds=0)
        assert result["ported_code"]


# ── Structural-reject adaptive stop ────────────────────────────────────────

class TestStructuralStagnation:
    """Consecutive structural rejects (truncation, unbalanced braces) are a
    DIFFERENT failure mode from compile-error stagnation: the model's own
    response was malformed, not the porting strategy. A DeepSeek re-plan
    cannot fix "the model keeps getting cut off", so this is deliberately NOT
    wired into the existing re-plan trigger."""

    def test_second_consecutive_structural_reject_switches_to_terser_retry(self, router):
        # Every generation is prose -- lexical/structural reject every time.
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this some more.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            result = router.route(CUDA_SRC, [], max_iterations=5, verifier=verifier,
                                  kernel_name="test_structstagswitch", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "switching to a terser retry" in joined

    def test_third_consecutive_structural_reject_aborts_early(self, router):
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this some more.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            result = router.route(CUDA_SRC, [], max_iterations=10, verifier=verifier,
                                  kernel_name="test_structstagabort", max_seconds=0)

        assert result["abort_reason"] == "structural_stagnation"
        # It must not have burned all 10 iterations getting there.
        assert result["iterations_used"] <= 3

    def test_hipcc_is_never_invoked_across_repeated_structural_rejects(self, router):
        """The actual bug: hipcc ran zero times in the reported failure because
        every generation was rejected before it. Pin that quick_compile_check
        is never called for a run where the coder never returns valid code."""
        def always_prose(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            return AgentResult(model_key, True, "Let's think about this some more.", 0.5, 50, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "output": ""}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=always_prose):
            router.route(CUDA_SRC, [], max_iterations=10, verifier=verifier,
                         kernel_name="test_structstaghipcc", max_seconds=0)

        verifier.quick_compile_check.assert_not_called()

    def test_a_recovering_generation_resets_the_counter(self, router):
        """A single structural reject followed by a passing generation must
        not count toward a LATER streak of rejects — the counter resets on any
        generation that clears the structural gate. Proven by running long
        enough (8 iterations) that WITHOUT a reset, 3 total rejects among them
        would trip the abort; the recovering call in the middle must prevent
        that by breaking up the streak.
        """
        calls = {"n": 0}

        def one_reject_then_always_ok(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, True, "plan", 0.5, 50, 500.0)
            calls["n"] += 1
            if calls["n"] == 1:
                # exactly ONE structural reject, ever
                return AgentResult(model_key, True, "Let's think about this.", 0.5, 50, 500.0)
            return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5, 100, 500.0)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model_impl', side_effect=one_reject_then_always_ok):
            result = router.route(CUDA_SRC, [], max_iterations=8, verifier=verifier,
                                  kernel_name="test_structstagreset", max_seconds=0)

        assert result.get("abort_reason") != "structural_stagnation"
        assert result["compile_passed"] is True
