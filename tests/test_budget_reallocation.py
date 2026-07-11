"""Part B: budget re-allocation and adaptive-stop tests.

The 2026-07-10 failing trace burned 156s of a 180s budget in Plan (22s) +
Codegen (90s) + Refine (43s), then hit "24s remain — no room to retry"
without ever completing a compile → patch → recompile cycle. The two
fixes are:

1. Cap initial codegen tighter AND reserve a protected repair slice the
   initial codegen may not consume — so the first refine has real budget.
2. Fingerprint iteration failures; when the same signature repeats twice,
   switch strategy (force patch mode) instead of regenerating from
   scratch. A rewrite from the same feedback yields the same output.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from router import (
    ModelRouter,
    IterationState,
    CODE_RESERVE_FRACTION,
    COMPILE_RESERVE_SECONDS,
    REPAIR_RESERVE_FRACTION,
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


class TestBudgetConstants:
    def test_code_reserve_shrunk_from_pre_part_b(self):
        """Pre-Part B was 0.55; Part B tightens to 0.35 to leave room for
        the repair reserve. The exact value is a knob, but the invariant
        is that code + repair + compile reserves fit inside the budget
        with headroom for planner + verify."""
        assert CODE_RESERVE_FRACTION == 0.35

    def test_repair_reserve_is_a_meaningful_fraction(self):
        """A repair reserve of zero would be the pre-Part B behavior. A
        repair reserve above ~40% would starve the initial coder. The
        chosen 0.30 gives ~54s on a 180s budget — long enough for one
        real refine + compile."""
        assert 0.20 <= REPAIR_RESERVE_FRACTION <= 0.40

    def test_code_and_repair_and_compile_fit_in_the_budget(self):
        """The three protected slices, taken together, must leave enough
        for the planner (~20%) + verify (~10%) + slack. Otherwise the
        first codegen sees a negative cap and the loop cannot proceed."""
        budget = 180.0
        used = (budget * CODE_RESERVE_FRACTION
                + budget * REPAIR_RESERVE_FRACTION
                + COMPILE_RESERVE_SECONDS)
        assert used < budget * 0.85, (
            f"protected reserves ({used:.0f}s of {budget}s) leave no "
            f"headroom for planner + verify")


class TestFailureSignature:
    """Fingerprint format need only be stable and same-in / same-out."""

    def _mk(self, iteration=1, **overrides):
        s = IterationState(iteration=iteration)
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_no_failure_returns_empty_string(self):
        assert self._mk(compile_ran=True, compile_success=True).failure_signature() == ""

    def test_structural_reject_signature_is_stable(self):
        errs = ["unbalanced braces (depth +1)"]
        a = self._mk(structural_reject=True, structural_errors=errs)
        b = self._mk(structural_reject=True, structural_errors=list(errs))
        assert a.failure_signature() == b.failure_signature()
        assert a.failure_signature().startswith("structural:")

    def test_different_structural_errors_get_different_signatures(self):
        a = self._mk(structural_reject=True,
                     structural_errors=["unbalanced braces (depth +1)"])
        b = self._mk(structural_reject=True,
                     structural_errors=["duplicate definitions: k"])
        assert a.failure_signature() != b.failure_signature()

    def test_compile_signature_ignores_file_line_col(self):
        """Two runs of the same defect at different line numbers should
        still fingerprint alike — otherwise the adaptive-stop guard would
        never trip on real hipcc output."""
        a = self._mk(compile_ran=True, compile_success=False,
                     compile_errs=["test.cpp:242:12: error: no matching function"])
        b = self._mk(compile_ran=True, compile_success=False,
                     compile_errs=["test.cpp:99:5: error: no matching function"])
        assert a.failure_signature() == b.failure_signature()

    def test_structural_and_compile_never_collide(self):
        a = self._mk(structural_reject=True, structural_errors=["x"])
        b = self._mk(compile_ran=True, compile_success=False,
                     compile_errs=["error: x"])
        assert a.failure_signature() != b.failure_signature()


class TestBudgetReserveInvariant:
    """The initial codegen cap must leave room for at least one refine
    plus one compile. This is the load-bearing invariant: without it, the
    2026-07-10 failure recurs.
    """

    def test_code_cap_leaves_room_for_refine_and_compile(self):
        """budget=180, deadline just started, no time spent:
        code_cap = 180 - COMPILE_RESERVE - budget*REPAIR_RESERVE_FRACTION.
        This must be ≥ MIN_LLM_TIMEOUT and repair reserve must be ≥ 30s
        (loose but meaningful — enough for one refine attempt)."""
        budget = 180.0
        repair_reserve = budget * REPAIR_RESERVE_FRACTION
        code_cap = budget - COMPILE_RESERVE_SECONDS - repair_reserve
        assert code_cap >= 60, (
            f"code_cap={code_cap:.0f}s is too small for the initial coder")
        assert repair_reserve >= 30, (
            f"repair_reserve={repair_reserve:.0f}s cannot fit a real refine")
