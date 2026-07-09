"""Unit tests for prompt_evolution module — RL-style prompt versioning.

TRIZ principles under test:
  #15 Dynamics      — checklist grows/shrinks per iteration
  #23 Feedback      — reward = compile error delta
  #22 Throwing Away — negative-score items are dropped
  #9  Preliminary Anti-Action — seed checklist + targeted expansion
  #20 Continuation  — evolution continues until compile passes or max iter
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from prompt_evolution import (
    PromptVersion,
    PromptOptimizer,
    prompt_opt,
)


# ── PromptVersion property tests ──────────────────────────────────────────────

def test_prompt_version_reward_positive():
    """errors_before=10, errors_after=3, iterations=1 → reward == 7.0"""
    pv = PromptVersion("v1", ["item1"], iterations_used=1,
                       compile_errors_before=10, compile_errors_after=3)
    assert pv.reward == 7.0


def test_prompt_version_reward_zero_iterations():
    """iterations=0 → reward == 0.0 (no division by zero)"""
    pv = PromptVersion("v1", ["item1"], iterations_used=0,
                       compile_errors_before=10, compile_errors_after=3)
    assert pv.reward == 0.0


def test_prompt_version_win_rate():
    """win_rate = wins / (wins + losses)"""
    pv = PromptVersion("v1", ["item1"], wins=3, losses=1)
    assert pv.win_rate == 0.75


def test_prompt_version_win_rate_zero():
    """win_rate == 0.0 when no wins or losses"""
    pv = PromptVersion("v1", ["item1"], wins=0, losses=0)
    assert pv.win_rate == 0.0


# ── PromptOptimizer.record_iteration tests ────────────────────────────────────

def test_optimizer_record_iteration_positive_reward():
    """errors_before=10, errors_after=5 → reward=+1, wins=1"""
    opt = PromptOptimizer()
    checklist = opt.get_checklist()
    result = opt.record_iteration(10, 5, checklist)
    assert result["reward"] == 1
    assert opt.current_version.wins == 1
    assert opt.current_version.losses == 0


def test_optimizer_record_iteration_negative_reward():
    """errors_before=5, errors_after=8 → reward=-1, losses=1"""
    opt = PromptOptimizer()
    checklist = opt.get_checklist()
    result = opt.record_iteration(5, 8, checklist)
    assert result["reward"] == -1
    assert opt.current_version.losses == 1
    assert opt.current_version.wins == 0


def test_optimizer_record_iteration_zero_reward():
    """errors_before=5, errors_after=5 → reward=0"""
    opt = PromptOptimizer()
    result = opt.record_iteration(5, 5, opt.get_checklist())
    assert result["reward"] == 0
    assert opt.current_version.wins == 0
    assert opt.current_version.losses == 0


def test_optimizer_record_iteration_updates_item_scores():
    """Item scores should increase with positive reward."""
    opt = PromptOptimizer()
    checklist = opt.get_checklist()[:3]
    opt.record_iteration(10, 5, checklist)
    for item in checklist:
        assert opt._item_scores[item] == 1


# ── PromptOptimizer.evolve_prompt tests ───────────────────────────────────────

def test_optimizer_evolve_drops_negative_items():
    """Record several iterations where an item gets negative scores, evolve,
    verify it's dropped from checklist (TRIZ #22)."""
    opt = PromptOptimizer()
    checklist = opt.get_checklist()

    # Record 3 negative iterations — all items get -1 each time.
    for _ in range(3):
        opt.record_iteration(10, 15, checklist)

    # All items should have score -3 now.
    evolved = opt.evolve_prompt([])

    # Bottom-2 items with negative scores should be dropped.
    # Since all items have the same score (-3), the last 2 in sorted order
    # (which are the last 2 of the seed list, stable-sorted) should be dropped.
    original_count = len(checklist)
    assert len(evolved.checklist) <= original_count
    assert len(evolved.checklist) == original_count - 2

    # Verify version_id incremented.
    assert evolved.version_id == "v2"


def test_optimizer_evolve_adds_expansion_items():
    """Pass compile_errors containing 'cudaMalloc' and 'cudaEvent', evolve,
    verify expansion items added (TRIZ #15 Dynamics)."""
    opt = PromptOptimizer()
    compile_errors = [
        "error: use of undeclared identifier 'cudaMalloc'",
        "error: use of undeclared identifier 'cudaEventCreate'",
    ]
    evolved = opt.evolve_prompt(compile_errors)

    # Expansion pool item 0 (cudaMalloc) and item 1 (cudaEvent) should be added.
    assert any("hipMalloc" in item for item in evolved.checklist)
    assert any("hipEvent_t" in item for item in evolved.checklist)


def test_optimizer_evolve_adds_multiple_expansion_items():
    """Multiple CUDA keywords in errors → multiple expansion items added."""
    opt = PromptOptimizer()
    compile_errors = [
        "error: 'cudaMalloc' was not declared",
        "error: 'checkCudaErrors' was not declared",
        "error: 'cudaDeviceSynchronize' was not declared",
        "error: 'cudaError_t' does not name a type",
    ]
    evolved = opt.evolve_prompt(compile_errors)
    assert any("hipMalloc" in item for item in evolved.checklist)
    assert any("checkCudaErrors" in item for item in evolved.checklist)
    assert any("hipDeviceSynchronize" in item for item in evolved.checklist)
    assert any("hipError_t" in item for item in evolved.checklist)


def test_optimizer_evolve_keeps_seed_on_no_errors():
    """Evolve with empty compile_errors → checklist stays same or similar
    (no expansion items added, no negative items to drop)."""
    opt = PromptOptimizer()
    # Record a neutral iteration (delta=0) so no items get negative scores.
    opt.record_iteration(5, 5, opt.get_checklist())
    evolved = opt.evolve_prompt([])
    # No expansion items should be added.
    assert len(evolved.checklist) == len(PromptOptimizer.SEED_CHECKLIST)
    # All seed items should still be present (none had negative scores).
    for item in PromptOptimizer.SEED_CHECKLIST:
        assert item in evolved.checklist


def test_optimizer_evolve_version_counter_increments():
    """Each evolve_prompt call increments the version counter."""
    opt = PromptOptimizer()
    v2 = opt.evolve_prompt([])
    assert v2.version_id == "v2"
    v3 = opt.evolve_prompt([])
    assert v3.version_id == "v3"


def test_optimizer_evolve_stores_old_version():
    """evolve_prompt should store the old version in _versions."""
    opt = PromptOptimizer()
    old_id = opt.current_version.version_id
    opt.evolve_prompt([])
    assert old_id in opt._versions


# ── PromptOptimizer.get_stats tests ───────────────────────────────────────────

def test_optimizer_get_stats():
    """After a few iterations, get_stats returns dict with win_rate, reward,
    iterations_used, version_id."""
    opt = PromptOptimizer()
    opt.record_iteration(10, 5, opt.get_checklist())
    opt.record_iteration(5, 2, opt.get_checklist())
    stats = opt.get_stats()
    assert "win_rate" in stats
    assert "reward" in stats
    assert "iterations_used" in stats
    assert "version_id" in stats
    assert stats["iterations_used"] == 2
    assert stats["win_rate"] == 1.0  # 2 wins, 0 losses
    assert stats["version_id"] == "v1_seed"


# ── PromptOptimizer.reset tests ───────────────────────────────────────────────

def test_optimizer_reset():
    """reset() restores to v1_seed state."""
    opt = PromptOptimizer()
    opt.record_iteration(10, 15, opt.get_checklist())  # negative reward
    opt.evolve_prompt(["cudaMalloc error"])
    assert opt.current_version.version_id != "v1_seed"

    opt.reset()
    assert opt.current_version.version_id == "v1_seed"
    assert opt.current_version.checklist == list(PromptOptimizer.SEED_CHECKLIST)
    assert opt._item_scores == {}
    assert opt._error_history == []
    assert opt._versions == {}
    assert opt._version_counter == 1


# ── Seed / expansion pool integrity tests ─────────────────────────────────────

def test_seeded_checklist_has_9_items():
    """SEED_CHECKLIST must have exactly 9 items (TRIZ #9)."""
    assert len(PromptOptimizer.SEED_CHECKLIST) == 9


def test_expansion_pool_has_10_items():
    """EXPANSION_POOL must have exactly 10 items (TRIZ #15)."""
    assert len(PromptOptimizer.EXPANSION_POOL) == 10


def test_singleton_exists():
    """The module-level prompt_opt singleton should be a PromptOptimizer."""
    assert isinstance(prompt_opt, PromptOptimizer)


# ── Integration: evolve → record → evolve cycle ───────────────────────────────

def test_evolve_record_cycle():
    """Full cycle: evolve, record, evolve again — scores persist across
    versions (TRIZ #20 Continuation)."""
    opt = PromptOptimizer()

    # First evolve (adds nothing since no errors).
    v2 = opt.evolve_prompt([])
    assert v2.version_id == "v2"

    # Record a positive iteration.
    result = opt.record_iteration(10, 3, v2.checklist)
    assert result["reward"] == 1

    # Evolve again — scores from v2 should influence ordering.
    v3 = opt.evolve_prompt(["cudaMalloc undeclared"])
    assert v3.version_id == "v3"
    # Expansion item for cudaMalloc should be present.
    assert any("hipMalloc" in item for item in v3.checklist)
