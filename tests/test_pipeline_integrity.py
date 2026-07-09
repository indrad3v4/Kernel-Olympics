"""Tests for the P0 pipeline-integrity work (T0.1–T0.5):
verification-gated confidence, the honest run verdict, and the
verified/quarantine split in Pattern Memory.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import gate_confidence, compute_run_verdict
from pattern_memory.memory import PatternMemory


# ── T0.2: verification-gated confidence ─────────────────────────────

def test_gate_confidence_passed_keeps_score():
    assert gate_confidence(47, True) == 47
    assert gate_confidence(0.9, True) == 0.9


def test_gate_confidence_failed_is_zero():
    assert gate_confidence(47, False) == 0


def test_gate_confidence_compiled_but_not_verified_is_zero():
    """Item-4 edge case: compiled in-loop but the authoritative verify did
    NOT pass (crash / diff / orchestrator bailed) → 0, not the porter score.
    The gate keys on the verifier's `passed`, which is False here."""
    assert gate_confidence(47, False) == 0
    assert gate_confidence(88, None) == 0  # missing/None passed is falsy → 0


# ── T0.3: one honest run verdict + exit code ────────────────────────

def _v(passed=False, hipcc=True, compile_output=""):
    return {"passed": passed, "hipcc_available": hipcc, "compile_output": compile_output}

def test_verdict_no_ports():
    assert compute_run_verdict([]) == ("NO PORTS NEEDED", False)

def test_verdict_all_passed():
    assert compute_run_verdict([_v(True), _v(True)]) == ("PASSED", False)

def test_verdict_none_passed_is_failed_and_exits_nonzero():
    verdict, exit_fail = compute_run_verdict([_v(False)])
    assert verdict == "FAILED"
    assert exit_fail is True

def test_verdict_partial():
    verdict, exit_fail = compute_run_verdict([_v(True), _v(False)])
    assert verdict.startswith("PARTIAL")
    assert exit_fail is False

def test_verdict_no_gpu_via_flag_is_unverified_not_failed():
    verdict, exit_fail = compute_run_verdict([_v(False, hipcc=False)])
    assert verdict == "UNVERIFIED (no GPU)"
    assert exit_fail is False  # can't test != tested-and-failed

def test_verdict_no_gpu_via_compile_output():
    v = _v(False, hipcc=True, compile_output="hipcc not found on PATH")
    assert compute_run_verdict([v]) == ("UNVERIFIED (no GPU)", False)

def test_verdict_gpu_present_failure_beats_no_gpu_sibling():
    """A real GPU failure alongside a no-GPU result is still FAILED — 'all
    no GPU' is what makes it UNVERIFIED, not 'any'."""
    verdict, exit_fail = compute_run_verdict([_v(False, hipcc=True), _v(False, hipcc=False)])
    assert verdict == "FAILED"
    assert exit_fail is True


# ── T0.4 / T0.5: verified vs quarantined in Pattern Memory ──────────

def _mem():
    return PatternMemory(db_path=os.path.join(tempfile.mkdtemp(), "t.db"))

FIND = [{"pattern": "warp_size_constant", "line": 1, "severity": "high"}]

def test_unverified_store_is_never_served():
    m = _mem()
    m.store(pattern_snippet="x", verified_fix="BROKEN", confidence=0.085,
            findings=FIND, verified=False)
    assert m.retrieve(findings=FIND) is None      # quarantined → miss
    assert m.count() == 1                          # but kept for resume
    assert m.count(verified_only=True) == 0

def test_verified_store_is_served():
    m = _mem()
    m.store(pattern_snippet="x", verified_fix="GOOD", confidence=0.9,
            findings=FIND, verified=True)
    r = m.retrieve(findings=FIND)
    assert r is not None and r["verified_fix"] == "GOOD"
    assert m.count(verified_only=True) == 1

def test_unverified_never_clobbers_verified():
    m = _mem()
    m.store(pattern_snippet="x", verified_fix="GOOD", confidence=0.5,
            findings=FIND, verified=True)
    # A higher-confidence but UNVERIFIED attempt must not overwrite it.
    m.store(pattern_snippet="x", verified_fix="BROKEN", confidence=0.99,
            findings=FIND, verified=False)
    assert m.retrieve(findings=FIND)["verified_fix"] == "GOOD"

def test_verified_promotes_prior_unverified():
    m = _mem()
    m.store(pattern_snippet="x", verified_fix="RESUME", confidence=0.8,
            findings=FIND, verified=False)
    assert m.retrieve(findings=FIND) is None       # not served while unverified
    m.store(pattern_snippet="x", verified_fix="GOOD", confidence=0.1,
            findings=FIND, verified=True)           # lower conf, but verified
    assert m.retrieve(findings=FIND)["verified_fix"] == "GOOD"

def test_verified_flag_survives_reopen():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    m = PatternMemory(db_path=path)
    m.store(pattern_snippet="x", verified_fix="BROKEN", confidence=0.1,
            findings=FIND, verified=False)
    m2 = PatternMemory(db_path=path)               # reopen from disk
    assert m2.retrieve(findings=FIND) is None       # still quarantined
    assert m2.count() == 1 and m2.count(verified_only=True) == 0
