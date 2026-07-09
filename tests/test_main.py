"""Tests for main.py's verification status labeling.

Bug 0 (docs/fix-plan-self-contained-programs.md): compile_success /
run_success / output_match are three DIFFERENT failure points, but the old
code derived its terminal label from compile_success alone — so a binary
that compiled and then crashed at runtime printed "Output mismatch",
identically to a genuine diff failure, even though verify() (verifier.py)
returns before the diff step ever runs on a run failure.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# main.py auto-loads the repo's .env file at import time via
# os.environ.setdefault(...) — correct for `python main.py` from the CLI,
# but importing main here as a test dependency would otherwise leak a real
# FIREWORKS_API_KEY into the shared pytest process for every test file that
# runs afterward (e.g. flips ReportGenerator.use_gemma to True in
# test_reporter.py, which then fails over the network mid-test and falls
# back to a path that silently drops memory_stats). Snapshot/restore
# os.environ around the import so this module has zero side effects on the
# rest of the suite.
_env_snapshot = dict(os.environ)
from main import verification_failure_label
os.environ.clear()
os.environ.update(_env_snapshot)


def test_not_compiled_label():
    result = {"compile_success": False, "run_success": False}
    assert verification_failure_label(result) == "Not compiled — saved for manual hipcc"


def test_runtime_crash_is_not_labeled_output_mismatch():
    """The exact regression this bug describes: compiled fine, crashed at
    runtime, no diff ever ran — must NOT say 'Output mismatch'."""
    result = {"compile_success": True, "run_success": False, "run_exit_code": 1}
    label = verification_failure_label(result)
    assert label != "Output mismatch"
    assert "crashed at runtime" in label
    assert "exit 1" in label


def test_runtime_failure_without_exit_code():
    result = {"compile_success": True, "run_success": False, "run_exit_code": None}
    label = verification_failure_label(result)
    assert label != "Output mismatch"
    assert "failed to run" in label


def test_genuine_output_mismatch_label():
    """compiled AND ran successfully, only the diff failed — this is the
    one case that should actually say 'Output mismatch'."""
    result = {"compile_success": True, "run_success": True, "output_match": False}
    assert verification_failure_label(result) == "Output mismatch"


def test_waived_exit_code_reported_distinctly():
    """The NVIDIA sample's EXIT_WAIVED=2 on unsupported hardware is still a
    run failure (nonzero exit) — the label must surface the real exit code
    rather than collapsing it into the same bucket as a crash."""
    result = {"compile_success": True, "run_success": False, "run_exit_code": 2}
    label = verification_failure_label(result)
    assert "exit 2" in label
