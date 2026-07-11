"""Tests for :mod:`verification.typed_diagnostic`.

The router's stagnation check was previously done on normalised error
*strings*. Two failure modes that motivated the switch to typed sets:

  * Two different missing symbols reported at different sites produced
    identical normalised strings after path/line stripping, so the loop
    falsely flagged a cycle and aborted early.
  * Two runs of the same defect sometimes produced different strings
    (column varied), so a real cycle went undetected.

These tests pin the collision semantics of ``diagnostic_set``.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verification.typed_diagnostic import diagnostic_set


def test_empty_input_returns_empty_set():
    assert diagnostic_set([]) == frozenset()
    assert diagnostic_set([""]) == frozenset()


def test_same_defect_at_different_lines_collides():
    a = diagnostic_set([
        "test.cpp:42:10: error: use of undeclared identifier 'foo'",
    ])
    b = diagnostic_set([
        "test.cpp:999:1: error: use of undeclared identifier 'foo'",
    ])
    assert a == b
    assert ("undeclared", "foo", "") in a


def test_different_symbols_do_not_collide():
    a = diagnostic_set([
        "test.cpp:42:10: error: use of undeclared identifier 'foo'",
    ])
    b = diagnostic_set([
        "test.cpp:42:10: error: use of undeclared identifier 'bar'",
    ])
    assert a != b


def test_missing_include_captured():
    triples = diagnostic_set([
        "test.cpp:1:10: fatal error: 'hip/hip_runtime.h' file not found",
    ])
    assert ("missing-include", "hip/hip_runtime.h", "") in triples


def test_no_member_captures_owner():
    triples = diagnostic_set([
        "test.cpp:5:2: error: no member named 'launch' in 'HipStream'",
    ])
    assert ("no-member", "launch", "HipStream") in triples


def test_unknown_kind_ignores_line_column():
    """The router's collision test must not falsely diverge on line drift."""
    a = diagnostic_set(["test.cpp:242:12: error: no matching function"])
    b = diagnostic_set(["test.cpp:99:5: error: no matching function"])
    assert a == b


def test_notes_and_carets_do_not_leak_into_set():
    triples = diagnostic_set([
        "test.cpp:42:10: error: use of undeclared identifier 'foo'",
        "test.cpp:42:10: note: expanded from macro 'BAR'",
        "         ^~~~",
    ])
    assert triples == frozenset({("undeclared", "foo", "")})
