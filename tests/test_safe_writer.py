"""Tests for verification.safe_writer — the file-writer hardening layer."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from verification.safe_writer import safe_write_source  # noqa: E402


CLEAN_KERNEL = """\
#include <hip/hip_runtime.h>

__global__ void vec_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
"""

PROSE = """\
#include <iostream>
#include <hip/hip_runtime.h>

Let's search memory more concretely.

I think we need a different approach.
"""


def test_writes_clean_code(tmp_path):
    dst = tmp_path / "k.hip.cpp"
    r = safe_write_source(dst, CLEAN_KERNEL, source_cuda=CLEAN_KERNEL)
    assert r.written
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == CLEAN_KERNEL


def test_rejects_prose_payload(tmp_path):
    dst = tmp_path / "k.hip.cpp"
    r = safe_write_source(dst, PROSE, source_cuda=CLEAN_KERNEL)
    assert not r.written
    assert not dst.exists()
    assert "lexical" in r.reason.lower()


def test_rejects_empty_payload(tmp_path):
    dst = tmp_path / "k.hip.cpp"
    r = safe_write_source(dst, "", source_cuda=CLEAN_KERNEL)
    assert not r.written
    assert not dst.exists()


def test_rejects_unbalanced_braces(tmp_path):
    dst = tmp_path / "k.hip.cpp"
    truncated = "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
    r = safe_write_source(dst, truncated, source_cuda=CLEAN_KERNEL)
    assert not r.written
    assert "structural" in r.reason.lower()


def test_preserves_previous_good_file_on_reject(tmp_path):
    """A rejected write must leave the last good file untouched."""
    dst = tmp_path / "k.hip.cpp"
    ok = safe_write_source(dst, CLEAN_KERNEL, source_cuda=CLEAN_KERNEL)
    assert ok.written

    prev_bytes = dst.read_bytes()
    bad = safe_write_source(dst, PROSE, source_cuda=CLEAN_KERNEL)
    assert not bad.written
    # File on disk is byte-identical to before the failed write.
    assert dst.read_bytes() == prev_bytes


def test_result_serializes_to_dict(tmp_path):
    dst = tmp_path / "k.hip.cpp"
    r = safe_write_source(dst, PROSE, source_cuda=CLEAN_KERNEL)
    d = r.to_dict()
    assert d["written"] is False
    assert "reason" in d
    assert "lexical_ok" in d
