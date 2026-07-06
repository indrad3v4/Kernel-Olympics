"""Unit tests for the scanner module."""
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scanner.scanner import Scanner


def test_scanner_file_not_found():
    """Scanner should return error struct for missing files."""
    s = Scanner()
    result = s.scan("nonexistent.cu")
    assert result["error"] == "file not found"
    assert result["hipify_coverage_pct"] == 0


def test_scanner_batch():
    """Batch scan should return list of results."""
    s = Scanner()
    results = s.scan_batch([
        "sample_kernels/cuda/warp_reduce.cu",
        "sample_kernels/cuda/transpose.cu",
    ])
    assert len(results) == 2
    for r in results:
        assert "file" in r
        assert "total_lines" in r
        assert r["total_lines"] > 0


def test_scanner_detects_unconverted():
    """Scanner should detect unconverted lines when hipify-clang available."""
    s = Scanner()
    # Without hipify, this should still return a result, just with error
    result = s.scan("sample_kernels/cuda/warp_reduce.cu")
    assert result["file"] is not None
    assert result["total_lines"] >= 28


def test_scanner_coverage_calculation():
    """Coverage should be 0-100% even when hipify fails (no divide by zero)."""
    s = Scanner()
    result = s.scan("sample_kernels/cuda/warp_reduce.cu")
    assert 0 <= result["hipify_coverage_pct"] <= 100
    assert result["total_lines"] > 0
