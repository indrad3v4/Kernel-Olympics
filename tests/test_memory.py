"""Unit tests for the pattern memory module (SQLite + trigram index)."""
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pattern_memory.memory import PatternMemory


def _fresh_memory():
    """Create a PatternMemory with a unique temp DB to avoid cross-test pollution."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    m = PatternMemory(db_path=path)
    return m, path


def _cleanup(path):
    try:
        os.unlink(path)
        # Also clean up WAL/SHM files SQLite may create
        for ext in ('-wal', '-shm'):
            try:
                os.unlink(path + ext)
            except OSError:
                pass
    except OSError:
        pass


def test_memory_init():
    """PatternMemory should initialize with empty patterns."""
    m, path = _fresh_memory()
    assert m.count() == 0
    _cleanup(path)


def test_store_and_retrieve():
    """Storing and retrieving a pattern should work."""
    m, path = _fresh_memory()
    pattern_id = m.store(
        pattern_snippet="__shared__ float shared[32];",
        verified_fix="__shared__ float shared[64];",
        confidence=0.95,
        verification_run_id="run_001"
    )
    assert pattern_id is not None
    assert len(pattern_id) == 12
    assert m.count() == 1

    match = m.retrieve("__shared__ float shared[32]; int tid = threadIdx.x;")
    assert match is not None
    assert match["confidence"] >= 0.95
    assert "jaccard" in match
    assert match["jaccard"] > 0
    assert "retrieval_ms" in match
    _cleanup(path)


def test_retrieve_no_match():
    """Retrieve with unrelated code should return None."""
    m, path = _fresh_memory()
    m.store("int x = 42;", "int x = 42;", 0.9, "run_001")

    match = m.retrieve("std::vector<int> data; for (auto& v : data) { v *= 2; }")
    assert match is None
    _cleanup(path)


def test_get_stats_empty():
    """get_stats on empty memory should return zeros."""
    m, path = _fresh_memory()
    stats = m.get_stats()
    assert stats["total_patterns"] == 0
    assert stats["avg_confidence"] == 0
    assert stats["total_retrievals"] == 0
    _cleanup(path)


def test_get_stats_with_data():
    """get_stats should return accurate aggregations."""
    m, path = _fresh_memory()
    m.store("pattern_a", "fix_a", 0.9, "run_a")
    m.store("pattern_b", "fix_b", 0.75, "run_b")
    stats = m.get_stats()
    assert stats["total_patterns"] == 2
    assert 0.8 <= stats["avg_confidence"] <= 0.85
    assert stats["total_retrievals"] == 0
    _cleanup(path)


def test_update_existing_pattern():
    """Storing the same pattern should update confidence (max)."""
    m, path = _fresh_memory()
    pid1 = m.store("same code block", "fix v1", 0.8, "run_001")
    pid2 = m.store("same code block", "fix v2", 0.95, "run_002")
    assert pid1 == pid2
    assert m.count() == 1
    match = m.retrieve("same code block")
    assert match is not None
    assert match["confidence"] == 0.95
    _cleanup(path)


def test_persistence():
    """Pattern memory should persist to disk and reload."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    m1 = PatternMemory(db_path=path)
    m1.store("persist test", "persist fix", 0.85, "run_persist")
    assert m1.count() == 1
    del m1

    m2 = PatternMemory(db_path=path)
    assert m2.count() == 1
    match = m2.retrieve("persist test")
    assert match is not None
    assert match["confidence"] == 0.85
    _cleanup(path)


def test_trigram_computation():
    """Trigram computation should be deterministic for same code."""
    m, path = _fresh_memory()
    code = "int x = 0; for (int i = 0; i < 10; i++) { x += i; }"
    trig1 = m._compute_trigrams(code)
    trig2 = m._compute_trigrams(code)
    assert trig1 == trig2
    assert len(trig1) > 0
    _cleanup(path)
