"""Unit tests for the signature-keyed Pattern Memory cache."""
import sys
import os
import sqlite3
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pattern_memory.memory import PatternMemory
from pattern_memory.signature import (
    PatternSignature,
    normalize_severity,
    SEVERITY_UNKNOWN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fresh_memory():
    """Create a PatternMemory with a unique temp DB to avoid cross-test pollution."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return PatternMemory(db_path=path), path


def _cleanup(path):
    try:
        os.unlink(path)
        for ext in ('-wal', '-shm'):
            try:
                os.unlink(path + ext)
            except OSError:
                pass
    except OSError:
        pass


# Two findings sets with genuinely different migration problems.
WARP_REDUCE_FINDINGS = [
    {"pattern": "shfl_down_sync", "severity": "high", "line": 21},
    {"pattern": "shared_mem_warp_tiling", "severity": "medium", "line": 8},
    {"pattern": "warp_size_constant", "severity": "medium", "line": 8},
]
HISTOGRAM_FINDINGS = [
    {"pattern": "shfl_xor_sync", "severity": "high", "line": 34},
    {"pattern": "warp_size_constant", "severity": "medium", "line": 54},
]


# ── PatternSignature (pure) ───────────────────────────────────────────────────

def test_signature_is_order_independent():
    """Reordering classifier findings must not change the signature."""
    a = PatternSignature.from_findings(WARP_REDUCE_FINDINGS)
    b = PatternSignature.from_findings(list(reversed(WARP_REDUCE_FINDINGS)))
    assert a == b
    assert a.serialize() == b.serialize()


def test_signature_dedupes_and_sorts():
    """Duplicate findings collapse; serialisation is sorted + canonical."""
    sig = PatternSignature.from_findings([
        {"pattern": "ballot", "severity": "high"},
        {"pattern": "ballot", "severity": "high"},          # duplicate
        {"pattern": "warp_shuffle", "severity": "high"},
        {"pattern": "wavefront_assumption", "severity": "medium"},
    ])
    assert sig.serialize() == "ballot:HIGH|warp_shuffle:HIGH|wavefront_assumption:MEDIUM"


def test_signature_missing_and_unknown_severity():
    """Missing/garbage severities normalise to UNKNOWN without raising."""
    assert normalize_severity(None) == SEVERITY_UNKNOWN
    assert normalize_severity("bogus") == SEVERITY_UNKNOWN
    sig = PatternSignature.from_findings([{"pattern": "texture_api"}])  # no severity
    assert sig.serialize() == "texture_api:UNKNOWN"


def test_signature_empty():
    """No findings -> empty signature."""
    assert PatternSignature.from_findings([]).is_empty
    assert PatternSignature.from_findings(None).is_empty


def test_signature_digest_is_stable_and_deterministic():
    """Digest depends only on content (not Python's salted hash())."""
    sig1 = PatternSignature.from_findings(WARP_REDUCE_FINDINGS)
    sig2 = PatternSignature.deserialize(sig1.serialize())
    assert sig1.digest() == sig2.digest()
    assert len(sig1.digest()) == 16


def test_signature_round_trips():
    """serialize() / deserialize() is lossless."""
    sig = PatternSignature.from_findings(HISTOGRAM_FINDINGS)
    assert PatternSignature.deserialize(sig.serialize()) == sig


# ── Cache basics ──────────────────────────────────────────────────────────────

def test_memory_init():
    m, path = _fresh_memory()
    assert m.count() == 0
    _cleanup(path)


def test_store_and_retrieve_by_findings():
    """Store then retrieve using classifier findings (the primary API)."""
    m, path = _fresh_memory()
    sig_id = m.store(
        verified_fix="__shared__ float shared[64];",
        confidence=0.95,
        findings=WARP_REDUCE_FINDINGS,
    )
    assert sig_id is not None and len(sig_id) == 16
    assert m.count() == 1

    match = m.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match is not None
    assert match["confidence"] == 0.95
    assert match["verified_fix"] == "__shared__ float shared[64];"
    assert match["match_type"] == "exact_signature"
    assert "retrieval_ms" in match
    _cleanup(path)


def test_retrieve_different_signature_misses():
    """The core fix: unrelated kernels (different signatures) never collide."""
    m, path = _fresh_memory()
    m.store(verified_fix="warp reduce fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)

    # Histogram has a different problem set -> must be a MISS, not warp's fix.
    match = m.retrieve(findings=HISTOGRAM_FINDINGS)
    assert match is None
    _cleanup(path)


def test_store_and_retrieve_derive_from_source():
    """When only raw source is given, the signature is derived via the classifier."""
    m, path = _fresh_memory()
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    hist_source = open("sample_kernels/cuda/histogram.cu").read()

    m.store(pattern_snippet=warp_source, verified_fix="warp fix", confidence=0.9)

    # Same kernel -> hit.
    assert m.retrieve(warp_source) is not None
    # Different kernel -> miss (this is what was broken before).
    assert m.retrieve(hist_source) is None
    _cleanup(path)


def test_signature_is_formatting_invariant():
    """Comments / blank lines that don't change detected patterns -> same hit."""
    m, path = _fresh_memory()
    warp_source = open("sample_kernels/cuda/warp_reduce.cu").read()
    m.store(pattern_snippet=warp_source, verified_fix="warp fix", confidence=0.9)

    reformatted = "// an extra banner comment\n\n\n" + warp_source + "\n// trailing\n"
    match = m.retrieve(reformatted)
    assert match is not None
    assert match["verified_fix"] == "warp fix"
    _cleanup(path)


def test_empty_signature_is_not_cached():
    """A kernel with no migration problems has nothing to cache."""
    m, path = _fresh_memory()
    safe = "__global__ void safe(float* a, float* b) { *a = *b; }"
    assert m.store(pattern_snippet=safe, verified_fix="noop", confidence=0.9) is None
    assert m.count() == 0
    assert m.retrieve(safe) is None
    _cleanup(path)


def test_update_same_signature_keeps_strongest_fix():
    """Re-storing the same signature updates in place with the higher-confidence fix."""
    m, path = _fresh_memory()
    id1 = m.store(verified_fix="fix v1", confidence=0.80, findings=WARP_REDUCE_FINDINGS)
    id2 = m.store(verified_fix="fix v2", confidence=0.95, findings=WARP_REDUCE_FINDINGS)
    assert id1 == id2
    assert m.count() == 1

    match = m.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match["confidence"] == 0.95
    assert match["verified_fix"] == "fix v2"

    # A lower-confidence re-store must not clobber the better fix.
    m.store(verified_fix="fix v3 (worse)", confidence=0.50, findings=WARP_REDUCE_FINDINGS)
    match = m.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match["confidence"] == 0.95
    assert match["verified_fix"] == "fix v2"
    _cleanup(path)


def test_explicit_signature_key():
    """Callers may pass a precomputed signature object or its serialised form."""
    m, path = _fresh_memory()
    sig = PatternSignature.from_findings(WARP_REDUCE_FINDINGS)
    m.store(verified_fix="fix", confidence=0.9, signature=sig)
    assert m.retrieve(signature=sig.serialize()) is not None
    _cleanup(path)


# ── Stats & persistence ───────────────────────────────────────────────────────

def test_get_stats_empty():
    m, path = _fresh_memory()
    stats = m.get_stats()
    assert stats["total_patterns"] == 0
    assert stats["avg_confidence"] == 0
    assert stats["total_retrievals"] == 0
    _cleanup(path)


def test_get_stats_with_data():
    m, path = _fresh_memory()
    m.store(verified_fix="a", confidence=0.9, findings=WARP_REDUCE_FINDINGS)
    m.store(verified_fix="b", confidence=0.75, findings=HISTOGRAM_FINDINGS)
    stats = m.get_stats()
    assert stats["total_patterns"] == 2
    assert 0.8 <= stats["avg_confidence"] <= 0.85

    m.retrieve(findings=WARP_REDUCE_FINDINGS)
    stats = m.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["total_retrievals"] == 1
    _cleanup(path)


def test_hit_and_miss_counters():
    m, path = _fresh_memory()
    m.store(verified_fix="fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)
    m.retrieve(findings=WARP_REDUCE_FINDINGS)   # hit
    m.retrieve(findings=HISTOGRAM_FINDINGS)     # miss
    stats = m.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.5
    _cleanup(path)


def test_persistence():
    """Cache persists to disk and reloads with correct signatures."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    m1 = PatternMemory(db_path=path)
    m1.store(verified_fix="persist fix", confidence=0.85, findings=WARP_REDUCE_FINDINGS)
    assert m1.count() == 1
    del m1

    m2 = PatternMemory(db_path=path)
    assert m2.count() == 1
    match = m2.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match is not None
    assert match["confidence"] == 0.85
    assert match["verified_fix"] == "persist fix"
    _cleanup(path)


def test_clear():
    m, path = _fresh_memory()
    m.store(verified_fix="fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)
    assert m.count() == 1
    m.clear()
    assert m.count() == 0
    assert m.retrieve(findings=WARP_REDUCE_FINDINGS) is None
    _cleanup(path)


def test_store_negative_placeholder_is_not_served():
    """A signature that only ever failed verification must not be served."""
    m, path = _fresh_memory()
    m.store_negative(findings=WARP_REDUCE_FINDINGS, error_message="compile failed")
    assert m.count() == 1                       # placeholder recorded
    assert m.retrieve(findings=WARP_REDUCE_FINDINGS) is None  # but never served
    _cleanup(path)


def test_store_negative_increments_failure_count():
    """Repeated failures accumulate on the signature's failure_count."""
    m, path = _fresh_memory()
    m.store_negative(findings=WARP_REDUCE_FINDINGS, error_message="e1")
    m.store_negative(findings=WARP_REDUCE_FINDINGS, error_message="e2")
    m.store(verified_fix="good fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)

    match = m.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match is not None                    # a real fix now exists -> served
    assert match["failure_count"] == 2          # prior failures preserved
    assert match["verified_fix"] == "good fix"
    _cleanup(path)


def test_store_negative_empty_signature_is_noop():
    m, path = _fresh_memory()
    m.store_negative(pattern_snippet="__global__ void safe(float* a){ *a = 1.0f; }")
    assert m.count() == 0
    _cleanup(path)


def test_failure_count_persists(tmp_path=None):
    """failure_count survives a reload from SQLite."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    m1 = PatternMemory(db_path=path)
    m1.store(verified_fix="fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)
    m1.store_negative(findings=WARP_REDUCE_FINDINGS)
    m1.close()

    m2 = PatternMemory(db_path=path)
    match = m2.retrieve(findings=WARP_REDUCE_FINDINGS)
    assert match["failure_count"] == 1
    m2.close()
    _cleanup(path)


def test_corrupted_row_is_skipped():
    """A malformed metadata blob must not break cache loading."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    m1 = PatternMemory(db_path=path)
    m1.store(verified_fix="fix", confidence=0.9, findings=WARP_REDUCE_FINDINGS)
    del m1

    # Corrupt the metadata JSON directly in SQLite.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE signature_cache SET metadata = ?", ("{not valid json",))
    conn.commit()
    conn.close()

    m2 = PatternMemory(db_path=path)          # must not raise
    # The corrupted-metadata row is skipped; the cache degrades gracefully.
    assert m2.count() == 0
    _cleanup(path)
