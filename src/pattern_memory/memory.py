"""
Pattern Memory — signature-keyed cache of verified CUDA→ROCm migrations.

The cache answers one question:

    "Have we already solved this collection of migration problems before?"

It is keyed on the **pattern signature** produced by the risk classifier — the
canonical set of ``(pattern, severity)`` problems in a kernel — *not* on
source-text similarity. See :mod:`pattern_memory.signature`.

Why this replaces the previous trigram / Jaccard design
-------------------------------------------------------
The old cache compared character trigrams of the first ~500 characters of
source and accepted any match above a 0.25 Jaccard threshold. Unrelated kernels
share large amounts of CUDA boilerplate (``#include``, ``__global__``, launch
configuration, ``threadIdx`` arithmetic), so their trigram sets overlapped and
the cache returned the **wrong** ported kernel — e.g. a warp-reduction fix
served for a histogram kernel. That silently breaks correctness.

A signature depends only on migration *semantics*. It is invariant to
whitespace, comments, identifier names and code ordering. Two kernels are a hit
only when they present the same migration problems, so a hit always returns an
applicable fix. Lookup is an exact O(1) dict / primary-key match: deterministic,
reproducible, and free of any fuzzy threshold to tune.

Signature source of truth
--------------------------
Callers should pass the classifier ``findings`` (or a precomputed signature) to
:meth:`store` / :meth:`retrieve`. When only raw source is supplied the module
derives the signature by running the classifier itself — this keeps older
call-sites working, but note that a *truncated* snippet may yield an incomplete
signature, so passing ``findings`` computed on the full source is preferred.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from pattern_memory.signature import PatternSignature


@dataclass
class CacheEntry:
    """One cached migration keyed by its pattern signature."""

    signature: str            # serialised PatternSignature (the primary key)
    sig_id: str               # stable SHA-256 short id of the signature
    verified_fix: str         # the ported HIP kernel we serve on a hit
    confidence: float         # 0..1
    llm_time_s: float = 0.0   # measured LLM time this fix originally cost
    times_retrieved: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_public(self, retrieval_ms: Optional[float] = None) -> Dict:
        """Serialise to the plain dict returned from :meth:`PatternMemory.retrieve`."""
        result = {
            "id": self.sig_id,
            "signature": self.signature,
            "verified_fix": self.verified_fix,
            "confidence": self.confidence,
            "llm_time_s": self.llm_time_s,
            "times_retrieved": self.times_retrieved,
            "match_type": "exact_signature",
            "metadata": dict(self.metadata),
        }
        if retrieval_ms is not None:
            result["retrieval_ms"] = round(retrieval_ms, 3)
        return result


class PatternMemory:
    """Exact-match cache of verified fixes, keyed by classifier pattern signature.

    Storage is dual: an authoritative in-memory dict for O(1) lookup, plus a
    write-through SQLite table for persistence across runs. Both are keyed by the
    serialised signature.
    """

    _TABLE = "signature_cache"

    def __init__(self, db_path: str = "data/pattern_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Authoritative in-memory cache: serialized signature -> CacheEntry.
        self._cache: Dict[str, CacheEntry] = {}

        # Hit/miss + timing telemetry (all measured, never fabricated).
        self._hits = 0
        self._misses = 0
        self._last_cache_time_ms = 0.0
        self._last_llm_time_s = 0.0

        # Lazily-constructed classifier, used only to derive a signature when a
        # caller supplies raw source instead of findings.
        self._classifier = None

        self._init_db()
        self._load_cache()

    # ── Public API ───────────────────────────────────────────────────────────

    def store(self, pattern_snippet: str = "", verified_fix: str = "",
              confidence: float = 0.0, verification_run_id: str = "",
              llm_time_s: float = 0.0, *, findings=None,
              signature=None, metadata: Optional[Dict] = None) -> Optional[str]:
        """Cache a verified fix under its pattern signature.

        The signature is taken from (in priority order) ``signature``, then
        ``findings``, then by classifying ``pattern_snippet``. Kernels with no
        migration problems produce an empty signature and are **not** cached —
        there is nothing to reuse — in which case ``None`` is returned.

        Storing a signature that already exists updates it in place (keeping the
        higher-confidence fix), so the same migration problem is stored once.

        Returns the stable signature id, or ``None`` if nothing was cached.
        """
        sig = self._derive_signature(source=pattern_snippet, findings=findings, signature=signature)
        if sig.is_empty:
            return None

        key = sig.serialize()
        confidence = round(float(confidence), 4)
        meta = dict(metadata or {})
        if verification_run_id:
            meta.setdefault("verification_run_id", verification_run_id)
        if pattern_snippet:
            meta.setdefault("source_preview", pattern_snippet[:200])

        existing = self._cache.get(key)
        if existing is not None:
            # Same migration problem seen before — keep the strongest fix.
            if confidence >= existing.confidence:
                existing.verified_fix = verified_fix
                existing.confidence = confidence
                existing.llm_time_s = round(float(llm_time_s), 3)
                existing.metadata.update(meta)
            else:
                existing.confidence = max(existing.confidence, confidence)
            self._persist(existing)
            return existing.sig_id

        entry = CacheEntry(
            signature=key,
            sig_id=sig.digest(),
            verified_fix=verified_fix,
            confidence=confidence,
            llm_time_s=round(float(llm_time_s), 3),
            metadata=meta,
        )
        self._cache[key] = entry
        self._persist(entry)
        return entry.sig_id

    def retrieve(self, query_snippet: Optional[str] = None, *, findings=None,
                 signature=None) -> Optional[Dict]:
        """Return the cached fix for this kernel's signature, or ``None``.

        Exact signature match only — no fuzzy similarity. The signature is taken
        from ``signature``, then ``findings``, then by classifying
        ``query_snippet``. An empty signature (no migration problems) is always a
        miss.
        """
        start = time.perf_counter()
        sig = self._derive_signature(source=query_snippet, findings=findings, signature=signature)
        if sig.is_empty:
            self._misses += 1
            return None

        entry = self._cache.get(sig.serialize())
        if entry is None:
            self._misses += 1
            return None

        self._hits += 1
        entry.times_retrieved += 1
        self._last_cache_time_ms = (time.perf_counter() - start) * 1000.0
        self._persist(entry)  # persist the retrieval count
        return entry.to_public(retrieval_ms=self._last_cache_time_ms)

    def count(self) -> int:
        """Number of distinct migration signatures cached."""
        return len(self._cache)

    def get_stats(self) -> Dict:
        """Aggregate telemetry for reporting (all values measured)."""
        total = self._hits + self._misses
        confidences = [e.confidence for e in self._cache.values()]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0
        return {
            "total_patterns": self.count(),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
            "total_retrievals": sum(e.times_retrieved for e in self._cache.values()),
            "last_cache_time_ms": round(self._last_cache_time_ms, 3),
            "last_llm_time_s": round(self._last_llm_time_s, 3),
            "avg_confidence": avg_confidence,
        }

    def record_llm_time(self, seconds: float) -> None:
        """Record the most recent measured LLM time (for speedup reporting)."""
        self._last_llm_time_s = float(seconds)

    def close(self) -> None:
        """Release the SQLite connection. Safe to call multiple times."""
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __del__(self):  # best-effort cleanup if the caller forgets to close()
        self.close()

    def clear(self) -> None:
        """Wipe the cache (both SQLite and in-memory) and reset telemetry."""
        if getattr(self, "_conn", None) is not None:
            try:
                self._conn.execute(f"DELETE FROM {self._TABLE}")
                self._conn.commit()
            except sqlite3.Error:
                pass
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._last_cache_time_ms = 0.0
        self._last_llm_time_s = 0.0

    # ── Signature derivation ─────────────────────────────────────────────────

    def _derive_signature(self, *, source=None, findings=None, signature=None) -> PatternSignature:
        """Resolve a :class:`PatternSignature` from whatever the caller provided.

        Priority: explicit ``signature`` > classifier ``findings`` > classifying
        raw ``source``. Falls back to an empty signature when nothing usable is
        available, so callers never have to guard against exceptions.
        """
        if signature is not None:
            if isinstance(signature, PatternSignature):
                return signature
            return PatternSignature.deserialize(str(signature))
        if findings is not None:
            return PatternSignature.from_findings(findings)
        if source:
            return self._classify_to_signature(source)
        return PatternSignature(())

    def _classify_to_signature(self, source: str) -> PatternSignature:
        """Derive a signature by running the risk classifier on raw source.

        Lazily imports the classifier so Pattern Memory has no hard import-time
        dependency on it; any failure degrades gracefully to an empty signature.
        """
        try:
            if self._classifier is None:
                from risk_classifier.classifier import RiskClassifier
                self._classifier = RiskClassifier()
            result = self._classifier.classify(source)
            return PatternSignature.from_findings(result.get("findings", []))
        except Exception:
            return PatternSignature(())

    # ── SQLite persistence ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                signature       TEXT PRIMARY KEY,
                sig_id          TEXT NOT NULL,
                verified_fix    TEXT,
                confidence      REAL DEFAULT 0,
                llm_time_s      REAL DEFAULT 0,
                times_retrieved INTEGER DEFAULT 0,
                metadata        TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def _load_cache(self) -> None:
        """Populate the in-memory cache from SQLite, skipping corrupted rows."""
        try:
            rows = self._conn.execute(
                f"SELECT signature, sig_id, verified_fix, confidence, llm_time_s, "
                f"times_retrieved, metadata FROM {self._TABLE}"
            ).fetchall()
        except sqlite3.Error:
            return

        for row in rows:
            try:
                signature, sig_id, fix, confidence, llm_t, times, meta_json = row
                if not signature:
                    continue
                metadata = json.loads(meta_json) if meta_json else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                self._cache[signature] = CacheEntry(
                    signature=signature,
                    sig_id=sig_id or PatternSignature.deserialize(signature).digest(),
                    verified_fix=fix or "",
                    confidence=float(confidence or 0),
                    llm_time_s=float(llm_t or 0),
                    times_retrieved=int(times or 0),
                    metadata=metadata,
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                # Corrupted entry — skip it rather than failing the whole load.
                continue

    def _persist(self, entry: CacheEntry) -> None:
        """Write-through upsert of a single entry. Best-effort: the in-memory
        cache stays authoritative even if the DB write fails."""
        if getattr(self, "_conn", None) is None:
            return
        try:
            self._conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (signature, sig_id, verified_fix, confidence, llm_time_s,
                     times_retrieved, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(signature) DO UPDATE SET
                    sig_id          = excluded.sig_id,
                    verified_fix    = excluded.verified_fix,
                    confidence      = excluded.confidence,
                    llm_time_s      = excluded.llm_time_s,
                    times_retrieved = excluded.times_retrieved,
                    metadata        = excluded.metadata,
                    updated_at      = CURRENT_TIMESTAMP
                """,
                (
                    entry.signature, entry.sig_id, entry.verified_fix,
                    entry.confidence, entry.llm_time_s, entry.times_retrieved,
                    json.dumps(entry.metadata),
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            pass
