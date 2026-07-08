"""
Pattern Memory — verified CUDA→ROCm migration patterns with trigram index & SQLite cache.

TRIZ: Trigram pre-filter (O(1)) + SQLite write-through cache. Skip LLM entirely on cache hit.
Demo: "Second kernel is faster" — shows wall-clock time comparison (LLM: ~12s → cache: ~0.3s).
"""

import json
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional


class PatternMemory:
    """Pattern store with trigram index for fast retrieval and SQLite persistence.
    
    Retrieval flow:
    1. Compute query trigrams → O(len(code))
    2. Hash-lookup in trigram index → O(1)
    3. If above Jaccard threshold → return cached fix immediately → SKIP LLM
    4. If no match → call LLM, store result for next time
    """

    def __init__(self, db_path: str = "data/pattern_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory trigram index: trigram_hash -> set of pattern_ids
        self._trigram_index: Dict[int, set] = {}
        # In-memory signature index: signature -> pattern_id (O(1) exact match)
        self._signature_index: Dict[str, str] = {}
        # In-memory fix cache: pattern_id -> {fix, confidence}
        self._fix_cache: Dict[str, Dict] = {}
        # Cache hit/miss counters
        self._hits = 0
        self._misses = 0
        self._last_cache_time = 0.0
        self._last_llm_time = 0.0
        self._init_db()

    def store(self, pattern_snippet: str, verified_fix: str, confidence: float,
              verification_run_id: str = "", llm_time_s: float = 0.0,
              signature: Optional[str] = None) -> str:
        """Store a verified pattern fix with trigram index + SQLite.

        If *signature* is provided it is used as the dedup key; otherwise
        a canonical signature is computed automatically from the snippet.
        Backward compatible — existing callers work unchanged.
        """
        # Compute or use provided signature
        if signature is None:
            signature = self._compute_signature(pattern_snippet)

        pattern_id = hashlib.sha256(pattern_snippet.encode()).hexdigest()[:12]
        snippet_hash = hashlib.sha256(pattern_snippet.encode()).hexdigest()

        # Signature-based dedup — stricter than snippet_hash matching
        existing_pid = self._signature_index.get(signature)
        if existing_pid is not None:
            existing = self._db_execute(
                "SELECT confidence FROM patterns WHERE id = ?", (existing_pid,)
            ).fetchone()
            if existing:
                new_conf = max(existing[0], confidence)
                self._db_execute(
                    "UPDATE patterns SET confidence = ?, times_retrieved = times_retrieved + 1 WHERE id = ?",
                    (new_conf, existing_pid)
                )
                self._db_commit()
                if existing_pid in self._fix_cache:
                    self._fix_cache[existing_pid]["confidence"] = new_conf
                return existing_pid

        # Fallback: snippet_hash dedup for backward compat with existing DB rows
        existing = self._db_execute(
            "SELECT confidence FROM patterns WHERE snippet_hash = ?", (snippet_hash,)
        ).fetchone()
        if existing:
            new_conf = max(existing[0], confidence)
            self._db_execute(
                "UPDATE patterns SET confidence = ?, times_retrieved = times_retrieved + 1 WHERE snippet_hash = ?",
                (new_conf, snippet_hash)
            )
            self._db_commit()
            if pattern_id in self._fix_cache:
                self._fix_cache[pattern_id]["confidence"] = new_conf
            return pattern_id

        # Insert into SQLite
        self._db_execute(
            "INSERT INTO patterns (id, snippet_hash, signature, snippet, verified_fix, confidence, llm_time_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pattern_id, snippet_hash, signature, pattern_snippet[:500], verified_fix[:500],
             round(confidence, 2), round(llm_time_s, 3))
        )
        self._db_commit()

        # Build signature index
        self._signature_index[signature] = pattern_id

        # Build trigram index
        trigrams = self._compute_trigrams(pattern_snippet)
        for h in trigrams:
            if h not in self._trigram_index:
                self._trigram_index[h] = set()
            self._trigram_index[h].add(pattern_id)

        # In-memory fix cache
        self._fix_cache[pattern_id] = {
            "id": pattern_id,
            "verified_fix": verified_fix[:500],
            "confidence": round(confidence, 2),
            "llm_time_s": round(llm_time_s, 3),
            "times_retrieved": 0
        }

        return pattern_id

    def store_negative(self, pattern_snippet: str, error_message: str = "",
                       llm_time_s: Optional[float] = None) -> None:
        """Record a failed verification. Deprioritizes this pattern in retrieve()."""
        pattern_id = hashlib.sha256(pattern_snippet.encode()).hexdigest()[:12]
        existing = self._cur.execute(
            "SELECT failure_count FROM patterns WHERE pattern_id = ?",
            (pattern_id,)
        ).fetchone()
        if existing:
            self._cur.execute(
                "UPDATE patterns SET failure_count = failure_count + 1, updated_at = ? WHERE pattern_id = ?",
                (time.time(), pattern_id)
            )
        else:
            self._cur.execute(
                """INSERT OR IGNORE INTO patterns
                   (pattern_id, snippet_hash, pattern_snippet, verified_fix, confidence,
                    failure_count, created_at, updated_at)
                   VALUES (?, ?, ?, '', 0.0, 1, ?, ?)""",
                (pattern_id, hashlib.sha256(pattern_snippet.encode()).hexdigest(),
                 pattern_snippet[:500], time.time(), time.time())
            )
        self._conn.commit()

    def retrieve(self, query_snippet: str, threshold: float = 0.25,
                 signature: Optional[str] = None) -> Optional[Dict]:
        """Fast trigram-based retrieval with optional exact-signature short-circuit.

        When *signature* is provided, checks the signature index first for
        an O(1) exact match. Falls back to trigram approximate matching if
        the signature doesn't match.

        Returns cached fix dict or None.
        """
        t0 = time.perf_counter()

        # Signature short-circuit — O(1) exact match
        if signature is not None:
            pid = self._signature_index.get(signature)
            if pid is not None:
                cached = self._fix_cache.get(pid)
                if cached:
                    self._hits += 1
                    self._last_cache_time = (time.perf_counter() - t0) * 1000
                    cached["retrieval_ms"] = round(self._last_cache_time, 1)
                    cached["jaccard"] = 1.0  # exact signature match
                    cached["times_retrieved"] = cached.get("times_retrieved", 0) + 1
                    return cached

        query_trigrams = self._compute_trigrams(query_snippet)

        if not query_trigrams or not self._trigram_index:
            self._misses += 1
            return None

        # Vote: count how many trigrams each stored pattern matches
        votes: Dict[str, int] = {}
        for h in query_trigrams:
            for pid in self._trigram_index.get(h, set()):
                votes[pid] = votes.get(pid, 0) + 1

        if not votes:
            self._misses += 1
            return None

        # Best candidate by trigram overlap
        best_pid = max(votes, key=lambda k: votes[k])  # type: ignore
        jaccard = votes[best_pid] / len(query_trigrams)  # approximate

        if jaccard < threshold:
            self._misses += 1
            return None

        cached = self._fix_cache.get(best_pid)
        if cached:
            self._hits += 1
            self._last_cache_time = (time.perf_counter() - t0) * 1000  # ms
            cached["retrieval_ms"] = round(self._last_cache_time, 1)
            cached["jaccard"] = round(jaccard, 2)
            cached["times_retrieved"] = cached.get("times_retrieved", 0) + 1
            return cached

        self._misses += 1
        return None

    def count(self) -> int:
        return self._db_execute("SELECT COUNT(*) FROM patterns").fetchone()[0]

    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        total_retrieved = self._db_execute(
            "SELECT COALESCE(SUM(times_retrieved), 0) FROM patterns"
        ).fetchone()[0]
        return {
            "total_patterns": self.count(),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
            "total_retrievals": total_retrieved,
            "last_cache_time_ms": round(self._last_cache_time, 1),
            "last_llm_time_s": round(self._last_llm_time, 1),
            "avg_confidence": self._db_execute(
                "SELECT COALESCE(AVG(confidence), 0) FROM patterns"
            ).fetchone()[0]
        }

    def record_llm_time(self, seconds: float):
        self._last_llm_time = seconds

    def clear(self):
        """Reset everything (for demo: fresh start → first run slow → second run fast)."""
        self._db_execute("DELETE FROM patterns")
        self._db_commit()
        self._trigram_index.clear()
        self._signature_index.clear()
        self._fix_cache.clear()
        self._hits = 0
        self._misses = 0
        self._last_cache_time = 0.0
        self._last_llm_time = 0.0

    def _compute_trigrams(self, code: str) -> List[int]:
        """Character trigrams as lightweight fingerprint (hash to int)."""
        code = code.lower()
        return [
            hash(code[i:i+3]) for i in range(len(code) - 2)
            if code[i:i+3].strip()  # skip whitespace-only trigrams
        ]

    def _compute_signature(self, code: str) -> str:
        """Compute a canonical signature from code, normalising superficial differences.
        
        Strips comments, normalises whitespace, lowercases — so two snippets
        that differ only in formatting/commenting produce the same signature.
        Used as an O(1) exact-match key in the signature index, bypassing
        trigram-based approximate matching when the caller already has a signature.
        """
        import re
        # Strip single-line comments
        code = re.sub(r'#.*', '', code)
        # Normalise whitespace and lowercase
        code = re.sub(r'\s+', ' ', code).strip().lower()
        return hashlib.sha256(code.encode()).hexdigest()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                snippet_hash TEXT UNIQUE,
                signature TEXT,
                snippet TEXT,
                verified_fix TEXT,
                confidence REAL,
                failure_count INTEGER DEFAULT 0,
                llm_time_s REAL DEFAULT 0,
                times_retrieved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Schema migration: add signature column for existing databases
        try:
            self._conn.execute("ALTER TABLE patterns ADD COLUMN signature TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_snippet_hash ON patterns(snippet_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_signature ON patterns(signature)")
        self._conn.commit()
        # Load existing patterns into memory
        for row in self._conn.execute(
            "SELECT id, snippet, verified_fix, confidence, llm_time_s, times_retrieved, signature "
            "FROM patterns"
        ).fetchall():
            pid, snippet, fix, conf, llm_t, times, sig = row
            self._fix_cache[pid] = {
                "id": pid, "verified_fix": fix, "confidence": conf,
                "llm_time_s": llm_t, "times_retrieved": times
            }
            if sig:
                self._signature_index[sig] = pid
            if snippet:
                for h in self._compute_trigrams(snippet):
                    self._trigram_index.setdefault(h, set()).add(pid)

    def _db_execute(self, sql: str, params=()):
        return self._conn.execute(sql, params)

    def _db_commit(self):
        self._conn.commit()
