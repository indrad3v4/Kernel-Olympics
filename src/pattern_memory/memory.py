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
        # In-memory fix cache: pattern_id -> {fix, confidence}
        self._fix_cache: Dict[str, Dict] = {}
        # Cache hit/miss counters
        self._hits = 0
        self._misses = 0
        self._last_cache_time = 0.0
        self._last_llm_time = 0.0
        self._init_db()

    def store(self, pattern_snippet: str, verified_fix: str, confidence: float,
              verification_run_id: str = "", llm_time_s: float = 0.0) -> str:
        """Store a verified pattern fix with trigram index + SQLite."""
        pattern_id = hashlib.sha256(pattern_snippet.encode()).hexdigest()[:12]
        snippet_hash = hashlib.sha256(pattern_snippet.encode()).hexdigest()

        # Already exists — update confidence
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
            "INSERT INTO patterns (id, snippet_hash, snippet, verified_fix, confidence, llm_time_s) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pattern_id, snippet_hash, pattern_snippet[:500], verified_fix[:500],
             round(confidence, 2), round(llm_time_s, 3))
        )
        self._db_commit()

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

    def retrieve(self, query_snippet: str, threshold: float = 0.25) -> Optional[Dict]:
        """Fast trigram-based retrieval. Returns cached fix or None.
        
        Returns immediately with cached fix if trigram Jaccard similarity > threshold.
        No cosine similarity, no linear scan — O(1) dict lookup.
        """
        t0 = time.perf_counter()
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
        return {
            "total_patterns": self.count(),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
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

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                snippet_hash TEXT UNIQUE,
                snippet TEXT,
                verified_fix TEXT,
                confidence REAL,
                llm_time_s REAL DEFAULT 0,
                times_retrieved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_snippet_hash ON patterns(snippet_hash)")
        self._conn.commit()
        # Load existing patterns into memory
        for row in self._conn.execute("SELECT id, snippet, verified_fix, confidence, llm_time_s, times_retrieved FROM patterns").fetchall():
            pid, snippet, fix, conf, llm_t, times = row
            self._fix_cache[pid] = {
                "id": pid, "verified_fix": fix, "confidence": conf,
                "llm_time_s": llm_t, "times_retrieved": times
            }
            if snippet:
                for h in self._compute_trigrams(snippet):
                    self._trigram_index.setdefault(h, set()).add(pid)

    def _db_execute(self, sql: str, params=()):
        return self._conn.execute(sql, params)

    def _db_commit(self):
        self._conn.commit()
