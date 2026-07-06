"""
Pattern Memory — vector store for verified CUDA→ROCm migration patterns.

Stores: pattern_embedding, original_snippet, verified_fix, confidence, verification_run_id
Retrieval: on new red-flagged kernel, search for nearest match above threshold.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PatternMemory:
    """Simple vector store for verified migration patterns.
    
    Uses normalized code signatures as lightweight embeddings.
    For hackathon: in-memory + JSON persistence. 
    Production: replace with Chroma/FAISS.
    """

    def __init__(self, storage_path: str = "data/pattern_memory.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: List[Dict] = []
        self._load()

    def store(self, pattern_snippet: str, verified_fix: str, confidence: float,
              verification_run_id: str = "") -> str:
        """Store a verified pattern fix."""
        pattern_id = hashlib.sha256(pattern_snippet.encode()).hexdigest()[:12]
        
        entry = {
            "id": pattern_id,
            "pattern_signature": self._compute_signature(pattern_snippet),
            "original_snippet": pattern_snippet[:500],
            "verified_fix": verified_fix[:500],
            "confidence": round(confidence, 2),
            "verification_run_id": verification_run_id,
            "times_retrieved": 0
        }
        
        # Check if already exists (update confidence)
        for i, p in enumerate(self.patterns):
            if p["id"] == pattern_id:
                # Update: new confidence is max of old and new
                self.patterns[i]["confidence"] = max(p["confidence"], confidence)
                self.patterns[i]["times_retrieved"] = p.get("times_retrieved", 0) + 1
                self._save()
                return pattern_id
        
        self.patterns.append(entry)
        self._save()
        return pattern_id

    def retrieve(self, query_snippet: str, threshold: float = 0.7) -> Optional[Dict]:
        """Find nearest matching pattern above similarity threshold."""
        query_sig = self._compute_signature(query_snippet)
        
        best_match = None
        best_similarity = 0.0
        
        for pattern in self.patterns:
            similarity = self._cosine_similarity(query_sig, pattern["pattern_signature"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        
        if best_match and best_similarity >= threshold:
            best_match["similarity"] = round(best_similarity, 3)
            best_match["times_retrieved"] = best_match.get("times_retrieved", 0) + 1
            self._save()
            return best_match
        
        return None

    def count(self) -> int:
        """Number of stored patterns."""
        return len(self.patterns)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.patterns:
            return {"total_patterns": 0, "avg_confidence": 0, "total_retrievals": 0}
        return {
            "total_patterns": len(self.patterns),
            "avg_confidence": round(sum(p["confidence"] for p in self.patterns) / len(self.patterns), 2),
            "total_retrievals": sum(p.get("times_retrieved", 0) for p in self.patterns)
        }

    def _compute_signature(self, code: str) -> List[float]:
        """Compute a lightweight normalized code signature.
        
        Uses character-level n-gram frequencies + structural features.
        NOT a real embedding — good enough for hackathon MVP.
        """
        # Structural features
        lines = code.splitlines()
        n_lines = len(lines)
        
        # Count various code features as a signature vector
        features = [
            n_lines,
            code.count("__shfl"),
            code.count("__syncthreads"),
            code.count("shared"),
            code.count("__shared__"),
            code.count("threadIdx"),
            code.count("blockIdx"),
            code.count("for"),
            code.count("if"),
            code.count("return"),
            code.count("#define"),
            code.count("template"),
            code.count("int "),
            code.count("float "),
            code.count("double "),
            code.count("sizeof"),
            code.count("volatile"),
            sum(1 for c in code if c == ';'),
            sum(1 for c in code if c == '{'),
            sum(1 for c in code if c == '}'),
        ]
        
        # Normalize
        arr = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norm) if norm > 0 else 0.0

    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump({"patterns": self.patterns, "version": 1}, f, indent=2)

    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
                self.patterns = data.get("patterns", [])
                # Update times_retrieved from load
                for p in self.patterns:
                    p["times_retrieved"] = p.get("times_retrieved", 0)
