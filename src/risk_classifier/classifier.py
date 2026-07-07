"""
Risk Classifier — pattern-matches CUDA source for known danger signals.

Rule-based (not LLM) for speed, determinism, and reliability.
Detects warp(32)→wavefront(64) divergence patterns.

Patterns detected (start with 3-5, expand later):
P1: Hardcoded warp size (32) in shared memory or loop bounds
P2: __shfl_down_sync with offset > 1 (assumes 32-lane warp)
P3: __shfl_xor_sync (butterfly) patterns
P4: warp-size-dependent shared memory tiling
P5: __syncwarp() usage (semantics differ on AMD)
P6: __shfl_up_sync (upward lane shift, assumes 32-lane warp)
P7: __activemask() (32-bit lane mask, can't represent 64-lane wavefront)
"""

import re
from typing import Dict, List, Tuple


# Known danger patterns as (name, regex_pattern, description)
DANGER_PATTERNS = [
    (
        "warp_size_constant",
        r"32(?=\s*[\;\}\]\)])|warp_size|WARP_SIZE|warpSz",
        "Hardcoded constant 32 (NVIDIA warp size) used — AMD wavefront is 64"
    ),
    (
        "shfl_down_sync",
        r"__shfl_down_sync\s*\([^)]*,\s*[^,]+,\s*(16|8)\s*\)",
        "__shfl_down_sync with offset 16 or 8 — assumes 32-lane warp, "
        "will silently skip half the lanes on wavefront64"
    ),
    (
        "shfl_xor_sync",
        r"__shfl_xor_sync\s*\(",
        "__shfl_xor_sync (butterfly) — warp-size-dependent, needs wavefront adaptation"
    ),
    (
        "shared_mem_warp_tiling",
        r"__shared__\s+\w+\s*\[32\]|shared_mem\[\d*\s*\]\[\s*32\s*\]|tile\[\s*32\s*\]",
        "Shared memory sized to 32 (warp) — needs 64 for wavefront-aware tiling"
    ),
    (
        "syncwarp",
        r"__syncwarp\s*\(",
        "__syncwarp() — semantics differ between CUDA and HIP; use __syncthreads() instead"
    ),
    (
        "shfl_up_sync",
        r"__shfl_up_sync\s*\(",
        "__shfl_up_sync (upward lane shift) — warp-size-dependent offset, "
        "will silently shift from the wrong lane on wavefront64"
    ),
    (
        "activemask",
        r"__activemask\s*\(\s*\)",
        "__activemask() returns a 32-bit lane mask — cannot represent all 64 lanes "
        "of an AMD wavefront, so lanes 32-63 are silently dropped from the mask"
    ),
]


class RiskClassifier:
    """Rule-based classifier for CUDA→ROCm portability risk."""

    def __init__(self):
        self.patterns = [(name, re.compile(pattern), desc) for name, pattern, desc in DANGER_PATTERNS]
        self.pattern_counters = {name: 0 for name, _, _ in DANGER_PATTERNS}
        self.total_scans = 0

    def classify(self, source_code: str, filepath: str = "") -> Dict:
        """
        Classify a CUDA kernel's portability risk.
        Returns: {file, risk_level, findings, matched_patterns, total_patterns_checked}
        """
        self.total_scans += 1
        findings = []
        matched_count = 0

        for name, compiled_re, description in self.patterns:
            matches = list(compiled_re.finditer(source_code))
            if matches:
                matched_count += 1
                self.pattern_counters[name] += len(matches)
                for m in matches:
                    # Get surrounding context (2 lines before and after)
                    lines = source_code.splitlines()
                    line_num = source_code[:m.start()].count('\n') + 1
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context = "\n".join(lines[context_start:context_end])

                    findings.append({
                        "pattern": name,
                        "description": description,
                        "line": line_num,
                        "matched_text": m.group()[:100],
                        "context": context,
                        "severity": self._severity(name)
                    })

        # Determine overall risk level
        if not findings:
            risk_level = "green"
        elif any(f["severity"] == "high" for f in findings):
            risk_level = "red"
        elif any(f["severity"] == "medium" for f in findings):
            risk_level = "yellow"
        else:
            risk_level = "green"

        return {
            "file": filepath,
            "risk_level": risk_level,
            "findings": findings,
            "matched_pattern_count": matched_count,
            "total_patterns_checked": len(self.patterns),
            "patterns_memory": dict(self.pattern_counters)
        }

    def classify_batch(self, file_sources: Dict[str, str]) -> List[Dict]:
        """Classify multiple files. Input: {filepath: source_code}."""
        return [self.classify(code, fp) for fp, code in file_sources.items()]

    def _severity(self, pattern_name: str) -> str:
        """Determine severity of a pattern match."""
        high_severity = {"shfl_down_sync", "shfl_xor_sync", "shfl_up_sync"}
        medium_severity = {"warp_size_constant", "shared_mem_warp_tiling", "activemask"}
        if pattern_name in high_severity:
            return "high"
        elif pattern_name in medium_severity:
            return "medium"
        return "low"

    def reset_counters(self):
        """Reset pattern counters."""
        self.pattern_counters = {name: 0 for name, _, _ in DANGER_PATTERNS}
        self.total_scans = 0


if __name__ == "__main__":
    # Quick test with the sample kernel
    source = open("sample_kernels/cuda/warp_reduce.cu").read()
    classifier = RiskClassifier()
    result = classifier.classify(source, "sample_kernels/cuda/warp_reduce.cu")
    print(f"Risk level: {result['risk_level']}")
    print(f"Findings: {len(result['findings'])}")
    for f in result['findings']:
        print(f"  [{f['severity']}] Line {f['line']}: {f['pattern']} — {f['description'][:60]}...")
