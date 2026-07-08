"""
Scanner — hipify-clang dry-run wrapper.

Input: path to 1-3 CUDA .cu files
Tool: hipify-clang in dry-run/report mode
Output: structured JSON per file with hipify coverage and flagged lines
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


class Scanner:
    """Wraps hipify-clang dry-run to assess portability risk."""

    def __init__(self, hipify_bin: str = "hipify-clang"):
        self.hipify_bin = hipify_bin

    def scan(self, filepath: str) -> Dict:
        """Run hipify-clang in dry-run mode on a single file."""
        path = Path(filepath)
        if not path.exists():
            return {"file": filepath, "error": "file not found", "hipify_coverage_pct": 0, "flagged_lines": []}

        try:
            result = subprocess.run(
                [self.hipify_bin, str(path), "--dry-run"],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            total_lines = len(path.read_text(encoding="utf-8").splitlines())
            return {"file": filepath, "error": str(e), "hipify_coverage_pct": 0,
                    "total_lines": total_lines, "flagged_lines": []}

        # Parse hipify output for coverage info
        total_lines = len(path.read_text(encoding="utf-8").splitlines())
        unconverted = self._count_unconverted(output)
        coverage = max(0, ((total_lines - unconverted) / total_lines) * 100) if total_lines > 0 else 0

        # Extract flagged/warning lines
        warnings = self._extract_warnings(output)

        return {
            "file": filepath,
            "hipify_coverage_pct": round(coverage, 1),
            "unconverted_lines": unconverted,
            "total_lines": total_lines,
            "flagged_lines": warnings,
            "raw_output": output[:2000]  # truncate for storage
        }

    def scan_batch(self, filepaths: List[str]) -> List[Dict]:
        """Scan multiple files."""
        return [self.scan(f) for f in filepaths]

    def _count_unconverted(self, output: str) -> int:
        """Count lines hipify flagged as unconverted."""
        lines = output.splitlines()
        count = 0
        for line in lines:
            if any(marker in line.lower() for marker in ["unconverted", "warning:", "error:", "unable to"]):
                count += 1
        return count

    def _extract_warnings(self, output: str) -> List[Dict]:
        """Parse hipify warnings into structured format."""
        warnings = []
        pattern = re.compile(r'(?:warning|error):\s*(.+?)(?:\s*\[(\d+)\])?', re.IGNORECASE)
        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                warnings.append({
                    "message": match.group(1).strip(),
                    "line": int(match.group(2)) if match.group(2) else None
                })
        return warnings


if __name__ == "__main__":
    # Quick test
    scanner = Scanner()
    result = scanner.scan("sample_kernels/cuda/warp_reduce.cu")
    print(json.dumps(result, indent=2))
