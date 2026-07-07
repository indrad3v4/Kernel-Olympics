#verifier.py

"""
Verification Agent — Docker-based compile + run + diff on AMD Developer Cloud.

Environment: Docker container on AMD Developer Cloud
Steps: compile ported HIP code → run against test input → diff vs CUDA reference
Output: pass/fail + diff report
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    passed: bool
    compile_success: bool
    run_success: bool
    output_match: bool
    compile_output: str
    run_output: str
    diff_report: str
    benchmark_us: Optional[float] = None


class VerificationAgent:
    """Verifies ported HIP kernels by compiling and running on AMD GPU."""

    def __init__(self, docker_image: str = "rocm/dev-ubuntu-22.04:latest"):
        self.docker_image = docker_image
        self._hipcc_available = self._check_hipcc()

    def verify(self, hip_source: str, cuda_reference_output: str,
               test_input: str = "", kernel_name: str = "test_kernel") -> Dict:
        """
        Verify a ported HIP kernel:
        1. Write source to temp file
        2. Compile with hipcc
        3. Run executable
        4. Diff output against CUDA reference
        """
        result = {
            "kernel": kernel_name,
            "compile_success": False,
            "run_success": False,
            "output_match": False,
            "passed": False,
            "compile_output": "",
            "run_output": "",
            "diff_report": "",
            "benchmark_us": None
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write source
            src_file = tmp_path / f"{kernel_name}.hip.cpp"
            src_file.write_text(hip_source)

            # Write test harness if not provided
            harness = self._generate_harness(kernel_name, test_input, hip_source)
            harness_file = tmp_path / f"test_{kernel_name}.cpp"
            harness_file.write_text(harness)

            # Step 1: Compile
            compile_ok, compile_out = self._compile(harness_file, tmp_path, kernel_name)
            result["compile_success"] = compile_ok
            result["compile_output"] = compile_out[:1000] if compile_out else ""

            if not compile_ok:
                result["passed"] = False
                return result

            # Step 2: Run
            run_ok, run_output, benchmark = self._run(tmp_path, kernel_name)
            result["run_success"] = run_ok
            result["run_output"] = run_output[:1000] if run_output else ""
            result["benchmark_us"] = benchmark

            if not run_ok:
                result["passed"] = False
                return result

            # Step 3: Diff against reference
            diff_ok, diff_report = self._diff(run_output, cuda_reference_output)
            result["output_match"] = diff_ok
            result["diff_report"] = diff_report[:500] if diff_report else ""
            result["passed"] = diff_ok

        return result

    def _compile(self, harness_file: Path, build_dir: Path, kernel_name: str) -> tuple:
        """Compile HIP kernel with hipcc."""
        output_bin = build_dir / kernel_name

        if self._hipcc_available:
            try:
                result = subprocess.run(
                    ["hipcc", "-o", str(output_bin), str(harness_file),
                     "-std=c++17", "-O2", "--offload-arch=gfx942"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(build_dir)
                )
                return result.returncode == 0, result.stdout + result.stderr
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return False, str(e)
        else:
            # hipcc not available — return a descriptive message
            return False, (
                "hipcc not found locally. This is expected — compilation happens "
                "on AMD Developer Cloud GPU instance.\n"
                "In production: ssh to AMD instance → compile with hipcc → return result.\n"
                "For demo: we provide a pre-recorded compilation + execution log."
            )

    def _run(self, build_dir: Path, kernel_name: str) -> tuple:
        """Run the compiled HIP kernel."""
        binary = build_dir / kernel_name
        if binary.exists() and os.access(binary, os.X_OK):
            try:
                import time
                start = time.perf_counter()
                result = subprocess.run(
                    [str(binary)], capture_output=True, text=True, timeout=30,
                    cwd=str(build_dir)
                )
                elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
                output = result.stdout + result.stderr
                return result.returncode == 0, output, round(elapsed, 2)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return False, str(e), None
        return False, "Binary not found — compile step may have failed.", None

    def _diff(self, actual_output: str, expected_output: str) -> tuple:
        """Compare actual output against CUDA reference output."""
        if not actual_output or not expected_output:
            return False, "Missing output data for comparison."

        actual_lines = actual_output.strip().splitlines()
        expected_lines = expected_output.strip().splitlines()

        if actual_lines == expected_lines:
            return True, "Outputs match exactly (byte-for-byte)."

        # Try floating-point tolerant diff
        try:
            actual_floats = [float(l) for l in actual_lines if l.strip()]
            expected_floats = [float(l) for l in expected_lines if l.strip()]
            if len(actual_floats) == len(expected_floats):
                max_diff = max(abs(a - e) for a, e in zip(actual_floats, expected_floats))
                if max_diff < 1e-5:
                    return True, f"Outputs match within tolerance (max diff: {max_diff:.2e})."
                return False, f"Outputs differ (max diff: {max_diff:.4f})."
        except ValueError:
            pass

        # Show diff
        import difflib
        diff = difflib.unified_diff(expected_lines, actual_lines,
                                    fromfile='expected', tofile='actual', lineterm='')
        return False, "\n".join(list(diff)[:20])

    def _generate_harness(self, kernel_name: str, test_input: str, ported_kernel_source: str) -> str:
        """Generate a test harness that wraps the REAL ported kernel."""
        return f"""
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <cmath>

{ported_kernel_source}

int main() {{
    const int N = 256;
    std::vector<float> input(N, 1.0f);
    std::vector<float> output(N, 0.0f);

    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));

    hipMemcpy(d_input, input.data(), N * sizeof(float), hipMemcpyHostToDevice);

    warp_reduce_kernel<<<4, 64>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(output.data(), d_output, 4 * sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {{
        std::cout << output[i] << std::endl;
    }}

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}}
"""

    def _check_hipcc(self) -> bool:
        """Check if hipcc is available on this system."""
        try:
            result = subprocess.run(["which", "hipcc"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except FileNotFoundError:
            return False