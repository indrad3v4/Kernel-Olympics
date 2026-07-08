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
from typing import Dict, Optional, Any
from dataclasses import dataclass

# ── Spec directory (relative to this file) ──────────────────────────
_SPEC_DIR = Path(__file__).parent / "specs"


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
    spec_name: Optional[str] = None


class VerificationAgent:
    """Verifies ported HIP kernels by compiling and running on AMD GPU."""

    def __init__(self, docker_image: str = "rocm/dev-ubuntu-22.04:latest"):
        self.docker_image = docker_image
        self.offload_arch = os.environ.get("AMD_OFFLOAD_ARCH", "gfx942")
        self._hipcc_available = self._check_hipcc()
        self._spec_cache: Dict[str, dict] = {}

        # Persistent build directory — reuse across verify() calls
        env_build_dir = os.environ.get("VERIFIER_BUILD_DIR")
        if env_build_dir:
            self.build_dir = Path(env_build_dir)
            self.build_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.build_dir = Path(tempfile.mkdtemp(prefix="verifier_build_"))

    # ── hipcc warmup ────────────────────────────────────────────────

    def warmup(self) -> str:
        """Pre-warm hipcc by compiling a trivial kernel.

        hipcc has noticeable cold-start latency on first invocation
        (LLVM IR parsing, device-lib linking, etc.).  Calling
        *warmup()* once before the first real verification skips
        that penalty for the first real compile.

        The compiled binary is also executed briefly to warm GPU
        driver / ROCr runtime state.
        """
        if not self._hipcc_available:
            return "hipcc not available — skipping warmup."

        warmup_src = (
            '#include <iostream>\n'
            '#include <hip/hip_runtime.h>\n'
            '\n'
            '__global__ void _verifier_warmup_kernel() {\n'
            '    // Intentionally empty — pre-warms hipcc compilation pipeline\n'
            '}\n'
            '\n'
            'int main() {\n'
            '    _verifier_warmup_kernel<<<1, 1>>>();\n'
            '    hipDeviceSynchronize();\n'
            '    std::cout << "warmup_ok" << std::endl;\n'
            '    return 0;\n'
            '}\n'
        )

        kernel_name = "_verifier_warmup"
        src_file = self.build_dir / f"{kernel_name}.hip.cpp"
        src_file.write_text(warmup_src)

        compile_ok, compile_out = self._compile(src_file, self.build_dir, kernel_name)

        if compile_ok:
            self._run(self.build_dir, kernel_name)

        # Clean up warmup artifacts
        for f in self.build_dir.glob(f"{kernel_name}*"):
            try:
                f.unlink()
            except OSError as e:
                print(f"║ ⚠️ Warmup cleanup failed: {e}")
                pass
        try:
            src_file.unlink()
        except OSError as e:
            print(f"║ ⚠️ Warmup cleanup (src_file) failed: {e}")
            pass

        return compile_out

    # ── Spec loading ────────────────────────────────────────────────

    def list_specs(self) -> list[str]:
        """Return all available kernel spec names (without .json)."""
        if not _SPEC_DIR.is_dir():
            return []
        return sorted(f.stem for f in _SPEC_DIR.glob("*.json"))

    def load_spec(self, kernel_name: str) -> Optional[dict]:
        """Load a kernel spec by name from the specs/ directory."""
        if kernel_name in self._spec_cache:
            return self._spec_cache[kernel_name]

        spec_path = _SPEC_DIR / f"{kernel_name}.json"
        if not spec_path.exists():
            return None

        with open(spec_path) as f:
            spec = json.load(f)

        # Resolve reference_output relative to repo root
        ref = spec.get("reference_output", "")
        if ref and not Path(ref).is_absolute():
            # Walk up from specs/ to find the repo root
            repo_root = _SPEC_DIR.parent.parent.parent  # src/verification/specs → repo
            resolved = repo_root / ref
            if resolved.exists():
                spec["_reference_path"] = str(resolved)

        self._spec_cache[kernel_name] = spec
        return spec

    # ── Spec-driven harness generation ──────────────────────────────

    def _generate_harness(self, kernel_name: str, test_input: str,
                          ported_kernel_source: str) -> str:
        """
        Generate a test harness driven by the kernel's JSON spec.

        If a spec exists for *kernel_name* it is used to produce the
        correct launch configuration, parameter list, input setup, and
        output readback.  Otherwise falls back to the legacy heuristic
        (auto-detect kernel function, generic 256‑element harness).
        """
        # Try loading the spec
        spec = self.load_spec(kernel_name)

        if spec is not None:
            return self._harness_from_spec(spec, ported_kernel_source)

        # Fallback: legacy heuristic
        import re
        match = re.search(r'__global__\s+void\s+(\w+)\s*\(', ported_kernel_source)
        actual_kernel = match.group(1) if match else f"{kernel_name}_kernel"
        return self._legacy_harness(actual_kernel, ported_kernel_source)

    def _harness_from_spec(self, spec: dict, ported_kernel_source: str) -> str:
        """Build a C++ harness from a kernel spec dictionary."""
        fn = spec.get("kernel_function", "kernel")
        params = spec.get("params", [])
        launch = spec.get("launch", {"grid": {"x": 1}, "block": {"x": 64}})
        inp = spec.get("input_setup", {})
        out = spec.get("output_readback", {"count": 4, "element_type": "float",
                                             "format": "float_per_line"})

        grid = launch["grid"]
        block = launch["block"]
        gx = grid.get("x", 1); gy = grid.get("y", 1); gz = grid.get("z", 1)
        bx = block.get("x", 64); by = block.get("y", 1); bz = block.get("z", 1)
        total_threads = gx * gy * gz * bx * by * bz

        readback_count = out.get("count", 4)
        elem_type = out.get("element_type", "float")
        is_int = elem_type.startswith("int")

        # Input data size
        input_count = inp.get("count", total_threads)
        default_val = inp.get("default_value", 0.0)
        linear_ramp = inp.get("linear_ramp", False)

        # Collect scalar argument overrides
        scalar_overrides = {}
        kernel_args_override = spec.get("kernel_args_override", "")
        if kernel_args_override:
            vals = [v.strip() for v in kernel_args_override.split(",")]
            scalar_idx = 0
            for p in params:
                if p["direction"] == "scalar":
                    if scalar_idx < len(vals):
                        scalar_overrides[p["name"]] = vals[scalar_idx]
                    scalar_idx += 1

        # Build input setup lines
        input_lines = []
        if linear_ramp:
            input_lines.append(
                f"    std::vector<{elem_type}> input({input_count});"
            )
            input_lines.append(
                f"    for (int i = 0; i < {input_count}; i++) input[i] = static_cast<{elem_type}>(i + 1);"
            )
        else:
            input_lines.append(
                f"    std::vector<{elem_type}> input({input_count}, "
                f"static_cast<{elem_type}>({default_val}));"
            )

        # Build output buffer
        output_lines = [
            f"    std::vector<{elem_type}> output({input_count}, 0);"
        ]

        # Build variable declarations for pointer params
        decl_lines = []
        alloc_lines = []
        memcpy_h2d_lines = []
        memcpy_d2h_lines = []
        free_lines = []
        kernel_args = []
        scalar_init_lines = []

        for p in params:
            name = p["name"]
            ptype = p["type"]
            direction = p["direction"]

            if direction == "scalar":
                if name in scalar_overrides:
                    scalar_init_lines.append(
                        f"    int {name} = {scalar_overrides[name]};"
                    )
                else:
                    val = p.get("value", input_count)
                    scalar_init_lines.append(
                        f"    int {name} = {val};"
                    )
                kernel_args.append(name)
            else:
                size_expr = p.get("size_expr", str(input_count))
                decl_lines.append(f"    {ptype} d_{name};")
                alloc_lines.append(
                    f"    hipMalloc(&d_{name}, {size_expr} * sizeof({elem_type}));"
                )
                if direction == "in":
                    memcpy_h2d_lines.append(
                        f"    hipMemcpy(d_{name}, {name}.data(), "
                        f"{size_expr} * sizeof({elem_type}), hipMemcpyHostToDevice);"
                    )
                else:
                    memcpy_d2h_lines.append(
                        f"    hipMemcpy({name}.data(), d_{name}, "
                        f"{size_expr} * sizeof({elem_type}), hipMemcpyDeviceToHost);"
                    )
                free_lines.append(f"    hipFree(d_{name});")
                kernel_args.append(f"d_{name}")

        # Dynamic shared memory
        dynamic_smem = spec.get("dynamic_shared_mem", 0)
        dsmem_suffix = f", {dynamic_smem}" if dynamic_smem else ""

        # Build print / output format
        print_lines = []
        if out.get("format") == "int_per_line":
            for i in range(readback_count):
                print_lines.append(
                    f'        std::cout << output[{i}] << std::endl;'
                )
        else:
            for i in range(readback_count):
                print_lines.append(
                    f'        std::cout << std::fixed << output[{i}] << std::endl;'
                )

        # Build parameter and kernel-call string
        kernel_call_args = ", ".join(kernel_args)

        # Assemble full harness
        lines = [
            '#include <iostream>',
            '#include <iomanip>',
            '#include <vector>',
            '#include <hip/hip_runtime.h>',
            '#include <cmath>',
            '',
            ported_kernel_source,
            '',
            'int main() {',
        ]

        for line in scalar_init_lines:
            lines.append(line)
        for line in input_lines:
            lines.append(line)
        for line in output_lines:
            lines.append(line)
        for line in decl_lines:
            lines.append(line)
        for line in alloc_lines:
            lines.append(line)
        for line in memcpy_h2d_lines:
            lines.append(line)

        lines.append(
            f'    {fn}<<<dim3({gx},{gy},{gz}), dim3({bx},{by},{bz}){dsmem_suffix}>>>({kernel_call_args});'
        )
        lines.append('    hipDeviceSynchronize();')

        for line in memcpy_d2h_lines:
            lines.append(line)

        for line in print_lines:
            lines.append(line)

        for line in free_lines:
            lines.append(line)
        lines.append('    return 0;')
        lines.append('}')

        return '\n'.join(lines)

    def _legacy_harness(self, actual_kernel: str, ported_kernel_source: str) -> str:
        """Legacy fallback harness (hardcoded 256-element generic harness)."""
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

    {actual_kernel}<<<4, 64>>>(d_input, d_output, N);
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

    # ── Verify method (using persistent build directory) ────────────

    def verify(self, hip_source: str, cuda_reference_output: str = "",
               test_input: str = "", kernel_name: str = "test_kernel") -> Dict:
        """
        Verify a ported HIP kernel:
        1. Write source to persistent build directory
        2. Compile with hipcc
        3. Run executable
        4. Diff output against CUDA reference

        Uses ``self.build_dir`` (set via the ``VERIFIER_BUILD_DIR`` env
        var or an auto-created temporary directory) so artifacts survive
        across calls for inspection and to avoid re-creating the build
        tree each time.
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
            "benchmark_us": None,
            "spec_used": None
        }

        # Record spec if one exists
        spec = self.load_spec(kernel_name)
        if spec:
            result["spec_used"] = spec.get("kernel_name", kernel_name)

        # Persistent build directory — per-kernel subdirectory for isolation
        kernel_build_dir = self.build_dir / kernel_name
        kernel_build_dir.mkdir(parents=True, exist_ok=True)

        # Write source
        src_file = kernel_build_dir / f"{kernel_name}.hip.cpp"
        src_file.write_text(hip_source)

        # Generate spec-driven harness
        harness = self._generate_harness(kernel_name, test_input, hip_source)
        harness_file = kernel_build_dir / f"test_{kernel_name}.cpp"
        harness_file.write_text(harness)

        # Step 1: Compile
        compile_ok, compile_out = self._compile(harness_file, kernel_build_dir, kernel_name)
        result["compile_success"] = compile_ok
        result["compile_output"] = compile_out[:1000] if compile_out else ""

        if not compile_ok:
            manual_dir = Path.cwd() / "ported_kernels"
            manual_dir.mkdir(parents=True, exist_ok=True)
            manual_path = manual_dir / f"{kernel_name}.hip.cpp"
            try:
                import shutil
                shutil.copy2(harness_file, manual_path)
                compile_out += f"\n\n⚠️ Ported kernel saved to: {manual_path}\n"
                compile_out += f"   Compile manually: hipcc -o /tmp/{kernel_name} {manual_path} -std=c++17 -O2\n"
                compile_out += f"   Run: /tmp/{kernel_name}"
            except Exception as e:
                print(f"║ ⚠️ Failed to save ported kernel to {manual_path}: {e}")
                pass
            result["passed"] = False
            return result

        # Step 2: Run
        run_ok, run_output, benchmark = self._run(kernel_build_dir, kernel_name)
        result["run_success"] = run_ok
        result["run_output"] = run_output[:1000] if run_output else ""
        result["benchmark_us"] = benchmark

        if not run_ok:
            result["passed"] = False
            return result

        # Step 3: Diff against reference (try spec path first, fallback to param)
        ref_text = cuda_reference_output
        spec_ref_path = spec.get("_reference_path") if spec else None
        if spec_ref_path and not ref_text:
            try:
                ref_text = Path(spec_ref_path).read_text()
            except OSError as e:
                print(f"║ ⚠️ Could not read spec reference file: {e}")
                pass

        if ref_text:
            diff_ok, diff_report = self._diff(run_output, ref_text)
            result["output_match"] = diff_ok
            result["diff_report"] = diff_report[:500] if diff_report else ""
            result["passed"] = diff_ok
        else:
            result["output_match"] = True
            result["diff_report"] = "No reference — marked pass (compiled + ran successfully)"
            result["passed"] = True

        return result

    # ── Compile / Run / Diff helpers (unchanged from original) ──────

    def _compile(self, harness_file: Path, build_dir: Path, kernel_name: str) -> tuple:
        """Compile HIP kernel with hipcc."""
        output_bin = build_dir / kernel_name

        if self._hipcc_available:
            try:
                if self._hipcc_path == "hipcc":
                    cmd_line = f"hipcc -o {output_bin} {harness_file} -std=c++17 -O2 --offload-arch={self.offload_arch}"
                    result = subprocess.run(
                        cmd_line, shell=True,
                        capture_output=True, text=True, timeout=60,
                        cwd=str(build_dir)
                    )
                else:
                    result = subprocess.run(
                        [self._hipcc_path, "-o", str(output_bin), str(harness_file),
                         "-std=c++17", "-O2", f"--offload-arch={self.offload_arch}"],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(build_dir)
                    )
                return result.returncode == 0, result.stdout + result.stderr
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                manual_dir = Path.cwd() / "ported_kernels"
                manual_dir.mkdir(parents=True, exist_ok=True)
                manual_path = manual_dir / f"{kernel_name}.hip.cpp"
                try:
                    import shutil
                    shutil.copy2(harness_file, manual_path)
                except Exception as e:
                    print(f"║ ⚠️ Failed to copy kernel to {manual_path} during compile fallback: {e}")
                    pass
                return False, (
                    f"hipcc compilation failed. Ported kernel saved to {manual_path}.\n"
                    f"To compile manually: hipcc -o /tmp/{kernel_name} {manual_path} -std=c++17 -O2\n"
                    f"Then run: /tmp/{kernel_name}"
                )
        else:
            manual_dir = Path.cwd() / "ported_kernels"
            manual_dir.mkdir(parents=True, exist_ok=True)
            manual_path = manual_dir / f"{kernel_name}.hip.cpp"
            try:
                import shutil
                shutil.copy2(harness_file, manual_path)
            except Exception as e:
                print(f"║ ⚠️ Failed to copy kernel to {manual_path} in else branch: {e}")
                pass
            msg = (
                f"hipcc not found in subprocess. Ported kernel saved to {manual_path}.\n"
                f"To compile manually: hipcc -o /tmp/{kernel_name} {manual_path} -std=c++17 -O2\n"
                f"Then run: /tmp/{kernel_name}"
            )
            return False, msg

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
            print("║ ⚠️ Diff: could not parse output lines as floats — falling back to textual diff")
            pass

        # Show diff
        import difflib
        diff = difflib.unified_diff(expected_lines, actual_lines,
                                    fromfile='expected', tofile='actual', lineterm='')
        return False, "\n".join(list(diff)[:20])

    # ── hipcc detection (unchanged from original) ───────────────────

    def _check_hipcc(self) -> bool:
        """Check if hipcc is available on this system (any known path)."""
        candidates = ["hipcc", "/opt/rocm/bin/hipcc", "/opt/rocm/lib/llvm/bin/hipcc",
                       "/opt/rocm-7.2.1/bin/hipcc", "/opt/rocm-7.2.1/lib/llvm/bin/hipcc",
                       "/usr/bin/hipcc"]
        for cmd in candidates:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self._hipcc_path = cmd
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"║ ⚠️ hipcc candidate '{cmd}' not available: {e}")
                continue
        try:
            result = subprocess.run("which hipcc 2>/dev/null || command -v hipcc 2>/dev/null",
                                    shell=True, capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                self._hipcc_path = result.stdout.strip()
                return True
        except subprocess.TimeoutExpired:
            print("║ ⚠️ 'which hipcc' timed out — continuing")
            pass
        try:
            result = subprocess.run("hipcc --version", shell=True,
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._hipcc_path = "hipcc"
                return True
        except subprocess.TimeoutExpired:
            print("║ ⚠️ 'hipcc --version' (shell) timed out — continuing")
            pass
        import glob
        for match in glob.glob("/opt/rocm*/**/hipcc", recursive=True):
            self._hipcc_path = match
            return True
        return False
