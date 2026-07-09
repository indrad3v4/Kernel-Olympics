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
        src_file.write_text(warmup_src, encoding="utf-8")

        compile_ok, compile_out, _log_path = self._compile(src_file, self.build_dir, kernel_name)

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

        with open(spec_path, encoding="utf-8") as f:
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
                          ported_kernel_source: str) -> tuple:
        """
        Generate a test harness driven by the kernel's JSON spec.

        If a spec exists for *kernel_name* it is used to produce the
        correct launch configuration, parameter list, input setup, and
        output readback.  Otherwise falls back to the legacy heuristic
        (auto-detect kernel function, generic 256‑element harness).

        Self-contained programs (source already defining ``int main(``) are
        returned as-is — wrapping a second driver/``main`` around a complete
        program is what caused the harness to redefine ``main`` and shadow
        the program's own logic (e.g. ``nvidia_shfl_scan.cu``, a full NVIDIA
        sample, not a bare kernel snippet).

        Returns (harness_text, kernel_line_start, kernel_line_end): the
        1-indexed, inclusive line range within *harness_text* that
        corresponds to *ported_kernel_source*. Callers use this to tell
        whether a hipcc error originates in the ported code or in
        harness-authored driver code — see :meth:`_classify_error_origin`.
        """
        import re
        if re.search(r'^\s*int\s+main\s*\(', ported_kernel_source, re.MULTILINE):
            return ported_kernel_source, 1, len(ported_kernel_source.splitlines())

        # Try loading the spec
        spec = self.load_spec(kernel_name)

        if spec is not None:
            return self._harness_from_spec(spec, ported_kernel_source)

        # Fallback: legacy heuristic
        match = re.search(r'__global__\s+void\s+(\w+)\s*\(', ported_kernel_source)
        actual_kernel = match.group(1) if match else f"{kernel_name}_kernel"
        return self._legacy_harness(actual_kernel, ported_kernel_source)

    def _warn_if_legacy_harness(self, kernel_name: str, ported_kernel_source: str) -> None:
        """Print a loud warning when compilation will use the guessed generic harness.

        Mirrors the exact branch condition in :meth:`_generate_harness`: the
        legacy harness fires only when the source is not self-contained
        (no ``int main(``) and no JSON spec exists for *kernel_name*. Silent
        use of this guessed harness is what made a signature mismatch look
        like a mystery ``hipMalloc`` compile error instead of an obvious
        "no spec" warning.
        """
        import re
        if re.search(r'^\s*int\s+main\s*\(', ported_kernel_source, re.MULTILINE):
            return
        if self.load_spec(kernel_name) is not None:
            return
        print(f"║ ⚠️ No spec for '{kernel_name}' — using generic 256-element harness "
              f"(assumes a (float*, float*, int) signature launched <<<4,64>>>). "
              f"Signature may not match; add src/verification/specs/{kernel_name}.json.")

    def _classify_error_origin(self, error_line: str, kernel_start: int, kernel_end: int) -> str:
        """Classify a hipcc diagnostic as pointing into the ported kernel or the harness.

        Errors in harness-authored lines (the driver's own main/mallocs/launch
        call) are not fixable by refining the ported kernel — no amount of
        LLM feedback can fix a line the model never wrote or saw.
        """
        import re
        m = re.search(r':(\d+):\d+:\s*(?:fatal )?error:', error_line)
        if not m:
            return "unknown"
        line_no = int(m.group(1))
        return "ported_code" if kernel_start <= line_no <= kernel_end else "harness"

    def _harness_from_spec(self, spec: dict, ported_kernel_source: str) -> tuple:
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
        preamble = [
            '#include <iostream>',
            '#include <iomanip>',
            '#include <vector>',
            '#include <hip/hip_runtime.h>',
            '#include <cmath>',
            '',
        ]
        kernel_start = len(preamble) + 1  # 1-indexed line where ported_kernel_source begins
        kernel_end = kernel_start + len(ported_kernel_source.splitlines()) - 1
        lines = [
            *preamble,
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

        return '\n'.join(lines), kernel_start, kernel_end

    def _legacy_harness(self, actual_kernel: str, ported_kernel_source: str) -> tuple:
        """Legacy fallback harness (hardcoded 256-element generic harness).

        This is a *guess* about the kernel's signature (assumes
        ``(float* in, float* out, int n)``-shaped params, launched as
        ``<<<4, 64>>>``) used only when no JSON spec exists for the kernel.
        It will misfire on kernels with a different signature — see the
        warning this emits in :meth:`_generate_harness` callers.
        """
        preamble = [
            "#include <iostream>",
            "#include <vector>",
            "#include <hip/hip_runtime.h>",
            "#include <cmath>",
            "",
        ]
        kernel_start = len(preamble) + 1  # 1-indexed line where ported_kernel_source begins
        kernel_end = kernel_start + len(ported_kernel_source.splitlines()) - 1
        driver = [
            "",
            "int main() {",
            "    const int N = 256;",
            "    std::vector<float> input(N, 1.0f);",
            "    std::vector<float> output(N, 0.0f);",
            "",
            "    float *d_input, *d_output;",
            "    hipMalloc(&d_input, N * sizeof(float));",
            "    hipMalloc(&d_output, N * sizeof(float));",
            "",
            "    hipMemcpy(d_input, input.data(), N * sizeof(float), hipMemcpyHostToDevice);",
            "",
            f"    {actual_kernel}<<<4, 64>>>(d_input, d_output, N);",
            "    hipDeviceSynchronize();",
            "",
            "    hipMemcpy(output.data(), d_output, 4 * sizeof(float), hipMemcpyDeviceToHost);",
            "",
            "    for (int i = 0; i < 4; i++) {",
            "        std::cout << output[i] << std::endl;",
            "    }",
            "",
            "    hipFree(d_input);",
            "    hipFree(d_output);",
            "    return 0;",
            "}",
        ]
        harness = "\n".join([*preamble, ported_kernel_source, *driver])
        return harness, kernel_start, kernel_end

    # ── Verify method (using persistent build directory) ────────────

    def quick_compile_check(self, hip_source: str, kernel_name: str = "test_kernel",
                            on_progress=None) -> Dict:
        """Fast in-loop compilation check — no run, no diff.

        Used INSIDE the Kimi→GLM loop to feed real hipcc errors back to Kimi.
        Returns dict with compile_success, compile_output, and errors list.
        """
        if on_progress: on_progress(0, "writing source")
        kernel_build_dir = self.build_dir / f"loop_{kernel_name}"
        kernel_build_dir.mkdir(parents=True, exist_ok=True)

        src_file = kernel_build_dir / f"{kernel_name}.hip.cpp"
        src_file.write_text(hip_source, encoding="utf-8")

        if on_progress: on_progress(15, "generating harness")
        harness, kernel_start, kernel_end = self._generate_harness(kernel_name, "", hip_source)
        harness_file = kernel_build_dir / f"test_{kernel_name}.cpp"
        harness_file.write_text(harness, encoding="utf-8")
        self._warn_if_legacy_harness(kernel_name, hip_source)

        if on_progress: on_progress(30, "starting hipcc")
        compile_ok, compile_out, compile_log_path = self._compile(harness_file, kernel_build_dir, kernel_name)

        if on_progress: on_progress(100, "done" if compile_ok else "compile failed")

        # Extract error lines for concise LLM feedback
        # TRIZ #22: Throwing away — filter template noise, keep only actionable errors
        errors = []
        origins = []  # parallel to primary error lines only (not caret context lines)
        lines = compile_out.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only keep "error:" lines (skip "note:", "warning:", template traces)
            if "error:" not in stripped:
                continue
            # Skip C++ template instantiation noise
            if "in instantiation of" in stripped:
                continue
            if "required from" in stripped:
                continue
            # Keep the error line + 1 context line after (often shows the code)
            errors.append(stripped[:200])
            origins.append(self._classify_error_origin(stripped, kernel_start, kernel_end))
            # Include the next line if it shows the code caret (^~~~)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                errors.append(lines[i + 1].strip()[:200])

        # If no "error:" lines found but compile failed, grab first 3 non-empty lines
        if not errors and not compile_ok and compile_out:
            for line in lines:
                if line.strip() and "note:" not in line.strip():
                    errors.append(line.strip()[:200])
                    if len(errors) >= 3:
                        break

        # Bug 5: if every classifiable error points outside the ported kernel's
        # line range, no amount of Kimi refinement can fix it — it's the
        # harness (or spec-driven driver code) that's broken, not the port.
        known_origins = [o for o in origins if o != "unknown"]
        all_harness_origin = bool(known_origins) and all(o == "harness" for o in known_origins)

        return {
            "compile_success": compile_ok,
            "compile_output": compile_out[:2000] if compile_out else "",
            "errors": errors[:8],  # concise — LLM doesn't need 10+ error lines
            "error_origins": origins[:8],
            "all_harness_origin": all_harness_origin,
            "compile_log_path": compile_log_path,
            "kernel_name": kernel_name,
        }

    def verify(self, hip_source: str, cuda_reference_output: str = "",
               test_input: str = "", kernel_name: str = "test_kernel",
               on_progress=None) -> Dict:
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

        on_progress: optional callback(percent: int, stage: str) for live progress bar.
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
            "spec_used": None,
            "hipcc_available": self._hipcc_available,
            "hipcc_path": getattr(self, "_hipcc_path", "not found"),
        }

        # Record spec if one exists
        spec = self.load_spec(kernel_name)
        if spec:
            result["spec_used"] = spec.get("kernel_name", kernel_name)

        # Persistent build directory — per-kernel subdirectory for isolation
        kernel_build_dir = self.build_dir / kernel_name
        kernel_build_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Write source (0→10%)
        if on_progress: on_progress(5, "writing source")
        src_file = kernel_build_dir / f"{kernel_name}.hip.cpp"
        src_file.write_text(hip_source, encoding="utf-8")

        # Step 2: Generate harness (10→20%)
        if on_progress: on_progress(15, "generating test harness")
        harness, _kernel_start, _kernel_end = self._generate_harness(kernel_name, test_input, hip_source)
        harness_file = kernel_build_dir / f"test_{kernel_name}.cpp"
        harness_file.write_text(harness, encoding="utf-8")
        self._warn_if_legacy_harness(kernel_name, hip_source)

        # Step 3: Compile with hipcc (20→70%)
        if on_progress: on_progress(25, "starting hipcc compilation")
        compile_ok, compile_out, compile_log_path = self._compile(
            harness_file, kernel_build_dir, kernel_name, on_progress=on_progress
        )
        if on_progress: on_progress(70, "compilation complete" if compile_ok else "compilation failed")
        result["compile_success"] = compile_ok
        result["compile_output"] = compile_out[:1000] if compile_out else ""
        result["compile_log_path"] = compile_log_path

        if not compile_ok:
            manual_dir = Path.cwd() / "ported_kernels"
            manual_dir.mkdir(parents=True, exist_ok=True)
            manual_path = manual_dir / f"{kernel_name}.hip.cpp"
            try:
                import shutil
                shutil.copy2(harness_file, manual_path)
                compile_out += f"\n\n⚠️ Ported kernel saved to: {manual_path}\n"
                compile_out += f"   Compile manually: hipcc -o /tmp/{kernel_name} {manual_path} -std=c++17 -O2\n"
                compile_out += f"   Run: /tmp/{kernel_name}\n"
                compile_out += f"   Full compile log: {compile_log_path}"
            except Exception as e:
                print(f"║ ⚠️ Failed to save ported kernel to {manual_path}: {e}")
                pass
            result["compile_output"] = compile_out[:1000] if compile_out else ""
            if on_progress: on_progress(100, "failed — saved for manual hipcc")
            result["passed"] = False
            return result

        # Step 4: Run executable (70→90%)
        if on_progress: on_progress(75, "running compiled kernel")
        run_ok, run_output, benchmark = self._run(kernel_build_dir, kernel_name)
        if on_progress: on_progress(90, "run complete" if run_ok else "run failed")
        result["run_success"] = run_ok
        result["run_output"] = run_output[:1000] if run_output else ""
        result["benchmark_us"] = benchmark

        if not run_ok:
            if on_progress: on_progress(100, "run failed")
            result["passed"] = False
            return result

        # Step 5: Diff against reference (90→100%)
        if on_progress: on_progress(95, "diffing against CUDA reference")
        ref_text = cuda_reference_output
        spec_ref_path = spec.get("_reference_path") if spec else None
        if spec_ref_path and not ref_text:
            try:
                ref_text = Path(spec_ref_path).read_text(encoding="utf-8")
            except OSError as e:
                print(f"║ ⚠️ Could not read spec reference file: {e}")

        if ref_text:
            diff_ok, diff_report = self._diff(run_output, ref_text)
            result["output_match"] = diff_ok
            result["diff_report"] = diff_report[:500] if diff_report else ""
            result["passed"] = diff_ok
        else:
            result["output_match"] = True
            result["diff_report"] = "No reference — marked pass (compiled + ran successfully)"
            result["passed"] = True

        if on_progress: on_progress(100, "PASSED ✅" if result["passed"] else "DIFF FAILED")
        return result

    # ── Compile / Run / Diff helpers (unchanged from original) ──────

    def _write_compile_log(self, build_dir: Path, kernel_name: str, output: str) -> str:
        """Write the FULL, untruncated compiler output to disk.

        Terminal/report display always truncates for readability — this file
        is the one place the complete diagnostic text survives, so it can be
        `cat`-ed when the truncated summary isn't enough to debug a failure.
        """
        log_path = build_dir / f"{kernel_name}.compile.log"
        try:
            log_path.write_text(output, encoding="utf-8")
        except OSError as e:
            print(f"║ ⚠️ Failed to write compile log to {log_path}: {e}")
        return str(log_path)

    def _shorten_error_line(self, line: str, build_dir: Path) -> str:
        """Strip the temp build-dir prefix from a hipcc diagnostic line.

        hipcc reports paths as ``<build_dir>/<file>:<line>:<col>: error: ...``.
        The build_dir portion is a long, high-entropy temp path (e.g.
        ``/tmp/verifier_build_0x_00_mm/loop_nvidia_shfl_scan/``) that eats the
        entire character budget of any truncated display, pushing the actual
        ``error: <message>`` text off the end. Stripping it here — once, at
        the source — means every caller that later does ``line[:N]`` for
        terminal width shows the message instead of the path.
        """
        for prefix in (str(build_dir) + os.sep, str(build_dir).replace(os.sep, "/") + "/"):
            if line.startswith(prefix):
                return line[len(prefix):]
        return line

    def _compile(self, harness_file: Path, build_dir: Path, kernel_name: str,
                 on_progress=None) -> tuple:
        """Compile HIP kernel with hipcc.

        on_progress: optional callback(percent: int, stage: str) — called
        at key compilation milestones so the UI can show a 0-100% bar.

        Returns (compile_ok, compile_output, compile_log_path). compile_output
        has build-dir path prefixes stripped for readability; compile_log_path
        points to the full, untruncated, unstripped output on disk.
        """
        output_bin = build_dir / kernel_name

        if self._hipcc_available:
            try:
                if on_progress: on_progress(30, "hipcc starting")
                if self._hipcc_path == "hipcc":
                    cmd_line = f"hipcc -o {output_bin} {harness_file} -std=c++17 -O2 --offload-arch={self.offload_arch} -ferror-limit=5"
                    if on_progress: on_progress(40, "hipcc compiling (shell)")
                    result = subprocess.run(
                        cmd_line, shell=True,
                        capture_output=True, text=True, timeout=60,
                        cwd=str(build_dir)
                    )
                else:
                    if on_progress: on_progress(40, "hipcc compiling (direct)")
                    result = subprocess.run(
                        [self._hipcc_path, "-o", str(output_bin), str(harness_file),
                         "-std=c++17", "-O2", f"--offload-arch={self.offload_arch}",
                         "-ferror-limit=5"],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(build_dir)
                    )
                if on_progress: on_progress(65, "hipcc finished")
                raw_output = result.stdout + result.stderr
                log_path = self._write_compile_log(build_dir, kernel_name, raw_output)
                shortened = "\n".join(
                    self._shorten_error_line(line, build_dir) for line in raw_output.splitlines()
                )
                return result.returncode == 0, shortened, log_path
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
                msg = (
                    f"hipcc compilation failed. Ported kernel saved to {manual_path}.\n"
                    f"To compile manually: hipcc -o /tmp/{kernel_name} {manual_path} -std=c++17 -O2\n"
                    f"Then run: /tmp/{kernel_name}"
                )
                log_path = self._write_compile_log(build_dir, kernel_name, msg)
                return False, msg, log_path
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
            log_path = self._write_compile_log(build_dir, kernel_name, msg)
            return False, msg, log_path

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
        # First: try direct subprocess (fastest)
        for cmd in ["hipcc", "/opt/rocm-7.2.1/bin/hipcc", "/opt/rocm/bin/hipcc",
                     "/opt/rocm-7.2.1/lib/llvm/bin/hipcc", "/opt/rocm/lib/llvm/bin/hipcc",
                     "/usr/bin/hipcc"]:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self._hipcc_path = cmd
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        # Second: try with shell=True (handles Jupyter's PATH)
        try:
            result = subprocess.run("hipcc --version", shell=True,
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._hipcc_path = "hipcc"
                return True
        except subprocess.TimeoutExpired:
            pass

        # Third: search via shell PATH
        try:
            result = subprocess.run("command -v hipcc 2>/dev/null || which hipcc 2>/dev/null",
                                    shell=True, capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                self._hipcc_path = result.stdout.strip()
                return True
        except subprocess.TimeoutExpired:
            pass

        # Fourth: glob search
        import glob
        for match in glob.glob("/opt/rocm*/**/hipcc", recursive=True):
            self._hipcc_path = match
            return True
        return False
