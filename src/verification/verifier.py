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
import re
import sys
import atexit
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

from verification.oracle import DifferentialOracle, OracleKind

# debug_session lives at the src/ root. Seed sys.path so importing this module
# in isolation (a bare `from verification.verifier import ...` in a test) works
# the same way it does under main.py, which seeds src/ itself.
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from debug_session import DebugSession, compiler_version as _compiler_version

# ── Spec directory (relative to this file) ──────────────────────────
_SPEC_DIR = Path(__file__).parent / "specs"
_REPO_ROOT = _SPEC_DIR.parent.parent.parent  # src/verification/specs/ -> src/ -> Kernel-Olympics/


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
    run_exit_code: Optional[int] = None
    spec_name: Optional[str] = None


class VerificationAgent:
    """Verifies ported HIP kernels by compiling and running on AMD GPU."""

    def __init__(self, docker_image: str = "rocm/dev-ubuntu-22.04:latest"):
        self.docker_image = docker_image
        self.offload_arch = os.environ.get("AMD_OFFLOAD_ARCH", "gfx942")
        self._hipcc_available = self._check_hipcc()
        self._spec_cache: Dict[str, dict] = {}
        # Phase 11: Debug Mode. Always a session object — a null one until the
        # router attaches a live one — so _compile() never branches on it.
        self.debug = DebugSession.disabled()

        # Phase 12: Execution-grounded differential oracle.
        # Replaces fabricated golden files with executed references.
        self.oracle = DifferentialOracle(
            repo_root=_REPO_ROOT,
            nvcc=shutil.which("nvcc"),
        )

        # Persistent build directory — reuse across verify() calls
        env_build_dir = os.environ.get("VERIFIER_BUILD_DIR")
        if env_build_dir:
            self.build_dir = Path(env_build_dir)
            self.build_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.build_dir = Path(tempfile.mkdtemp(prefix="verifier_build_"))
            # Register cleanup for temp dirs created by mkdtemp
            atexit.register(self.cleanup)

    # ── Debug Mode plumbing ─────────────────────────────────────────

    def attach_debug_session(self, session) -> None:
        """Route this agent's compiler artifacts into *session*.

        Called by ``ModelRouter.route()`` when Debug Mode is on. The verifier
        owns the only place the exact hipcc argv exists, so it must write the
        compiler stage itself rather than hand a summary back to the router.
        """
        self.debug = session

    def detach_debug_session(self) -> None:
        """Stop recording. A verifier outlives any one translation attempt."""
        self.debug = DebugSession.disabled()

    def _iteration_hint(self, kernel_name: str) -> int:
        """Best-effort iteration number for labeling compiler artifacts.

        The verifier is not told which loop iteration it is serving. Rather than
        thread that through four call sites for a debug label, we count compiles
        per kernel — which is the same number for every purpose a reader has.
        """
        counter = getattr(self, "_compile_counts", None)
        if counter is None:
            counter = self._compile_counts = {}
        counter[kernel_name] = counter.get(kernel_name, 0) + 1
        return counter[kernel_name]

    def cleanup(self):
        """Remove the temporary build directory if it was auto-created."""
        if self.build_dir and self.build_dir.exists():
            # Only clean up if not using a user-specified VERIFIER_BUILD_DIR
            if not os.environ.get("VERIFIER_BUILD_DIR"):
                try:
                    shutil.rmtree(self.build_dir)
                except OSError as e:
                    logging.debug("Failed to clean up build dir %s: %s", self.build_dir, e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

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

    @staticmethod
    def _sizeof_ctype(elem_type: str) -> int:
        """Byte size of a C element type name, for shared-memory sizing.

        Defaults to 4 (int/float) for anything unrecognized — a conservative
        choice that never under-allocates the common 4-byte scalars.
        """
        sizes = {
            "char": 1, "unsigned char": 1, "int8_t": 1, "uint8_t": 1,
            "short": 2, "unsigned short": 2, "int16_t": 2, "uint16_t": 2,
            "int": 4, "unsigned": 4, "unsigned int": 4, "float": 4,
            "int32_t": 4, "uint32_t": 4,
            "double": 8, "long": 8, "unsigned long": 8, "long long": 8,
            "int64_t": 8, "uint64_t": 8, "size_t": 8,
        }
        return sizes.get(elem_type.strip(), 4)

    @staticmethod
    def _strip_to_device_code(source: str) -> str:
        """Return only the ``__global__``/``__device__`` function definitions.

        Defensive extractor for DEVICE_SUBSET ports: a model that ignores the
        "drop the host driver" instruction leaks ``int main()`` and host helpers
        (``CPUverify``, ``shuffle_simple_test`` …) that reference unportable SDK
        symbols. Embedding that under the synthesized harness both duplicates
        ``main`` and drags the unresolved symbols back in. Keeping only the
        device functions lets the spec harness compile the kernel it targets.

        Brace-matched so a device function body with nested blocks/strings is
        not truncated at the first ``}``. Returns "" when no device function is
        found (caller then falls back to the raw source rather than an empty
        harness).
        """
        lines = source.splitlines()
        extracted = []
        in_kernel = False
        brace_depth = 0
        seen_open = False  # a K&R signature puts '{' on the *next* line — don't
                           # close the function before its opening brace appears.
        for line in lines:
            stripped = line.lstrip()
            if not in_kernel and ("__global__" in stripped or "__device__" in stripped):
                in_kernel = True
                seen_open = False
                extracted.append(line)
                brace_depth += line.count("{") - line.count("}")
                if line.count("{"):
                    seen_open = True
                    if brace_depth <= 0:
                        in_kernel = False
                        extracted.append("")
                continue
            if in_kernel:
                extracted.append(line)
                brace_depth += line.count("{") - line.count("}")
                if line.count("{"):
                    seen_open = True
                if seen_open and brace_depth <= 0:
                    in_kernel = False
                    extracted.append("")
        return "\n".join(extracted).strip()

    @staticmethod
    def _fix_hip_intrinsics(device_source: str) -> str:
        """Replace CUDA _sync warp intrinsics with HIP non-sync variants.

        On AMD CDNA2/CDNA3 (gfx942), ``__shfl_up_sync(mask, value, delta, width)``
        with a mask that has only 1 bit set (single-wavefront block) triggers an
        LDS out-of-bounds access because lane 0 reads from lane -1. The non-sync
        variants (``__shfl_up(value, delta, width)``) use a different compiler
        code path that handles the edge case gracefully.

        Safe replacements (mask is the first arg, always dropped):
          __shfl_up_sync    → __shfl_up       (upward shuffle)
          __shfl_down_sync  → __shfl_down     (downward shuffle)
          __shfl_xor_sync   → __shfl_xor      (butterfly shuffle)
        """
        import re
        result = device_source
        # __shfl_up_sync(mask, value, delta, width) → __shfl_up(value, delta, width)
        result = re.sub(
            r'__shfl_up_sync\s*\(\s*[^,]+,\s*',
            '__shfl_up(',
            result,
        )
        # __shfl_down_sync(mask, value, delta, width) → __shfl_down(value, delta, width)
        result = re.sub(
            r'__shfl_down_sync\s*\(\s*[^,]+,\s*',
            '__shfl_down(',
            result,
        )
        # __shfl_xor_sync(mask, value, delta, width) → __shfl_xor(value, delta, width)
        result = re.sub(
            r'__shfl_xor_sync\s*\(\s*[^,]+,\s*',
            '__shfl_xor(',
            result,
        )
        if result != device_source:
            changed_count = (
                device_source.count('__shfl_up_sync(')
                + device_source.count('__shfl_down_sync(')
                + device_source.count('__shfl_xor_sync(')
            )
            print(f"║  │  🔧 HIP intrinsic fix: {changed_count} CUDA _sync call(s) "
                  f"converted to non-sync variants{'':<12}║")
        return result

    # ── Device-only proof: compile a minimal harness from just the device kernel ──

    def _try_device_only_proof(self, kernel_name: str, device_source: str,
                                 build_dir: Path, spec: Optional[dict],
                                 on_progress=None,
                                 offload_arch: Optional[str] = None) -> tuple:
        """Attempt to compile a device-only proof harness.

        Some kernels (NVIDIA SDK samples, benchmarks) carry unportable host
        code (cudaEvent, printf, CPU-side verification) but have perfectly
        valid device functions.  This method wraps just the device functions
        in a minimal proof harness and tries hipcc.

        Returns (compile_ok, compile_output, compile_log_path) — same shape
        as :meth:`_compile` so the caller handles them identically.
        """
        import re

        # ── 1. Build the proof harness from device code + spec ──
        if spec and spec.get("params"):
            proof, *_ = self._harness_from_spec(spec, device_source)
        else:
            proof = self._legacy_device_proof_harness(kernel_name, device_source)

        proof_file = build_dir / f"{kernel_name}_proof.hip.cpp"
        proof_file.write_text(proof, encoding="utf-8")

        if on_progress: on_progress(35, "device-only compile attempt")
        return self._compile(proof_file, build_dir, f"{kernel_name}_proof",
                             on_progress=on_progress, offload_arch=offload_arch)

    @staticmethod
    def _legacy_device_proof_harness(kernel_name: str, device_source: str) -> str:
        """Fallback proof harness when no spec is available.

        Creates a minimal ``int``-based harness that allocates 256 values
        (1..256), launches ``<<<1, 256>>>``, syncs, and prints PASSED/FAILED.

        Assumes the kernel signature is ``(int* data, int width, ...)``
        with ``width`` as the second param — the pattern used by most
        warp-level prefix scan kernels.  The harness queries the real
        ``warpSize`` at runtime via ``hipGetDeviceProperties``.
        """
        import re
        match = re.search(r'__global__\s+void\s+(\w+)\s*\(', device_source)
        kern_name = match.group(1) if match else f"{kernel_name}_kernel"

        # Check device code for optional extra params after width
        # Common: shfl_scan_test(int *data, int width, int *partial_sums)
        extra = ""
        sig_match = re.search(
            rf'__global__\s+void\s+{re.escape(kern_name)}\s*\(([^)]*)\)',
            device_source,
        )
        if sig_match:
            parts = [p.strip() for p in sig_match.group(1).split(",")]
            if len(parts) > 2:
                # 3rd+ params — pass nullptr for pointers, 0 for scalars
                extras = []
                for p in parts[2:]:
                    pname = p.split()[-1] if p.split() else "arg"
                    extras.append("nullptr" if "*" in p else "0")
                if extras:
                    extra = ", " + ", ".join(extras)

        harness = [
            '#include <hip/hip_runtime.h>',
            '#include <cstdio>',
            '#include <cstdlib>',
            '',
            '// ── Device kernel ──',
            device_source,
            '',
            'int main() {',
            '    const int n = 256;',
            '    int *d_data, *h_data;',
            '    h_data = (int*)malloc(n * sizeof(int));',
            '    if (!h_data) { fprintf(stderr, "malloc failed\\n"); return 1; }',
            '    for (int i = 0; i < n; i++) h_data[i] = i + 1;',
            '    if (hipMalloc(&d_data, n * sizeof(int)) != hipSuccess) {',
            '        fprintf(stderr, "hipMalloc failed\\n"); return 1;',
            '    }',
            '    if (hipMemcpy(d_data, h_data, n * sizeof(int),',
            '                  hipMemcpyHostToDevice) != hipSuccess) {',
            '        fprintf(stderr, "hipMemcpy H2D failed\\n"); return 1;',
            '    }',
            '',
            '    hipDeviceProp_t prop;',
            '    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {',
            '        fprintf(stderr, "hipGetDeviceProperties failed\\n"); return 1;',
            '    }',
            '    int width = prop.warpSize;',
            f'    printf("Device: %s | warpSize=%d\\n", prop.name, width);',
            '',
            f'    {kern_name}<<<1, n>>>(d_data, width{extra});',
            '    if (hipDeviceSynchronize() != hipSuccess) {',
            '        fprintf(stderr, "hipDeviceSynchronize failed\\n"); return 1;',
            '    }',
            '',
            '    if (hipMemcpy(h_data, d_data, n * sizeof(int),',
            '                  hipMemcpyDeviceToHost) != hipSuccess) {',
            '        fprintf(stderr, "hipMemcpy D2H failed\\n"); return 1;',
            '    }',
            '',
            '    int pass = 1;',
            '    for (int i = 0; i < n; i++) {',
            '        int base = (i / width) * width;',
            '        int expected = 0;',
            '        for (int k = base; k <= i; k++) expected += (k + 1);',
            '        if (h_data[i] != expected) {',
            '            pass = 0;',
            '            printf("FAIL[%d]: got %d, expected %d\\n",',
            '                   i, h_data[i], expected);',
            '            break;',
            '        }',
            '    }',
            '    printf("%s\\n", pass ? "PASSED" : "FAILED");',
            '    hipFree(d_data); free(h_data);',
            '    return pass ? 0 : 1;',
            '}',
        ]
        return "\n".join(harness)

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

        # ── TRIZ #24 (Intermediary): Check spec FIRST for self-contained flag ──
        # The spec was auto-generated from the ORIGINAL CUDA source which
        # definitively tells us whether this is a full program or a bare
        # kernel. Kimi may strip int main() during porting, making the regex
        # check below unreliable. The spec is the authoritative source.
        spec = self.load_spec(kernel_name)
        if spec is not None and spec.get("port_mode") == "WHOLE_PROGRAM":
            return ported_kernel_source, 1, len(ported_kernel_source.splitlines())

        # DEVICE_SUBSET: the spec is authoritative — the port is meant to contain
        # ONLY device functions, driven by the synthesized spec harness. A model
        # that leaks the original main()/host driver (unportable SDK code) must
        # NOT hijack the self-contained early-return below; that recompiles the
        # exact broken driver the harness exists to replace. Strip any leaked
        # host code and always build the spec harness.
        if spec is not None and spec.get("port_mode") == "DEVICE_SUBSET":
            device_only = self._strip_to_device_code(ported_kernel_source)
            device_only = self._fix_hip_intrinsics(device_only)
            return self._harness_from_spec(spec, device_only or ported_kernel_source)

        if re.search(r'^\s*int\s+main\s*\(', ported_kernel_source, re.MULTILINE):
            return ported_kernel_source, 1, len(ported_kernel_source.splitlines())

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

        Bug 6: linker diagnostics (``ld.lld: error: undefined symbol: ...``)
        have no ``file:line:col:`` prefix, so they used to fall through to
        "unknown" — invisible to the harness-origin early abort and
        indistinguishable from a truly unclassifiable line. Given their own
        origin ("link") lets a caller give targeted guidance (e.g. "you
        dropped main() — restore it") instead of a raw linker string.
        """
        import re
        if re.search(r'undefined (?:symbol|reference to)[\s\S]{0,10}\bmain\b', error_line):
            return "link"
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

        # Build per-param host buffer declarations
        # Each pointer param gets its own std::vector host buffer
        host_buf_lines = []
        for p in params:
            if p["direction"] == "scalar":
                continue
            pname = p["name"]
            ptype = p["type"]
            pexpr = p.get("size_expr", str(input_count))
            if pexpr != str(input_count):
                pexpr = f"static_cast<int>({pexpr})"
            else:
                pexpr = str(input_count)
            if linear_ramp:
                host_buf_lines.append(f"    std::vector<{elem_type}> {pname}({pexpr});")
                host_buf_lines.append(
                    f"    for (size_t i = 0; i < {pexpr}; i++) {pname}[i] = static_cast<{elem_type}>(i + 1);"
                )
            else:
                host_buf_lines.append(
                    f"    std::vector<{elem_type}> {pname}({pexpr}, "
                    f"static_cast<{elem_type}>({default_val}));"
                )

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
                # Always do H2D copy for pointer params (seed input data)
                memcpy_h2d_lines.append(
                    f"    hipMemcpy(d_{name}, {name}.data(), "
                    f"{size_expr} * sizeof({elem_type}), hipMemcpyHostToDevice);"
                )
                # Always do D2H copy for pointer params (readback for verification)
                memcpy_d2h_lines.append(
                    f"    hipMemcpy({name}.data(), d_{name}, "
                    f"{size_expr} * sizeof({elem_type}), hipMemcpyDeviceToHost);"
                )
                free_lines.append(f"    hipFree(d_{name});")
                kernel_args.append(f"d_{name}")

        # Dynamic shared memory. A kernel that declares `extern __shared__` reads
        # its size from the launch's 3rd config argument; launching it with 0
        # bytes makes the first shared-memory access run out of bounds → SIGSEGV
        # before any output (exactly the nvidia_shfl_scan crash). When the spec
        # does not pin a size, derive a safe default from the block dimensions
        # (one element per thread covers both per-thread and per-warp shared
        # arrays) so the kernel at least does not crash on launch.
        dynamic_smem = spec.get("dynamic_shared_mem", 0)
        if not dynamic_smem and re.search(r'extern\s+__shared__', ported_kernel_source):
            block_threads = bx * by * bz
            dynamic_smem = block_threads * self._sizeof_ctype(elem_type)
            print(f"║ ⚠️ '{fn}' uses extern __shared__ but spec sets no "
                  f"dynamic_shared_mem — defaulting to {dynamic_smem} bytes "
                  f"({block_threads}×{elem_type}) to avoid a launch-time SIGSEGV.")
        dsmem_suffix = f", {dynamic_smem}" if dynamic_smem else ""

        # Build print / output format — read from last pointer param's host buffer
        # (convention: the last pointer param is the output buffer)
        readback_buf = "output_verify"
        for p in params:
            if p["direction"] != "scalar":
                readback_buf = p["name"]
        print_lines = []
        if out.get("format") == "int_per_line":
            for i in range(readback_count):
                print_lines.append(
                    f'        std::cout << {readback_buf}[{i}] << std::endl;'
                )
        else:
            for i in range(readback_count):
                print_lines.append(
                    f'        std::cout << std::fixed << {readback_buf}[{i}] << std::endl;'
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

        # A spec may reference the input element count symbolically as ``count``
        # in a param's ``size_expr`` (e.g. size_expr: "count"). That identifier
        # is otherwise undefined in the harness — declare it. Skip if a scalar
        # param is already literally named ``count`` (it would collide).
        if any(p.get("size_expr") == "count" for p in params) and \
                not any(p["name"] == "count" for p in params):
            lines.append(f'    const int count = {input_count};')

        for line in scalar_init_lines:
            lines.append(line)
        for line in host_buf_lines:
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

    @staticmethod
    def _signal_name(exit_code) -> str:
        """Translate a negative POSIX exit code to its signal name.

        subprocess reports 'killed by signal N' as returncode -N. Showing a
        raw 'exit -11' makes the operator (and the LLM feedback loop) do
        signal arithmetic; 'SIGSEGV' is the actionable spelling.
        """
        if exit_code is None or exit_code >= 0:
            return ""
        import signal as _signal
        try:
            return _signal.Signals(-exit_code).name
        except (ValueError, AttributeError):
            return f"signal {-exit_code}"

    def quick_run_check(self, kernel_name: str) -> Dict:
        """Run the binary produced by the last quick_compile_check for *kernel_name*.

        The in-loop compile check already links a real executable into
        build_dir/loop_<kernel>/ — running it costs ~1s and is the cheapest,
        highest-authority oracle in the pipeline. Until 2026-07-09 nobody
        called it: the loop declared victory on compile-pass, and a SIGSEGV
        was discovered once, in verify(), after the loop had exited — with
        no feedback path back to the models.

        Returns run_success / run_output / run_exit_code / signal /
        benchmark_us. A missing binary (compile never passed, or no hipcc)
        reports run_success=False with exit_code None — callers should only
        invoke this after a passing compile check.
        """
        kernel_build_dir = self.build_dir / f"loop_{kernel_name}"
        safe_kernel_name = re.sub(r'[^a-zA-Z0-9_-]', '', kernel_name)
        run_ok, run_output, benchmark, exit_code = self._run(kernel_build_dir, safe_kernel_name)
        return {
            "run_success": run_ok,
            "run_output": run_output[:1000] if run_output else "",
            "run_exit_code": exit_code,
            "signal": self._signal_name(exit_code),
            "benchmark_us": benchmark,
        }

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
        error_context = []  # TRIZ #24: the ACTUAL source lines at each error location
        harness_lines = harness.splitlines()
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
            # Hidden resource: we HAVE the compiled file in memory and hipcc
            # gives an exact line number — extract the offending source lines
            # deterministically instead of asking an LLM to count to line N
            # in a 15k-char blob. This is what exposed the 2026-07-09 failure
            # (every error lived in post-processor-injected shim lines the
            # models were never shown in isolation).
            lm = re.search(r':(\d+):\d+:\s*(?:fatal )?error:', stripped)
            if lm:
                err_line_no = int(lm.group(1))  # 1-indexed
                lo = max(0, err_line_no - 2)
                hi = min(len(harness_lines), err_line_no + 1)
                snippet = "\n".join(
                    f"  {n + 1:>4}{'>' if n + 1 == err_line_no else ' '} {harness_lines[n][:160]}"
                    for n in range(lo, hi)
                )
                error_context.append(snippet)
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
            # Kept SEPARATE from "errors": the loop's new/resolved diffing and
            # cycle detection hash the error strings — folding volatile source
            # context into them would make every error look "new" every time.
            "error_context": error_context[:8],
            "all_harness_origin": all_harness_origin,
            "compile_log_path": compile_log_path,
            "kernel_name": kernel_name,
        }

    @staticmethod
    def _build_seed(spec: Optional[dict]) -> bytes:
        """Derive the deterministic input seed from the spec's input_setup.

        Returns a consistent seed so that both the HIP kernel and the reference
        see identical input data. When the spec has no input_setup, returns an
        empty seed (both sides use whatever default the harness provides).
        """
        if not spec or "input_setup" not in spec:
            return b""
        setup = spec["input_setup"]
        # If setup is a dict with explicit values, serialize them.
        if isinstance(setup, dict):
            return json.dumps(setup, sort_keys=True).encode("utf-8")
        # If setup is a string (filename or seed literal), use it directly.
        if isinstance(setup, str):
            return setup.encode("utf-8")
        return b""

    def verify(self, hip_source: str, test_input: str = "",
               kernel_name: str = "test_kernel",
               on_progress=None, cuda_source: str = "") -> Dict:
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
            "run_exit_code": None,
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
        arch = self._resolve_offload_arch(spec)
        compile_ok, compile_out, compile_log_path = self._compile(
            harness_file, kernel_build_dir, kernel_name,
            on_progress=on_progress, offload_arch=arch,
        )
        if on_progress: on_progress(70, "compilation complete" if compile_ok else "compilation failed")
        result["compile_success"] = compile_ok
        result["compile_output"] = compile_out[:1000] if compile_out else ""
        result["compile_log_path"] = compile_log_path

        if not compile_ok:
            # ── DEVICE-ONLY PROOF: try extracting just the device kernel ──
            # Some kernels (e.g. NVIDIA SDK samples with int main()) contain
            # unportable host code but perfectly valid device kernels. Before
            # giving up, strip to device functions and try a minimal proof.
            device_only = self._strip_to_device_code(hip_source)
            if device_only:
                device_only = self._fix_hip_intrinsics(device_only)
                proof_ok, proof_out, proof_log = self._try_device_only_proof(
                    kernel_name, device_only, kernel_build_dir, spec,
                    on_progress, arch,
                )
                if proof_ok:
                    result["compile_success"] = True  # succeeded via device-only
                    result["compile_output"] = (
                        f"Full harness compile failed. "
                        f"Device-only proof succeeded.\n{proof_out[:800]}"
                    )
                    # Step 4: Run the device-only binary
                    if on_progress: on_progress(75, "running device-only proof")
                    run_ok, run_output, benchmark, run_exit_code = self._run(
                        kernel_build_dir, f"{kernel_name}_proof"
                    )
                    if on_progress: on_progress(90, "device-only run complete" if run_ok else "run failed")
                    result["run_success"] = run_ok
                    result["run_output"] = run_output[:1000] if run_output else ""
                    result["benchmark_us"] = benchmark
                    result["run_exit_code"] = run_exit_code
                    if run_ok:
                        result["device_only_verified"] = True
                        result["passed"] = True
                        if on_progress: on_progress(100, "PASSED ✅ (device-only proof)")
                        return result
                    else:
                        # Device-only compiled but crashed at runtime
                        compile_out += (
                            f"\n\n⚠️ Device-only proof compiled but RUNTIME FAILED:\n"
                            f"   {run_output[:300]}"
                        )

            # ── Save for manual hipcc (last resort) ──
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
        run_ok, run_output, benchmark, run_exit_code = self._run(kernel_build_dir, kernel_name)
        if on_progress: on_progress(90, "run complete" if run_ok else "run failed")
        result["run_success"] = run_ok
        result["run_output"] = run_output[:1000] if run_output else ""
        result["benchmark_us"] = benchmark
        result["run_exit_code"] = run_exit_code

        if not run_ok:
            if on_progress: on_progress(100, "run failed")
            result["passed"] = False
            return result

        # Step 4b: Determinism check — run the same binary a second time on the
        # same input. Kernels with unsynchronised writes to shared memory or
        # unordered atomic reductions typically produce different outputs
        # between runs. A bit-identical rerun is a necessary (not sufficient)
        # signal that the migration preserved barrier/sync semantics.
        # Skipped when spec opts out (deterministic=False) or the first run
        # produced no output at all.
        deterministic = True
        if run_output and (not spec or spec.get("deterministic", True)):
            run_ok2, run_output2, _bench2, _exit2 = self._run(kernel_build_dir, kernel_name)
            if run_ok2 and run_output2 != run_output:
                deterministic = False
                result["passed"] = False
                result["determinism"] = {
                    "ok": False,
                    "reason": "Two runs on the same input produced different outputs — "
                              "likely race on shared memory or unordered reduction.",
                }
                if on_progress: on_progress(100, "determinism failed")
                return result
        result["determinism"] = {"ok": deterministic}

        # Step 5: Diff against an EXECUTED reference (90->100%)
        if on_progress: on_progress(95, "resolving executed reference")

        # The seed MUST be the same input the harness fed the kernel. Derive it
        # from the spec's input_setup so both sides see identical data.
        seed = self._build_seed(spec)

        ref = self.oracle.resolve(kernel_name, cuda_source, seed)

        if ref.kind is OracleKind.UNVERIFIABLE:
            # Compiled + ran, but we CANNOT prove correctness. This is not PASSED
            # and not FAILED — it is UNVERIFIED. Never store as a verified pattern.
            result["output_match"] = None
            result["oracle"] = {"kind": ref.kind.value, **ref.provenance}
            result["passed"] = False
            result["unverifiable"] = True
            result["diff_report"] = ("Compiled and ran, but no executed reference "
                                     "exists — correctness not proven. Add "
                                     f"reference/{kernel_name}_ref.cpp or a source "
                                     "self-check to make this verifiable.")
            if on_progress: on_progress(100, "UNVERIFIABLE (compiled + ran)")
            return result

        atol, rtol = self._tolerances(spec)
        diff_ok, diff_report = self._diff(run_output, ref.output, atol=atol, rtol=rtol)
        result["output_match"] = diff_ok
        result["diff_report"] = diff_report[:500] if diff_report else ""
        result["tolerance"] = {"atol": atol, "rtol": rtol}
        result["oracle"] = {"kind": ref.kind.value, **ref.provenance}
        result["passed"] = diff_ok
        if on_progress: on_progress(100, "PASSED ✅" if diff_ok else "DIFF FAILED")
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

    @staticmethod
    def _write_compile_commands(build_dir: Path, harness_file: Path,
                                argv: list) -> Path:
        """Write a ``compile_commands.json`` describing this build.

        Standard clang compilation-database format (a JSON array of
        ``{directory, file, arguments}`` objects). One entry per
        harness compile — sufficient for Clang tools like ``clangd`` or a
        future AST-based rewriter to open the harness with the same flags
        hipcc actually used.

        Returns the path to the emitted file.
        """
        cdb_path = build_dir / "compile_commands.json"
        entry = {
            "directory": str(build_dir),
            "file": str(harness_file),
            "arguments": list(argv),
        }
        cdb_path.write_text(json.dumps([entry], indent=2), encoding="utf-8")
        return cdb_path

    def _resolve_offload_arch(self, spec: Optional[dict]) -> str:
        """Precedence: spec['target_arch'] > env AMD_OFFLOAD_ARCH > gfx942.

        The instance default (self.offload_arch) already resolves env vs the
        gfx942 fallback. Spec override wins because a kernel author knows the
        arch it was tuned for; a spec that hard-codes gfx90a should not silently
        drift to gfx942 on a machine whose env didn't set the var.
        """
        if spec and isinstance(spec.get("target_arch"), str) and spec["target_arch"]:
            return spec["target_arch"]
        return self.offload_arch

    def _compile(self, harness_file: Path, build_dir: Path, kernel_name: str,
                 on_progress=None, offload_arch: Optional[str] = None) -> tuple:
        """Compile HIP kernel with hipcc.

        on_progress: optional callback(percent: int, stage: str) — called
        at key compilation milestones so the UI can show a 0-100% bar.

        Returns (compile_ok, compile_output, compile_log_path). compile_output
        has build-dir path prefixes stripped for readability; compile_log_path
        points to the full, untruncated, unstripped output on disk.
        """
        output_bin = build_dir / kernel_name
        iteration = self._iteration_hint(kernel_name) if self.debug.enabled else 0

        if self._hipcc_available:
            cmd = []
            try:
                if on_progress: on_progress(30, "hipcc starting")
                # Sanitize kernel_name to prevent command injection
                safe_kernel_name = re.sub(r'[^a-zA-Z0-9_-]', '', kernel_name)
                output_bin = build_dir / safe_kernel_name
                # Always use list-form subprocess (no shell=True) to prevent injection
                if on_progress: on_progress(40, "hipcc compiling")
                arch = offload_arch or self.offload_arch
                cmd = [self._hipcc_path, "-o", str(output_bin), str(harness_file),
                       "-std=c++17", "-O2", f"--offload-arch={arch}",
                       "-ferror-limit=5"]
                # Emit compile_commands.json next to the harness so Clang-based
                # tooling (clangd, hipify-clang, or a future AST-repair stage)
                # sees the exact flags this build used. Best-effort — a failure
                # to write the JSON must never break the compile.
                try:
                    self._write_compile_commands(build_dir, harness_file, cmd)
                except Exception as e:
                    print(f"║ ⚠️ compile_commands.json write failed: {e}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60,
                    cwd=str(build_dir)
                )
                if on_progress: on_progress(65, "hipcc finished")
                raw_output = result.stdout + result.stderr
                log_path = self._write_compile_log(build_dir, kernel_name, raw_output)
                shortened = "\n".join(
                    self._shorten_error_line(line, build_dir) for line in raw_output.splitlines()
                )
                # Debug Mode: the exact argv, the resolved environment, the
                # compiler's own version, and the COMPLETE stdout/stderr. The
                # `shortened` copy above is for display; nothing truncated is
                # what gets persisted here.
                if self.debug.enabled:
                    self.debug.log_compile(
                        command=cmd, stdout=result.stdout, stderr=result.stderr,
                        returncode=result.returncode, cwd=str(build_dir),
                        compiler_version=_compiler_version(self._hipcc_path),
                        source_path=str(harness_file),
                        source_text=self._read_text_safely(harness_file),
                        diagnostics=[l for l in raw_output.splitlines() if "error:" in l],
                        artifacts=([str(output_bin)] if output_bin.exists() else []),
                        iteration=iteration, kernel_name=kernel_name,
                        offload_arch=arch,
                        compile_log_path=log_path,
                    )
                return result.returncode == 0, shortened, log_path
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                if self.debug.enabled:
                    self.debug.log_compile(
                        command=cmd, stdout="", stderr=str(e), returncode=None,
                        cwd=str(build_dir),
                        compiler_version=_compiler_version(self._hipcc_path),
                        source_path=str(harness_file),
                        source_text=self._read_text_safely(harness_file),
                        diagnostics=[f"{type(e).__name__}: {e}"],
                        iteration=iteration, kernel_name=kernel_name,
                        failure_mode=type(e).__name__,
                    )
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
            # "hipcc is absent" is a compiler-stage fact, and a debug session
            # that silently omits it looks identical to one where the compile
            # was never attempted.
            if self.debug.enabled:
                self.debug.log_compile(
                    command=[], stdout="", stderr=msg, returncode=None,
                    cwd=str(build_dir), compiler_version="",
                    source_path=str(harness_file),
                    source_text=self._read_text_safely(harness_file),
                    diagnostics=["hipcc not available on this host"],
                    iteration=iteration, kernel_name=kernel_name,
                    failure_mode="hipcc_not_found",
                )
            return False, msg, log_path

    @staticmethod
    def _read_text_safely(path: Path) -> str:
        """Read *path* for a debug artifact. A read failure is never fatal."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _run(self, build_dir: Path, kernel_name: str) -> tuple:
        """Run the compiled HIP kernel.

        Returns (run_ok, output, benchmark_us, exit_code). ``exit_code`` is
        ``None`` when the binary couldn't be launched at all (missing, timed
        out) — distinct from a real non-zero exit, so callers can tell
        "never ran" from "ran and failed" (e.g. the NVIDIA sample's
        EXIT_WAIVED=2 on unsupported hardware vs. a genuine crash).
        """
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
                return result.returncode == 0, output, round(elapsed, 2), result.returncode
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return False, str(e), None, None
        return False, "Binary not found — compile step may have failed.", None, None

    # Default tolerances for kernels whose spec omits them. atol handles the
    # near-zero region; rtol scales with magnitude. Chosen so exact-integer
    # kernels (histogram, scan) still match byte-for-byte.
    _DEFAULT_ATOL = 1e-5
    _DEFAULT_RTOL = 0.0

    def _tolerances(self, spec: Optional[dict]) -> tuple:
        """Resolve (atol, rtol) for numerical diff.

        Spec keys, in order of precedence:
          - ``tolerance: {atol: <float>, rtol: <float>}`` — explicit
          - ``atol`` / ``rtol`` at top level — flat form
          - default ``(1e-5, 0.0)`` — matches pre-existing behavior
        """
        if not spec:
            return self._DEFAULT_ATOL, self._DEFAULT_RTOL
        tol = spec.get("tolerance") if isinstance(spec.get("tolerance"), dict) else None
        if tol:
            atol = float(tol.get("atol", self._DEFAULT_ATOL))
            rtol = float(tol.get("rtol", self._DEFAULT_RTOL))
        else:
            atol = float(spec.get("atol", self._DEFAULT_ATOL))
            rtol = float(spec.get("rtol", self._DEFAULT_RTOL))
        return atol, rtol

    def _diff(self, actual_output: str, expected_output: str,
              atol: float = None, rtol: float = None) -> tuple:
        """Compare actual output against CUDA reference output.

        Passes when ``|a - e| <= atol + rtol * max(|a|, |e|)`` element-wise.
        ``atol`` defaults to 1e-5 (pre-existing behavior), ``rtol`` to 0.
        """
        if atol is None:
            atol = self._DEFAULT_ATOL
        if rtol is None:
            rtol = self._DEFAULT_RTOL

        if not actual_output or not expected_output:
            return False, "Missing output data for comparison."

        actual_lines = actual_output.strip().splitlines()
        expected_lines = expected_output.strip().splitlines()

        if actual_lines == expected_lines:
            return True, "Outputs match exactly (byte-for-byte)."

        # Try floating-point tolerant diff.
        try:
            actual_floats = [float(l) for l in actual_lines if l.strip()]
            expected_floats = [float(l) for l in expected_lines if l.strip()]
            if len(actual_floats) == len(expected_floats):
                worst_abs = 0.0
                worst_rel = 0.0
                worst_idx = -1
                for i, (a, e) in enumerate(zip(actual_floats, expected_floats)):
                    diff = abs(a - e)
                    bound = atol + rtol * max(abs(a), abs(e))
                    if diff > bound and diff > worst_abs:
                        worst_abs = diff
                        worst_rel = diff / max(abs(e), 1e-300) if abs(e) > 0 else float("inf")
                        worst_idx = i
                if worst_idx < 0:
                    max_abs = max(abs(a - e) for a, e in zip(actual_floats, expected_floats))
                    return True, (f"Outputs match within tolerance "
                                  f"(atol={atol:.1e}, rtol={rtol:.1e}, "
                                  f"max abs diff: {max_abs:.2e}).")
                return False, (f"Outputs differ at index {worst_idx}: "
                               f"|Δ|={worst_abs:.4g} exceeds "
                               f"atol+rtol·max = {atol + rtol * max(abs(actual_floats[worst_idx]), abs(expected_floats[worst_idx])):.4g} "
                               f"(rel={worst_rel:.2%}).")
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

        # Second: resolve through PATH without a shell.
        # This replaces two shell=True probes ("hipcc --version" and
        # "command -v hipcc || which hipcc"). The first was already subsumed by
        # the bare "hipcc" entry in the loop above — /bin/sh resolves against the
        # same inherited PATH — and the second is precisely what shutil.which does,
        # minus the shell. Keeps the project's own list-form policy (see the
        # "no shell=True to prevent injection" note in _compile).
        found = shutil.which("hipcc")
        if found:
            try:
                result = subprocess.run([found, "--version"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self._hipcc_path = found
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fourth: glob search
        import glob
        for match in glob.glob("/opt/rocm*/**/hipcc", recursive=True):
            self._hipcc_path = match
            return True
        return False
