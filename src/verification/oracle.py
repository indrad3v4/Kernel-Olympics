"""Execution-grounded differential oracle.

Replaces the fabricated golden files. The reference the port is diffed against
is PRODUCED BY EXECUTING code on the same input as the port, never typed by a
human and never inferred from static analysis.

Resolution order for a kernel's reference:
  1. SELF_CHECK   — the original source ships its own CPU verifier
                    (CPUverify / verifyDataRowSums / a printf pass-fail).
                    We reuse it: run the ORIGINAL host program, capture its
                    verdict/output, and treat that as ground truth.
  2. CPU_REF      — a checked-in host reference (reference/<kernel>_ref.cpp)
                    that computes the same math on the same seeded input.
  3. UNVERIFIABLE — neither exists. The kernel is compile-checked only and can
                    never be reported PASSED. This is a first-class outcome,
                    not an error.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class OracleKind(str, Enum):
    SELF_CHECK = "self_check"     # source's own CPUverify / pass-fail
    CPU_REF = "cpu_ref"           # checked-in host reference implementation
    UNVERIFIABLE = "unverifiable" # no executed reference is available


@dataclass
class Reference:
    """A reference the port will be diffed against — and how it was obtained.

    `output` is None only for UNVERIFIABLE. `input_seed` is the exact input
    both sides run on, so the diff is genuinely differential.
    """
    kind: OracleKind
    output: Optional[str] = None
    input_seed: bytes = b""
    provenance: dict = field(default_factory=dict)  # how it was produced

    @property
    def verifiable(self) -> bool:
        return self.kind is not OracleKind.UNVERIFIABLE and self.output is not None


class DifferentialOracle:
    """Produces an EXECUTED reference for a kernel. No fabrication.

    Constructor takes the paths it needs; it does not import the verifier, so
    it stays testable in isolation (the thing the review says the repo lacks).
    """

    # Symbols that mark a source as carrying its own reference check.
    _SELF_CHECK_SYMBOLS = ("CPUverify", "verifyDataRowSums", "checkResult",
                           "verify_result", "compareResults")

    def __init__(self, repo_root: Path, nvcc: Optional[str] = None,
                 host_cxx: str = "c++"):
        self.repo_root = Path(repo_root)
        self.nvcc = nvcc                      # optional; only used for SELF_CHECK on NVIDIA CI
        self.host_cxx = host_cxx
        self.ref_dir = self.repo_root / "sample_kernels" / "reference"

    # -- public API ----------------------------------------------------------

    def resolve(self, kernel_name: str, cuda_source: str,
                seed: bytes) -> Reference:
        """Return an EXECUTED reference for `kernel_name`, or UNVERIFIABLE.

        `seed` is the serialized input both port and reference must consume, so
        that whatever we compare the port against was computed on the SAME data.
        """
        # 1) Source ships its own verifier -> run the original program.
        if self._has_self_check(cuda_source):
            ref = self._run_self_check(kernel_name, cuda_source, seed)
            if ref is not None:
                return ref

        # 2) A checked-in CPU reference exists -> run it.
        cpu_ref_src = self.ref_dir / f"{kernel_name}_ref.cpp"
        if cpu_ref_src.exists():
            ref = self._run_cpu_ref(kernel_name, cpu_ref_src, seed)
            if ref is not None:
                return ref

        # 3) Nothing executable. Honest dead-end.
        return Reference(
            kind=OracleKind.UNVERIFIABLE,
            provenance={"reason": "no self-check in source and no "
                                  f"reference/{kernel_name}_ref.cpp on disk"},
        )

    # -- (1) self-check path -------------------------------------------------

    def _has_self_check(self, cuda_source: str) -> bool:
        return any(sym in cuda_source for sym in self._SELF_CHECK_SYMBOLS)

    def _run_self_check(self, kernel_name: str, cuda_source: str,
                        seed: bytes) -> Optional[Reference]:
        """Compile+run the ORIGINAL CUDA program and capture its verdict.

        This is the highest-fidelity oracle: it is the vendor's own reference,
        executed. It requires nvcc + an NVIDIA GPU, so it is expected to run in
        CI, not on a dev laptop. When nvcc is absent we return None and the
        caller falls through to the CPU reference. The captured output is then
        committed (reference/<kernel>.trace) so laptop runs reuse the CI trace
        WITHOUT re-inventing it — a cached execution is still an execution.
        """
        cached = self.ref_dir / f"{kernel_name}.trace"
        seed_hash = hashlib.sha256(seed).hexdigest()[:16]

        # Reuse a committed trace only if it was produced on THIS input.
        if cached.exists():
            meta = self.ref_dir / f"{kernel_name}.trace.json"
            if meta.exists():
                info = json.loads(meta.read_text())
                if info.get("seed_hash") == seed_hash:
                    return Reference(
                        kind=OracleKind.SELF_CHECK,
                        output=cached.read_text(),
                        input_seed=seed,
                        provenance={"source": "committed CI trace",
                                    "seed_hash": seed_hash, **info},
                    )

        if self.nvcc is None:
            return None  # can't execute here; caller tries CPU_REF next

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / f"{kernel_name}.cu"
            src.write_text(cuda_source, encoding="utf-8")
            bin_ = td / kernel_name
            compile_ = subprocess.run(
                [self.nvcc, str(src), "-o", str(bin_), "-std=c++17"],
                capture_output=True, text=True, timeout=120,
            )
            if compile_.returncode != 0:
                return None  # original didn't build in CI; nothing to trust
            run = subprocess.run([str(bin_)], input=seed, capture_output=True,
                                 timeout=60)
            if run.returncode != 0:
                return None
            out = run.stdout

        # Persist so laptop/offline runs reuse this EXECUTED trace.
        cached.write_text(out, encoding="utf-8")
        (self.ref_dir / f"{kernel_name}.trace.json").write_text(
            json.dumps({"seed_hash": seed_hash, "via": "nvcc"}), encoding="utf-8")
        return Reference(kind=OracleKind.SELF_CHECK, output=out, input_seed=seed,
                         provenance={"source": "nvcc execution", "seed_hash": seed_hash})

    # -- (2) CPU reference path ---------------------------------------------

    def _run_cpu_ref(self, kernel_name: str, ref_src: Path,
                     seed: bytes) -> Optional[Reference]:
        """Compile+run a host-only reference impl on the SAME seed as the port.

        The reference reads the seed from stdin and prints the expected output
        in the same format the harness prints. Runs anywhere — no GPU, no nvcc.
        """
        with tempfile.TemporaryDirectory() as td:
            bin_ = Path(td) / f"{kernel_name}_ref"
            compile_ = subprocess.run(
                [self.host_cxx, str(ref_src), "-o", str(bin_), "-std=c++17", "-O2"],
                capture_output=True, text=True, timeout=60,
            )
            if compile_.returncode != 0:
                return None
            run = subprocess.run([str(bin_)], input=seed, capture_output=True,
                                 timeout=30)
            if run.returncode != 0:
                return None
            out = run.stdout
        return Reference(
            kind=OracleKind.CPU_REF, output=out, input_seed=seed,
            provenance={"source": f"reference/{ref_src.name} execution"},
        )
