#!/usr/bin/env python3
"""Generate a device-only proof harness for a ported HIP kernel.

Usage:
    python3 scripts/generate_proof.py ported_kernels/x.hip.cpp kernel_name /tmp/proof.hip.cpp

Reads the ported kernel, strips host code, fixes intrinsics, wraps with
a minimal test harness. Prints nothing on success.
"""

import sys
from pathlib import Path

# Add src to path so we can import verifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from verification.verifier import VerificationAgent


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <ported_kernel.hip.cpp> <kernel_name> <output_path>",
              file=sys.stderr)
        sys.exit(1)

    hip_path = Path(sys.argv[1])
    kernel_name = sys.argv[2]
    out_path = Path(sys.argv[3])

    if not hip_path.exists():
        print(f"ERROR: {hip_path} not found", file=sys.stderr)
        print(f"Run  make port  or  python3 -m src.main --input <kernel.cu>  first",
              file=sys.stderr)
        sys.exit(1)

    v = VerificationAgent()
    src = hip_path.read_text(encoding="utf-8")

    # Strip host code, fix intrinsics, generate proof harness
    device = v._strip_to_device_code(src)
    device = v._fix_hip_intrinsics(device)
    proof = v._legacy_device_proof_harness(kernel_name, device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(proof, encoding="utf-8")

    print(f"  ✅ Proof harness: {out_path} ({len(proof.splitlines())} lines)")


if __name__ == "__main__":
    main()
