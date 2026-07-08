#!/usr/bin/env python3
"""
AMD Developer Cloud Deploy Script — R8.3

Usage:
    # Interactive (sets up environment):
    python3 scripts/deploy_to_amd_cloud.py --setup

    # Full run on AMD GPU:
    python3 scripts/deploy_to_amd_cloud.py --run

    # Just compile + verify on existing ported_kernels/:
    python3 scripts/deploy_to_amd_cloud.py --verify-only

Prerequisites:
    - Access to notebooks.amd.com (AMD Developer Cloud)
    - Fireworks API key in environment or .env
    - This script is designed to run INSIDE AMD Cloud Jupyter terminal
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utf8_console import enable_utf8_console
enable_utf8_console()

REPO_URL = "https://github.com/indrad3v4/Kernel-Olympics.git"
WORK_DIR = Path("/workspace/Kernel-Olympics")
FIREWORKS_KEY = os.environ.get("FIREWORKS_API_KEY", "")


def run(cmd, cwd=None, timeout=60, capture=True):
    """Run a shell command and print output."""
    print(f"  $ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True,
            cwd=str(cwd or WORK_DIR), timeout=timeout
        )
        if capture:
            for line in (result.stdout or "").splitlines()[:30]:
                print(f"    {line}")
            for line in (result.stderr or "").splitlines()[:10]:
                print(f"    ! {line}")
        return result
    except subprocess.TimeoutExpired:
        print(f"    ! TIMEOUT after {timeout}s")
        return None


def setup():
    """Clone repo and install dependencies."""
    print("═" * 60)
    print("KERNEL OLYMPICS — AMD CLOUD SETUP")
    print("═" * 60)

    if WORK_DIR.exists():
        print(f"\n[1] Repo exists at {WORK_DIR}, pulling latest...")
        run("git pull", timeout=30)
    else:
        print(f"\n[1] Cloning repo...")
        run(f"git clone {REPO_URL} {WORK_DIR}", timeout=120)

    print("\n[2] Installing Python dependencies...")
    run("pip install -r requirements.txt", timeout=120)

    print("\n[3] Checking ROCm environment...")
    run("rocm-smi --showproductname 2>/dev/null | head -5", timeout=10)
    run("hipcc --version 2>/dev/null | head -3", timeout=10)
    run("which python3 && python3 --version", timeout=5)

    print("\n[4] Checking Fireworks API key...")
    if FIREWORKS_KEY:
        print(f"    ✅ FIREWORKS_API_KEY set ({FIREWORKS_KEY[:8]}...)")
    elif (WORK_DIR / ".env").exists():
        print("    ✅ .env file found")
    else:
        print("    ⚠️ No FIREWORKS_API_KEY set — template fallback only")

    print("\n✅ Setup complete!")
    return True


def run_pipeline():
    """Run full pipeline on AMD GPU."""
    print("═" * 60)
    print("KERNEL OLYMPICS — PIPELINE ON AMD GPU")
    print("═" * 60)

    if not WORK_DIR.exists():
        print("❌ Repo not found. Run --setup first.\n")
        return False

    os.chdir(str(WORK_DIR))
    os.environ["PYTHONPATH"] = "src"
    if FIREWORKS_KEY:
        os.environ["FIREWORKS_API_KEY"] = FIREWORKS_KEY

    kernels = [
        "sample_kernels/cuda/warp_reduce.cu",
        "sample_kernels/cuda/new_kernel.cu",
        "sample_kernels/cuda/transpose.cu",
    ]

    for kernel in kernels:
        print(f"\n{'─' * 60}")
        print(f"[Pipeline] Processing: {kernel}")
        print(f"{'─' * 60}")
        rc = run(
            f"python3 src/main.py --input {kernel} --output /tmp/report_{Path(kernel).stem}.json",
            timeout=300
        )

    # Show summary
    print("\n\n📊 Pipeline Summary:")
    for kernel in kernels:
        report_path = Path(f"/tmp/report_{Path(kernel).stem}.json")
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                stats = data.get("statistics", {})
                risk = stats.get("risk_breakdown", {})
                print(f"  {kernel}: "
                      f"R={risk.get('red',0)}/Y={risk.get('yellow',0)}/G={risk.get('green',0)} "
                      f"| Patterns: {stats.get('total_danger_patterns_found', '?')} "
                      f"| Conf: {stats.get('avg_porting_confidence', '?')}%")
            except Exception:
                print(f"  {kernel}: (error reading report)")

    return True


def verify_kernels():
    """Compile and run all ported kernels on AMD GPU hardware."""
    print("═" * 60)
    print("KERNEL OLYMPICS — AMD GPU VERIFICATION")
    print("═" * 60)

    ported_dir = WORK_DIR / "ported_kernels"
    if not ported_dir.exists() or not list(ported_dir.glob("*.hip.cpp")):
        print("❌ No ported kernels found. Run --run first.")
        return False

    results = []
    for kernel_file in sorted(ported_dir.glob("*.hip.cpp")):
        name = kernel_file.stem
        binary = Path(f"/tmp/{name}")

        print(f"\n{'─' * 60}")
        print(f"[Verify] {kernel_file.name}")
        print(f"{'─' * 60}")

        # Step 1: Compile
        print("\n📦 Compiling with hipcc...")
        rc = run(
            f"hipcc -o {binary} {kernel_file} -std=c++17 -O2 --offload-arch=gfx942",
            timeout=60
        )
        if rc is None or rc.returncode != 0:
            print(f"  ❌ {name}: COMPILATION FAILED")
            results.append({"kernel": name, "compile": False, "run": False, "passed": False})
            continue

        print(f"  ✅ {name}: COMPILATION PASSED")

        # Step 2: Run
        print("\n🚀 Running on AMD GPU...")
        rc = run(f"{binary}", timeout=30)
        if rc is None or rc.returncode != 0:
            print(f"  ❌ {name}: EXECUTION FAILED")
            results.append({"kernel": name, "compile": True, "run": False, "passed": False})
            continue

        passed = "PASSED" in (rc.stdout or "")
        print(f"  ✅ {name}: EXECUTION {'PASSED' if passed else 'COMPLETED'}")
        results.append({"kernel": name, "compile": True, "run": True, "passed": passed})

    # Summary
    print(f"\n\n{'═' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'═' * 60}")
    passed_count = sum(1 for r in results if r["passed"])
    for r in results:
        status = "✅" if r["passed"] else "❌" if not r["compile"] else "⚠️"
        print(f"  {status} {r['kernel']}: compile={r['compile']} run={r['run']} passed={r['passed']}")
    print(f"\n{passed_count}/{len(results)} kernels verified on AMD GPU 🚀")

    # Save proof
    proof = {
        "verified_on": "AMD Developer Cloud (notebooks.amd.com)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rocm_version": "7.2",
        "results": results,
        "summary": f"{passed_count}/{len(results)} passed"
    }
    proof_path = WORK_DIR / "amd_gpu_proof.json"
    proof_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(f"\nProof saved to: {proof_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if "--setup" in sys.argv:
        setup()
    if "--run" in sys.argv:
        run_pipeline()
    if "--verify-only" in sys.argv:
        verify_kernels()
    if "--all" in sys.argv:
        setup()
        run_pipeline()
        verify_kernels()

    print("\nDone. 🌠 — Team Meteorite")


if __name__ == "__main__":
    main()
