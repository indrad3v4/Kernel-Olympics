#!/usr/bin/env python3
"""
Kernel Olympics — Jupyter-friendly CLI runner.
Usage in notebook:
    %run run_in_jupyter.py sample_kernels/cuda/vector_add.cu
or:
    !python run_in_jupyter.py ../../sample_kernels/cuda/vector_add.cu
"""

import sys, os, json, time, shutil
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# ── ANSI colors (safe for Jupyter output) ──
G = "\033[92m"  # green
Y = "\033[93m"  # yellow
R = "\033[91m"  # red
C = "\033[96m"  # cyan
B = "\033[1m"   # bold
D = "\033[2m"   # dim
N = "\033[0m"   # reset

# ── Check for hipcc ──
HIPCC = shutil.which("hipcc")

def box(title, lines):
    """Draw a simple bordered box."""
    w = 66
    bar = "═" * w
    print(f"{B}╔═{bar}╗{N}")
    print(f"{B}║{N}  {C}{title}{N}")
    for l in lines:
        print(f"{B}║{N}  {l}")
    print(f"{B}╚═{bar}╝{N}")

def print_stage(n, label, detail=""):
    print(f"\n{B}─── Stage {n}: {label} {D}{detail}{N}")

def run():
    if len(sys.argv) < 2:
        print(f"{R}Usage:{N} python run_in_jupyter.py <path-to-cu-file>")
        print(f"  {D}e.g.{N}  python run_in_jupyter.py sample_kernels/cuda/vector_add.cu")
        sys.exit(1)

    cu_path = Path(sys.argv[1])
    if not cu_path.exists():
        print(f"{R}File not found:{N} {cu_path}")
        sys.exit(1)

    kernel_name = cu_path.stem
    source = cu_path.read_text(encoding="utf-8")

    # ── Banner ──
    box("Kernel Olympics — Jupyter Runner", [
        f"  {B}Input:{N}  {cu_path.name} ({cu_path.stat().st_size:,} bytes)",
        f"  {B}hipcc:{N}  {'✓ available' if HIPCC else '✗ not found'}",
        f"  {B}API:{N}    {'✓ FIREWORKS key set' if os.environ.get('FIREWORKS_API_KEY') else '✗ no key — template fallback only'}",
    ])

    # ── Step 1: Quick CLI call ──
    print_stage(1, "Running pipeline", f"{cu_path.name} → port + verify")
    t0 = time.time()

    from src.main import KernelOlympics
    ko = KernelOlympics(fresh=False)

    report = ko.run(
        input_paths=[str(cu_path.resolve())],
        reference_dir=str(ROOT / "sample_kernels" / "reference"),
    )

    elapsed = time.time() - t0
    ps = report.get("pipeline_state", {})
    result = report.get("result", "N/A")
    cost = ps.get("total_cost", 0)
    iters = report.get("iterations_used", "N/A")

    print(f"\n  {G}✓ Pipeline finished in {elapsed:.1f}s | cost: ${cost:.4f} | result: {result}{N}")

    # ── Step 2: Find HIP output ──
    print_stage(2, "Reading ported HIP code")

    # Check verification results in report
    hip_code = ""
    sections = report.get("sections", {})
    verifications = sections.get("verification", [])
    if verifications:
        hip_code = verifications[0].get("ported_code", "")

    # Fallback: check ported_kernels/ on disk
    if not hip_code:
        ported_file = ROOT / "ported_kernels" / f"{kernel_name}.hip.cpp"
        if ported_file.exists():
            hip_code = ported_file.read_text(encoding="utf-8")

    if hip_code:
        print(f"  {G}✓ Ported code found ({len(hip_code):,} chars){N}")
        print(f"\n{B}─── Output HIP Code ──────────────────────────────────{N}")
        for i, line in enumerate(hip_code.strip().split("\n"), 1):
            print(f"  {D}{i:>3}{N} {line}")
    else:
        print(f"  {Y}No HIP code produced{N}")
        print(f"  {D}The pipeline ran but the coder generated empty output.{N}")

    # ── Step 3: Compile check (if hipcc available) ──
    if HIPCC and hip_code:
        print_stage(3, "Compiling with hipcc", "real GPU verification on MI300X")

        tmp_dir = ROOT / "jupyter_build"
        tmp_dir.mkdir(exist_ok=True)
        src_file = tmp_dir / f"{kernel_name}.hip.cpp"
        src_file.write_text(hip_code, encoding="utf-8")
        exe_file = tmp_dir / kernel_name

        import subprocess
        compile_result = subprocess.run(
            [HIPCC, "-o", str(exe_file), str(src_file)],
            capture_output=True, text=True, timeout=30,
        )

        if compile_result.returncode == 0:
            print(f"  {G}✓ Compiled successfully!{N}")
            # Try running
            run_result = subprocess.run(
                [str(exe_file)], capture_output=True, text=True, timeout=10
            )
            if run_result.returncode == 0:
                print(f"  {G}✓ Ran successfully! Output:{N}")
                for line in run_result.stdout.strip().split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {R}✗ Runtime failed (exit {run_result.returncode}):{N}")
                for line in run_result.stderr.strip().split("\n")[:5]:
                    print(f"    {Y}{line}{N}")
        else:
            print(f"  {R}✗ Compile failed:{N}")
            for line in compile_result.stderr.strip().split("\n")[:8]:
                print(f"    {Y}{line}{N}")

        # Cleanup
        if src_file.exists(): src_file.unlink()
        if exe_file.exists(): exe_file.unlink()
    elif not HIPCC:
        print_stage(3, "Compile check", "⚠️ hipcc not found — skipping")
    else:
        print_stage(3, "Compile check", "⚠️ no HIP code to compile — skipping")

    # ── Step 4: Summary ──
    print()
    box("Summary", [
        f"  {B}Input:{N}      {cu_path.name}",
        f"  {B}HIP code:{N}    {'✓ ' + str(len(hip_code)) + ' chars' if hip_code else '✗ none'}",
        f"  {B}Cost:{N}       ${cost:.4f}",
        f"  {B}Time:{N}       {elapsed:.1f}s",
        f"  {B}Verdict:{N}    {result}",
        f"",
        f"  {D}Tip: save with{N}",
        f"  {D}    %%writefile my_port.hip.cpp{N}",
    ])

if __name__ == "__main__":
    run()
