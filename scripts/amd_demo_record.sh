#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# AMD Real-Hardware Demo Recording — Jupyter → MI300X
# Records every Makefile command executing on real AMD GPU.
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/Kernel-Olympics
source .venv/bin/activate 2>/dev/null || true

amd() { python3 scripts/amd_run.py "$1" "${2:-120}" 2>&1; }

hr() {
    printf '\n%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf '  %s\n' "$*"
    printf '%s\n\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── INTRO ────────────────────────────────────────────────
clear 2>/dev/null || true
hr "KERNEL OLYMPICS — AMD TRACK 3"
echo ""
echo "  Real hardware: AMD MI300X (ROCm 7.2)"
echo "  Pipeline:      DeepSeek v4 (plan) → GLM-5.2 (code) → Kimi K2.7 (eval) → Gemma 4 (verify)"
echo "  Repo:          github.com/indrad3v4/Kernel-Olympics"
echo "  Recording:     $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# ── PHASE 1: Environment Check ─────────────────────────
hr "PHASE 0: Environment Check"
echo "  $ hipconfig --full (head -3)"
python3 scripts/amd_run.py 'hipconfig --full 2>/dev/null | head -3' 10
echo ""
echo "  $ git log --oneline -3"
python3 scripts/amd_run.py 'git -C /workspace/Kernel-Olympics log --oneline -3' 10
echo ""

# ── PHASE 2: make help ──────────────────────────────────
hr "PHASE 1: make help — All Targets"
echo "  $ make help"
python3 scripts/amd_run.py 'make help 2>&1 | head -40' 30
echo ""

# ── PHASE 3: make compile ──────────────────────────────
hr "PHASE 2: make compile — hipcc on MI300X"
echo "  Generates proof harness → compiles with AMD hipcc"
echo ""
echo "  $ make compile"
python3 scripts/amd_run.py 'make compile 2>&1' 60
echo ""

# ── PHASE 4: make run ──────────────────────────────────
hr "PHASE 3: make run — Execute on MI300X GPU"
echo "  Runs compiled binary on real AMD silicon"
echo ""
echo "  $ make run"
python3 scripts/amd_run.py 'make run 2>&1' 30
echo ""

# ── SUMMARY ──────────────────────────────────────────────
hr "DEMO COMPLETE"
echo "  ✓  CUDA→HIP port          — LLM multi-agent loop"
echo "  ✓  hipcc compile          — ROCm 7.2 on MI300X"
echo "  ✓  GPU execution          — Real AMD hardware"
echo "  ⚠  Correctness check      — Runtime bug caught"
echo ""
echo "  git log --oneline -3"
python3 scripts/amd_run.py 'git -C /workspace/Kernel-Olympics log --oneline -3' 10
echo ""
echo "  https://github.com/indrad3v4/Kernel-Olympics"
echo ""
