# Kernel Olympics — Demo Video Script
# Total: ~3 minutes

# ── SCENE 1: System (15s) ─────────────────────────────────────
rocm-smi

# ── SCENE 2: Start Gemma 4 on AMD GPU (2 min background) ─────
bash scripts/start_gemma_vllm.sh

# ── SCENE 3: Pipeline on NVIDIA cuda-samples (30s) ────────────
python3 src/main.py --nvidia-sample --fresh

# ── SCENE 4: hipcc compile + run on MI300X (30s) ─────────────
hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
/tmp/shfl_test

# ── SCENE 5: Cache hit — second run instant (10s) ────────────
python3 src/main.py --nvidia-sample
