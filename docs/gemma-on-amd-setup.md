# Gemma 4 on AMD Jupyter — Full Setup Strategy

## Start from scratch on notebooks.amd.com

```bash
# ── 0. Accept licenses ──
# (Do this first in a notebook cell or via huggingface-cli)
huggingface-cli login  # or set HF_TOKEN env var

# ── 1. Pull latest code ──
cd /workspace
git config --global http.sslVerify false  # fix cert issue on Jupyter
git clone https://github.com/indrad3v4/Kernel-Olympics.git
# OR if already cloned:
cd /workspace/Kernel-Olympics && git pull origin main

# ── 2. Load AMD GPU kernel module ──
sudo modprobe amdgpu
ls -la /dev/kfd /dev/dri  # verify devices exist

# ── 3. One-command Gemma 4 setup ──
bash scripts/start_gemma_vllm.sh

# ── 4. Run pipeline with Gemma 4 as verifier ──
# (in a new terminal, or after Gemma is ready)
source /workspace/.ko_venv/bin/activate
python3 src/main.py --nvidia-sample --fresh
```

## What the script does:
1. Loads `amdgpu` kernel module → creates `/dev/kfd`
2. Installs Python 3.12 (if missing) — vLLM ROCm wheel requires it
3. Creates venv at `/workspace/.ko_venv`
4. Installs vLLM ROCm nightly wheel (pre-compiled, ~30s)
5. Downloads + serves `google/gemma-4-31b-it` on port 8000

## If Gemma fails → automatic fallback:
Pipeline auto-detects localhost:8000. If down → uses DeepSeek via Fireworks.
No change needed in the pipeline code.
