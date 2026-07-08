#!/bin/bash
# Start Gemma 4 31B IT on AMD GPU — for AMD Jupyter (notebooks.amd.com/hackathon)
#
# Strategy:
#   1. Load amdgpu kernel module (creates /dev/kfd)
#   2. Use Python 3.12 (vLLM ROCm wheel requires 3.12, not 3.13)
#   3. Install vLLM ROCm nightly wheel (pre-compiled, 30s)
#   4. Download + serve Gemma 4 31B IT on localhost:8000
#
# Pipeline auto-detects localhost:8000 and uses Gemma 4 as verifier.
# If Gemma is unavailable → falls back to DeepSeek automatically.
#
# Usage:
#   bash scripts/start_gemma_vllm.sh          # Full setup
#   bash scripts/start_gemma_vllm.sh --quick   # Skip install, just start server

set -e

NOHUP_LOG="/tmp/vllm_gemma4.log"
PORT=8000
MODEL="google/gemma-4-31b-it"
PY312="/usr/bin/python3.12"
VENV_DIR="/workspace/.ko_venv"
HF_CACHE="${HF_HOME:-/workspace/.cache/huggingface}"

echo "╔═ Gemma 4 on AMD GPU ═══════════════════════════════╗"
echo "║"

# ── 1. Check AMD GPU ──
if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep "Product Name" | head -1 | sed 's/.*Product Name\s*://' | xargs)
    echo "║ ✅ AMD GPU: ${GPU_NAME:-detected}"
else
    echo "║ ⚠️ rocm-smi not found — GPU check skipped"
fi

# ── 2. Load amdgpu kernel module (creates /dev/kfd) ──
if [ ! -c /dev/kfd ]; then
    echo "║ ⚠️ /dev/kfd not found — loading amdgpu module..."
    sudo modprobe amdgpu 2>/dev/null && echo "║ ✅ amdgpu module loaded" || echo "║ ⚠️ Could not load amdgpu (might need sudo)"
    sleep 1
fi

if [ -c /dev/kfd ]; then
    echo "║ ✅ /dev/kfd available"
else
    echo "║ ⚠️ /dev/kfd still unavailable — GPU runtime may not work"
fi
ls /dev/dri/render* 2>/dev/null && echo "║ ✅ /dev/dri available" || echo "║ ⚠️ /dev/dri not found"

# ── 3. Ensure Python 3.12 (vLLM ROCm wheel requires it) ──
if [ ! -f "$PY312" ]; then
    echo "║ ⚠️ Python 3.12 not found — installing..."
    apt-get update -qq && apt-get install -y -qq python3.12 python3.12-venv python3.12-dev 2>&1 | tail -1
fi

if [ ! -f "$PY312" ]; then
    echo "║ ❌ Python 3.12 install failed. Trying uv..."
    pip install uv -q --break-system-packages 2>/dev/null
    uv python install 3.12 2>&1 | tail -1 || true
    PY312=$(uv python find 3.12 2>/dev/null || echo "/usr/bin/python3.12")
fi

echo "║ ✅ Python: $($PY312 --version 2>&1)"

# ── 4. Create venv + install vLLM ROCm wheel (once) ──
if [ ! -f "$VENV_DIR/bin/activate" ] || [ "$1" != "--quick" ]; then
    echo "║ Creating venv at $VENV_DIR..."
    "$PY312" -m venv "$VENV_DIR" --clear 2>/dev/null || "$PY312" -m venv "$VENV_DIR"
    
    echo "║ Installing vLLM with ROCm support (pre-compiled wheel)..."
    source "$VENV_DIR/bin/activate"
    
    # Try nightly ROCm wheel first (fastest — pre-compiled)
    pip install -q --upgrade pip 2>/dev/null
    pip install vllm --pre \
        --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm721 \
        -q 2>&1 | tail -1 || true
    
    # Verify
    if python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        echo "║ ✅ vLLM $(python3 -c 'import vllm; print(vllm.__version__)') with ROCm support"
    else
        echo "║ ⚠️ Nightly wheel failed — trying stable ROCm wheel..."
        pip install vllm \
            --extra-index-url https://wheels.vllm.ai/rocm/ \
            -q 2>&1 | tail -1 || true
    fi
    
    deactivate
else
    echo "║ ✅ vLLM venv already exists (use --quick to skip install)"
fi

# ── 5. Kill existing server ──
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 1

# ── 6. Start Gemma 4 via vLLM ──
echo "║"
echo "║ Starting $MODEL on port $PORT..."
echo "║ Cache: $HF_CACHE"
echo "║ Log:   $NOHUP_LOG"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"

source "$VENV_DIR/bin/activate"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "⏳ Waiting for Gemma to start (2-10 min)..."
echo "   tail -f $NOHUP_LOG"

for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ Gemma 4 READY → http://localhost:$PORT"
        echo "   PID: $PID"
        echo ""
        echo "   Run: source $VENV_DIR/bin/activate && python3 src/main.py --nvidia-sample --fresh"
        echo "   Pipeline uses Gemma 4 as verifier ✅"
        deactivate
        exit 0
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Waiting... (${i}0s)"
        tail -1 "$NOHUP_LOG" 2>/dev/null | head -c 80
        echo ""
    fi
    sleep 10
done

deactivate
echo ""
echo "❌ Gemma failed to start. Log:"
tail -5 "$NOHUP_LOG" 2>/dev/null
echo ""
echo "   Pipeline will use DeepSeek fallback automatically."
