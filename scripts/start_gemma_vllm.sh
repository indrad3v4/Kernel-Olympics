#!/bin/bash
# Start Gemma 4 on AMD GPU via vLLM — for AMD Jupyter (notebooks.amd.com)
#
# Usage:
#   bash scripts/start_gemma_vllm.sh          # First run (installs + starts)
#   bash scripts/start_gemma_vllm.sh --quick   # Skip install, just start
#
# The pipeline auto-detects localhost:8000 and uses it as verifier.
# If Gemma is unavailable → falls back to DeepSeek automatically.

set -e

MODEL="${GEMMA_MODEL:-google/gemma-4-31b-it}"
PORT=8000
CACHE_DIR="${HF_HOME:-/workspace/.cache/huggingface}"
NOHUP_LOG="/tmp/vllm_gemma4.log"

echo "╔═ Gemma 4 on AMD GPU ═══════════════════════════════╗"
echo "║"

# ── Step 1: Check AMD GPU ──
if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep "Product Name" | head -1 | sed 's/.*Product Name\s*://' | xargs)
    echo "║ ✅ AMD GPU: ${GPU_NAME:-detected}"
else
    echo "║ ⚠️ rocm-smi not found — GPU check skipped"
fi

# Show memory (useful for model size check)
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM" | head -3 || true
fi

# ── Step 2: Install vLLM (if not already) ──
INSTALL_VLLM=false
if python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
    echo "║ ✅ vLLM $(python3 -c 'import vllm; print(vllm.__version__)') already installed"
else
    echo "║ ⚠️ vLLM not found — installing..."
    echo "║    This takes 3-5 minutes (pre-compiled wheel)."
    echo "║    Only needed ONCE per environment."
    INSTALL_VLLM=true
fi

if [ "$INSTALL_VLLM" = true ] && [ "$1" != "--quick" ]; then
    # Try pre-compiled nightly wheel first (fast!)
    echo "║    Trying pre-compiled wheel..."
    pip install vllm --pre \
        --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm721 \
        --break-system-packages \
        -q 2>&1 | tail -3 || true

    # If that failed, try standard pip
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo "║    Nightly wheel unavailable — trying standard pip..."
        pip install vllm --break-system-packages -q 2>&1 | tail -3 || true
    fi

    # Verify installation
    if python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        echo "║ ✅ vLLM $(python3 -c 'import vllm; print(vllm.__version__)') installed"
        echo "║    (This install is now cached — next run will be instant)"
    else
        echo "║ ❌ vLLM installation failed."
        echo "║    Pipeline will use DeepSeek fallback instead."
        echo "║    To retry: call this script again after the env is ready."
        echo "╚══════════════════════════════════════════════════════╝"
        exit 1
    fi
fi

# ── Step 3: Kill any existing vLLM server ──
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 1

# ── Step 4: Create cache dir ──
mkdir -p "$CACHE_DIR"

# ── Step 5: Start vLLM with Gemma 4 ──
echo "║"
echo "║ Starting Gemma 4 ($MODEL) on port $PORT..."
echo "║ Model cache: $CACHE_DIR"
echo "║ Log: $NOHUP_LOG"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

export HF_HOME="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "⏳ Waiting for Gemma to start (can take 2-10 min)..."
echo "   Check progress: tail -f $NOHUP_LOG"

for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ Gemma 4 is READY on http://localhost:$PORT"
        echo "   PID: $PID"
        echo ""
        echo "   Now run: python3 src/main.py --nvidia-sample --fresh"
        echo "   Pipeline will use Gemma 4 as verifier ✅"
        exit 0
    fi
    # Show progress dots every 10 seconds
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still waiting... (${i}0s elapsed)"
        tail -1 "$NOHUP_LOG" 2>/dev/null | head -c 80
        echo ""
    fi
    sleep 10
done

echo ""
echo "❌ Gemma failed to start in 20 minutes."
echo "   Last log lines:"
tail -5 "$NOHUP_LOG" 2>/dev/null
echo ""
echo "   Pipeline will fall back to DeepSeek automatically."
