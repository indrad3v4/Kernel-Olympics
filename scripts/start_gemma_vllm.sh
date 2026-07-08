#!/bin/bash
# Start Gemma 4 31B IT on AMD GPU via vLLM
# Run this FIRST on the AMD Jupyter notebook before using the pipeline
#
# Usage: bash scripts/start_gemma_vllm.sh
# Pipeline will auto-detect and use localhost:8000 as Gemma verifier

set -e

echo "╔═ Gemma 4 on AMD GPU ═══════════════════════════════╗"
echo "║ Checking environment...                             ║"

# Check AMD GPU
if command -v rocm-smi &> /dev/null; then
    echo "║ ✅ AMD GPU detected"
    rocm-smi --showproductname 2>/dev/null | grep "Product Name" || echo "║    (rocm-smi OK)"
else
    echo "║ ⚠️ rocm-smi not found — continuing anyway"
fi

# Check vLLM
if python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
    echo "║ ✅ vLLM $(python3 -c 'import vllm; print(vllm.__version__)')"
else
    echo "║ ⚠️ vLLM not installed — installing..."
    pip install vllm -q --break-system-packages 2>/dev/null || pip install vllm -q
fi

# Kill any existing vLLM server
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 2

echo "║ Starting Gemma 4 31B IT on AMD GPU..."
echo "║ Model: accounts/fireworks/models/gemma-4-31b-it"
echo "║ Port:  localhost:8000"
echo "╚══════════════════════════════════════════════════════╝"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model accounts/fireworks/models/gemma-4-31b-it \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    > /tmp/vllm_gemma4.log 2>&1 &

echo "⏳ Waiting for Gemma to start (this takes 1-5 minutes)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ Gemma 4 is ready on http://localhost:8000"
        echo "   Free Fireworks credits = $(echo $FIREWORKS_API_KEY | head -c 10)..."
        echo ""
        echo "   Now run: python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu"
        echo "   Pipeline will use Gemma 4 as verifier, fallback to DeepSeek"
        exit 0
    fi
    sleep 5
done

echo "❌ Gemma failed to start in 5 minutes. Check /tmp/vllm_gemma4.log"
echo "   Pipeline will fall back to DeepSeek automatically."
