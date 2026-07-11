# Kernel Olympics 🏆

**CUDA→ROCm Migration Copilot** — Ship AMD-ready code in minutes, not months.

<div align="center">

[![AMD Compatible](https://img.shields.io/badge/AMD-COMPATIBLE%20✓-ed1c24?style=flat&logo=amd)](https://github.com/indrad3v4/Kernel-Olympics)
[![CI](https://github.com/indrad3v4/Kernel-Olympics/actions/workflows/ci.yml/badge.svg?style=flat)](https://github.com/indrad3v4/Kernel-Olympics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat)](LICENSE)

</div>

## The $10B Problem

AMD GPUs (MI300X) outperform NVIDIA on price/performance. Yet enterprises stay on NVIDIA because **20% of CUDA code won't port to ROCm** — custom kernels, warp-sensitive logic, library-specific calls. hipify handles the easy 80%. The remaining 20% is a manual, weeks-long slog per project.

AMD's #1 adoption blocker isn't hardware — it's software migration friction.

## Kernel Olympics

**One kernel at a time.** We scan, classify, auto-port, compile, and verify CUDA→ROCm migration on real AMD hardware.

```
Submit CUDA kernel → Risk classifier (RED/YELLOW/GREEN based on warp/wavefront
patterns) → Pattern memory (trigram index, 0.2ms) → DeepSeek-v4-pro plans →
GLM-5.2 codes → Kimi K2.7 evaluates ⟲ compile-fix retry loop (up to 10
attempts) → Gemma 4 / DeepSeek-v4-pro final verification → Verify on AMD GPU
(real hipcc + run + numeric diff)
```

**Demo:** `python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu`

### What makes it different

| Feature | hipify | Kernel Olympics |
|---------|--------|----------------|
| Coverage | ~80% (syntax only) | **Pattern-aware** — catches warp/wavefront divergence |
| Verification | Manual | **Auto-compile + run + diff** on AMD GPU |
| Memory | Stateless | **Trigram cache** — cache hits skip the LLM (0.2ms). Up to ~60,000× vs a simulated 12s LLM baseline; real ratio measured with a live API key |
| LLM Pipeline | None | **4-model MOA** — DeepSeek-v4-pro(planner) → GLM-5.2(coder) → Kimi K2.7(evaluator) ⟲ compile-fix loop → Gemma 4 / DeepSeek-v4-pro(verifier) |

## 🚀 Try it

```bash
git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics
pip install -r requirements.txt
python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu
python3 src/main.py --input sample_kernels/cuda/new_kernel.cu
```

## 🐞 Debug Mode

When a translation fails, you should never have to re-run the pipeline — three
debug modes let you inspect, fix, and retry specific stages:

### `inspect` — Read a spec, ported kernel, or proof

```bash
# Inspect the spec for a given CUDA file
make inspect CU_FILE=sample_kernels/cuda/warp_reduce.cu

# Inspect the ported kernel
make inspect PORTED=ported_kernels/warp_reduce.hip.cpp

# Inspect the compiled proof
make inspect BINARY=/tmp/warp_reduce_kernel_proof
```

### `debug-kernel` — Interactive kernel exploration

```bash
make debug-kernel CU_FILE=sample_kernels/cuda/warp_reduce.cu
```

Opens a structured view of the kernel's warp usage, CUDA constructs, and
classifier output — no need to grep through raw files.

### `retry` — Re-run a single pipeline stage

```bash
make retry CU_FILE=sample_kernels/cuda/warp_reduce.cu STAGE=port
```

Re-runs only the porting stage on an existing spec. Valid stages: `classify`,
`port`, `compile`, `verify`, `diff`.

## 🧪 Testing

```bash
make test              # full test suite
make test-verbose      # with progress
```

## 📦 Makefile targets

| Target | Description |
|--------|-------------|
| `help` | Show all available targets |
| `install` | Create venv and install dependencies |
| `port` | Run pipeline (CUDA→HIP) on one kernel: `make port CU_FILE=path.cu` |
| `port-all` | Run pipeline on *all* sample kernels |
| `compile` | hipcc device-only proof harness and compile |
| `run` | Run the compiled binary |
| `pipeline` | Full cycle: port → compile → run |
| `demo` | Run demo with OBS recording |
| `record` | Record demo with asciinema |
| `push` | Git push with message |
| `inspect` | Inspect spec, ported kernel, or proof |
| `debug-kernel` | Interactive kernel exploration |
| `retry` | Re-run a single pipeline stage |

## 🔧 Pipeline Architecture

```
                    ┌─────────────────────────────────┐
                    │     CUDA Kernel Source (.cu)     │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Risk Classifier           │
                    │  (RED/YELLOW/GREEN)          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Pattern Memory            │
                    │  (trigram cache, ~0.2ms)     │
                    └──────────────┬──────────────┘
                                   │
                    ╔══════════════╧══════════════╗
                    ║       LLM Agent Loop        ║
                    ║                              ║
                    ║  1. DeepSeek-v4-pro plans    ║
                    ║     → structured porting     ║
                    ║       checklist              ║
                    ║                              ║
                    ║  2. GLM-5.2 codes            ║
                    ║     → generates HIP kernel   ║
                    ║                              ║
                    ║  3. Kimi K2.7 evaluates      ║
                    ║     → compile-fix loop        ║
                    ║       (up to 10 attempts)     ║
                    ║                              ║
                    ║  4. Gemma 4 / DeepSeek-v4-pro║
                    ║     final verification        ║
                    ╚══════════════╧══════════════╝
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Real AMD GPU Verification │
                    │  (hipcc + run + diff)        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    HIP Kernel (.hip.cpp)     │
                    │  + proof artifact            │
                    └─────────────────────────────┘
```

### Pipeline output from the terminal

```
2026-07-11 02:44:23 | ── route() ── DeepSeek-v4-pro (plan) → GLM-5.2 (code) → Kimi K2.7 (evaluate) ──
Extracting kernel: warp_reduce from sample_kernels/cuda/warp_reduce.cu
  ── After planning: (plan recap, 2242 chars) ──
  ── After code gen: (port raw, 48 changes) ──
  ── Kimi K2.7 feedback ──
  🔧 HIP intrinsic fix: 2 CUDA __syncthreads call(s) converted to HIP
  ── hipcc compile check ──
  ✅ PASS — warp_reduce → ROCm ready
  ── Gemma 4 final verification ──
  ── route() done: SUCCESS — elapsed 47.7s
```

## 🐛 Known Issues

### NVIDIA SDK test harness symbols in self-contained programs

The LLM agent sometimes ports the full NVIDIA CUDA sample including the test
harness (``helper_cuda.h``, ``helper_timer.h``). These SDK functions don't
exist on AMD. The pipeline detects and strips them automatically:

| NVIDIA Symbol | Pipeline Action |
|--------------|----------------|
| `findCudaDevice()` | Replaced with `hipGetDevice()` |
| `sdkCreateTimer()` / `sdkGetTimerValue()` | Removed (NOP) |
| `StopWatchInterface` | Stripped |
| `EXIT_WAIVED` | → `EXIT_SUCCESS` |
| `hipDeviceGet()` | → `hipGetDevice()` |
| Undefined kernel launches | → caught by compile-fix loop |

### Warp size divergence (NVIDIA 32/64 vs AMD wave32)

NVIDIA modern GPUs use warpSize=32; AMD RDNA3 (gfx1100) also uses wave32 but
some AMD CDNA GPUs use wave64. The pipeline's GLM-5.2 coder is prompted to
always use the ``warpSize`` device constant instead of hardcoding 64.

## 📄 License

MIT
