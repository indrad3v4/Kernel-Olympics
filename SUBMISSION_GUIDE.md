# Kernel Olympics — Judge's Guide

## 📌 Submission Info

| Field | Value |
|-------|-------|
| **Event** | AMD Developer Hackathon ACT II |
| **Track** | Track 3 — Unicorn (Open Innovation) |
| **Team** | Kernel Olympics |
| **GitHub** | https://github.com/indrad3v4/Kernel-Olympics |
| **Demo Video** | [AMD GPU Happy Path → PASSED](amd_demo.gif) |

## 🎯 What to Evaluate

### 1. Does It Work? (Technical Implementation)

**The pipeline runs end-to-end on real AMD hardware:**

```bash
# Full pipeline: CUDA → LLM port → hipcc compile → GPU run
make port CU_FILE=sample_kernels/cuda/nvidia_shfl_scan.cu
```

**If you have an AMD GPU with ROCm:**
```bash
make pipeline CU_FILE=sample_kernels/cuda/warp_reduce.cu
```

**Without AMD hardware**, run the LLM pipeline to see the multi-agent orchestration:
```bash
python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu
```

### 2. Innovation (What's New?)

- **4-LLM multi-agent orchestration** — DeepSeek plans, GLM codes, Kimi evaluates, Gemma verifies
- **Pattern memory cache** — Trigram indexing for 60,000× faster repeated ports
- **Self-healing compile loop** — Up to 10 iterations with feedback-driven re-plans
- **NVIDIA SDK symbol sanitizer** — Auto-strips incompatible host code
- **Runtime warpSize detection** — Works on both wave32 (RDNA3) and wave64 (CDNA3)

### 3. AMD Technology Usage

- Code compiles with **hipcc** (ROCm 7.2)
- Runs on **AMD Instinct MI300X**
- Uses **HIP intrinsics** (`__shfl_up`, `hipMalloc`, `hipMemcpy`)
- Runtime device detection via **hipGetDeviceProperties()**

### 4. Completeness

- ✅ **665 passing tests** (run `make test`)
- ✅ **CI/CD pipeline** (GitHub Actions)
- ✅ **Demo video** ([amd_demo.gif](amd_demo.gif))
- ✅ **Makefile** with 20+ targets
- ✅ **11 CUDA sample kernels** to port
- ✅ **Debug tools** (inspect, retry, debug-kernel)
- ✅ **MIT License**

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/router.py` | Main pipeline orchestration (4-LLM loop) |
| `src/verification/verifier.py` | HIP compile + GPU run verification |
| `Makefile` | All targets (port, compile, run, test) |
| `sample_kernels/cuda/*.cu` | Input CUDA kernels to port |
| `ported_kernels/*.hip.cpp` | Output HIP kernels |
| `amd_demo.gif` | Demo video (happy path) |

## 🐞 Debug Mode for Judges

Inspect exactly what the pipeline produced:

```bash
# See the ported kernel
make inspect PORTED=ported_kernels/warp_reduce.hip.cpp

# Interactive kernel exploration
make debug-kernel CU_FILE=sample_kernels/cuda/warp_reduce.cu

# Re-run just the port stage
make retry CU_FILE=sample_kernels/cuda/warp_reduce.cu STAGE=port
```
