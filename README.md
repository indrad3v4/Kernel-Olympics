# 🚀 Kernel Olympics

### *Breaking Vendor Lock-In with Multi-Agent AI*

<p align="center">

Built during the 🏆 **AMD Developer Hackathon 2026** — An AI-powered **Autonomous GPU Migration Platform** that helps developers move CUDA applications to AMD ROCm using multi-agent reasoning.

</p>

---

## 🌍 Why Kernel Olympics Exists

The biggest challenge preventing organizations from adopting AMD GPUs isn't hardware. It isn't performance. It isn't software quality.

It's **migration**.

Thousands of CUDA applications remain locked to the NVIDIA ecosystem because migrating production GPU software is expensive, risky, time-consuming, and requires highly specialized expertise.

Kernel Olympics changes that.

Instead of acting as another AI coding assistant, Kernel Olympics functions as an **Autonomous GPU Migration Platform** that understands an entire CUDA project, plans the migration, performs intelligent code transformations, verifies correctness, benchmarks performance, explains every change, and produces a production-ready migration report.

> **Reduce GPU migration from weeks of engineering work to an AI-assisted workflow that developers can trust.**

---

## ✨ What Makes Kernel Olympics Different?

Kernel Olympics is **not**:
- ❌ another AI chatbot
- ❌ another GitHub Copilot
- ❌ another wrapper around hipify
- ❌ simple prompt engineering

Instead, Kernel Olympics behaves like an experienced GPU engineering team. Multiple specialized AI agents collaborate to understand an entire repository before making any modifications. Rather than translating files one by one, the platform reasons about architecture, dependencies, compatibility, performance implications, unsupported APIs, testing strategy, documentation, and migration risks.

Every decision is transparent. Every modification is explainable. Every migration produces evidence.

---

## 🎯 Vision

Imagine opening any CUDA repository and clicking one button. Within minutes you receive:
- ✅ Complete repository analysis
- ✅ Migration readiness score
- ✅ CUDA compatibility report
- ✅ Intelligent migration strategy
- ✅ Automatically migrated ROCm code
- ✅ Performance benchmark
- ✅ Validation report
- ✅ Pull Request
- ✅ Human-readable documentation

Instead of asking *"Can this file be converted?"*, Kernel Olympics answers: **"Your entire project is now ready for AMD GPUs."**

---

## 🧠 The Problem

Today, migrating GPU software is difficult because developers must manually:
- Understand large CUDA codebases
- Identify unsupported APIs
- Rewrite kernels
- Replace memory management
- Verify correctness
- Debug compilation failures
- Benchmark performance
- Write migration documentation

Even experienced GPU developers spend days or weeks doing this. The process is repetitive, error-prone, expensive, and hard to scale.

---

## 💡 Our Solution

Kernel Olympics introduces an AI-native migration workflow. Instead of treating migration as file conversion, the platform treats it as an engineering reasoning problem. The system first understands the repository, identifies architectural patterns, analyzes dependencies, detects unsupported CUDA features, proposes migration strategies, executes intelligent transformations, validates results, benchmarks performance, and generates a comprehensive migration report explaining every decision.

This creates a migration pipeline that is explainable, repeatable, and significantly easier for developers to trust.

---

## 🏆 Why This Matters

GPU ecosystems are becoming increasingly diverse. Organizations want flexibility. Researchers want portability. Companies want freedom from vendor lock-in.

Kernel Olympics enables that transition by making GPU migration dramatically easier. The long-term vision extends far beyond CUDA → ROCm — future versions can support CUDA → SYCL, Vulkan Compute, OpenCL, Metal, DirectML, and more. The platform becomes a **universal GPU migration engine** rather than a single-purpose converter.

---

## The $10B Problem

AMD GPUs (MI300X) outperform NVIDIA on price/performance. Yet enterprises stay on NVIDIA because **20% of CUDA code won't port to ROCm** — custom kernels, warp-sensitive logic, library-specific calls. hipify handles the easy 80%. The remaining 20% is a manual, weeks-long slog per project.

**AMD's #1 adoption blocker isn't hardware — it's software migration friction.**

The broader market is bigger: GPU architectures multiply (NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple Metal, custom NPUs) while the talent pool doesn't. Every hardware generation creates a **$2B+ migration tax** across the industry — teams rewriting kernels by hand instead of building new products.

---

## What is Kernel Olympics?

**An autonomous multi-agent pipeline** that takes GPU kernel code from any source architecture, ports it to any target, compiles, runs on real hardware, and verifies correctness — **zero human intervention**.

First supported path: **CUDA → ROCm/HIP** (the highest-demand migration today). Architecture-agnostic design means adding new paths (oneAPI, Metal) is a config change, not a rewrite.

```bash
# One command: CUDA in → HIP + proof + PASS/FAIL out
make port CU_FILE=sample_kernels/cuda/nvidia_shfl_scan.cu
```

## 🎬 Demo

**🐙 Live: [kernel-olympics-production.up.railway.app](https://kernel-olympics-production.up.railway.app)** — Upload a CUDA kernel and watch the pipeline port it live.

<p align="center">
  <a href="https://github.com/indrad3v4/Kernel-Olympics/blob/main/amd_demo.gif">
    <img src="amd_demo.gif" alt="Kernel Olympics Pipeline Demo" width="720">
  </a>
  <br>
  <sub>Full pipeline: CUDA source → 4-LLM loop ports it → hipcc compile → AMD MI300X run → <b>PASSED ✓</b></sub>
</p>

## 📈 Quick Stats

| Metric | Value |
|--------|-------|
| Pipeline budget | 1,800s (30 min) |
| Max iterations | 10 (compile-fix loop) |
| LLM cost per run | ~$0.09 |
| Cache hit speed | ~0.2ms |
| Tests | **665 passing** |
| CI/CD | ✅ Automated (GitHub Actions) |
| Hardware target | AMD MI300X (192GB HBM3, CDNA3) |

## 🧠 Multi-Agent Architecture

```
                    ┌─────────────────────────────────┐
                    │      CUDA Kernel (.cu)           │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  1. Risk Classifier          │
                    │  (RED/YELLOW/GREEN)          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  2. Pattern Memory Cache     │
                    │  (trigram ~0.2ms, 60,000×)   │
                    └──────────────┬──────────────┘
                                   │
                    ╔══════════════╧══════════════╗
                    ║      LLM Agent Loop         ║
                    ║  (full auto, no human)      ║
                    ║                              ║
                    ║  3. DeepSeek-v4-Pro          ║
                    ║     → architecture plan      ║
                    ║                              ║
                    ║  4. GLM-5.2                  ║
                    ║     → HIP kernel code gen    ║
                    ║                              ║
                    ║  5. Kimi K2.7                ║
                    ║     → 3-gate eval + refine   ║
                    ║       (compile-fix loop)     ║
                    ║                              ║
                    ║  6. Gemma 4 / DeepSeek-v4    ║
                    ║     → final verification     ║
                    ╚══════════════╧══════════════╝
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  7. REAL AMD GPU            │
                    │  hipcc + run + numerical    │
                    │  diff verification           │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   ✅ HIP Kernel + Proof     │
                    └─────────────────────────────┘
```

## 🔬 Key Innovation: Pattern Memory Cache

Instead of calling expensive LLMs for every kernel, we **cache porting patterns as trigram vectors**:

| Metric | Without Cache | With Cache | Speedup |
|--------|:------------:|:----------:|:-------:|
| Pattern lookup | N/A | **0.2ms** | — |
| LLM call (simulated) | ~12s | 0.2ms | **60,000×** |
| Verified with live API | — | ✓ measured | ✓ |

## 🚀 Quick Start

### Live Demo (no install required)

→ **[kernel-olympics-production.up.railway.app](https://kernel-olympics-production.up.railway.app)** — Upload a `.cu` file, see the autonomous pipeline port it to HIP in real time.

### Local Setup

```bash
git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics
pip install -r requirements.txt

# Run the full pipeline on a sample kernel
make port CU_FILE=sample_kernels/cuda/warp_reduce.cu

# Or try a more complex warp-level scan kernel
make port CU_FILE=sample_kernels/cuda/nvidia_shfl_scan.cu
```

## 📋 Makefile Targets

| Target | Description |
|--------|-------------|
| `help` | Show all available targets |
| `install` | Create venv + install deps |
| `port` | Pipeline on one kernel: `make port CU_FILE=path.cu` |
| `port-all` | Pipeline on ALL sample kernels |
| `compile` | hipcc proof harness + compile |
| `run` | Run compiled binary on AMD GPU |
| `pipeline` | Full cycle: port → compile → run |
| `pipeline-heavy` | Extended budget (1,800s) |
| `test` | Run 665 pytest tests |
| `demo` | Live demo with recording |
| `inspect` | Inspect spec/ported kernel/proof |
| `debug-kernel` | Interactive kernel explorer |
| `retry` | Re-run a single pipeline stage |

## 🐞 Debug Mode

Three levels of debugging for when things go wrong:

```bash
# Inspect specs and artifacts
make inspect CU_FILE=sample_kernels/cuda/warp_reduce.cu
make inspect PORTED=ported_kernels/warp_reduce.hip.cpp

# Interactive kernel exploration
make debug-kernel CU_FILE=sample_kernels/cuda/warp_reduce.cu

# Retry a single stage
make retry CU_FILE=sample_kernels/cuda/warp_reduce.cu STAGE=port
```

## 🧪 Running Tests

```bash
make test              # full suite
make test-verbose      # with progress
```

## 🔧 Pipeline Architecture Details

### CUDA → HIP Transformations

| CUDA Intrinsic | HIP Equivalent | Action |
|:--------------|:--------------|:-------|
| `__shfl_up_sync(mask, val, d, w)` | `__shfl_up(val, d, width)` | Mask dropped |
| `cudaMalloc()` | `hipMalloc()` | 1:1 rename |
| `cudaMemcpy()` | `hipMemcpy()` | 1:1 rename |
| `findCudaDevice()` | `hipGetDevice()` | SDK strip |
| `sdkCreateTimer()` | — | Removed (NOP) |
| `threadIdx.x` | `hipThreadIdx.x` | Namespace add |

### Known Issues Handled

- **NVIDIA SDK symbols** — auto-detected and stripped by `verifier.py`
- **Wave64 divergence** — `warpSize` constant used instead of hardcoded 64
- **SIGSEGV from host-code symbols** — sanitizer in verifier catches at compile time

## 👥 Team

| Role | Member | Focus |
|:-----|:-------|:------|
| 🚀 Lead | indradev_ | Architecture, pipeline orchestrator |
| ⚙️ Infra | cation | CI/CD, AMD cloud, Jupyter integration |
| 🔬 Kernel | Bromine185 | CUDA kernel analysis, warp primitives |
| 📝 Docs | _dD | Documentation, demo recording |
| 🔧 Infra | icodemun44 | Tooling, automation |
| 🧪 CI | meteorite67 | GitHub Actions, test suite |
| 🏭 AMD | Aahil-Riyaz (Satoru) | AMD MI300X testing, ROCm debugging |

## 📄 License

MIT

---

<p align="center">
  <sub>Built for the AMD Developer Hackathon ACT II · Track 3 (Open Innovation)</sub>
  <br>
  <a href="https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii">
    <img src="https://img.shields.io/badge/lablab.ai-AMD%20Hackathon%20ACT%20II-blueviolet?style=flat-square" alt="lablab.ai">
  </a>
</p>
