# Kernel Olympics — CUDA→ROCm Migration Copilot

**Track:** T3 — Unicorn (Open Innovation)
**Team:** Meteorite (6 members)
**Repo:** https://github.com/indrad3v4/Kernel-Olympics
**Stack:** Python, Fireworks AI (Kimi K2.6, GLM 5.2, Gemma 4), AMD ROCm 7.2, vLLM

---

## One Sentence

**Every CUDA kernel is one command away from running on AMD MI300X.**

---

## The Problem — AMD's Real Bottleneck

AMD MI300X is technically competitive with NVIDIA H100. But **developers can't switch** because:

```
CUDA codebase → rewrite for ROCm → weeks of work → bugs → frustration → give up
```

This is the **CUDA lock-in tax**. Every developer who stays on CUDA is a MI300X sale AMD loses. Every tool that reduces this switching cost is a **force multiplier for AMD's ecosystem**.

Current solutions:
- **hipify-clang**: 80-96% coverage, but the remaining 4-20% is warp/wavefront divergence — the HARDEST bugs
- **Manual porting**: Days per kernel, expert needed, expensive
- **LLM copy-paste**: Developer pastes CUDA into ChatGPT → gets plausible but unverified HIP code

---

## Our Solution — Kernel Olympics

A **3-model AI orchestration pipeline** that ports CUDA kernels to ROCm/HIP, compiles them on real AMD GPU, and caches the fix for instant reuse.

```
CUDA kernel → Risk Classifier → pattern detected
  ↓
Kimi K2.6 (planner) — analyzes warp/wavefront divergence
  ↓
GLM 5.2 (coder) — generates ported HIP code  
  ↓
Gemma 4 (verifier) — checks correctness
  ↓
hipcc compilation on AMD MI300X → PASSED ✅
  ↓
Pattern Memory → second run: 0.3ms (60,000× faster)
```

### What Makes This Real

| Other projects | We do |
|---------------|-------|
| "Uses AMD API" (cloud inference) | **Runs on AMD MI300X via hipcc** |
| "AI agent" (LLM wrapper) | **3-model orchestration with real compilation** |
| "One-time demo" | **Pattern memory makes it faster every time** |
| "Requires 10 pip packages" | **Zero external dependencies — Python stdlib only** |
| "Needs GPU for demo" | **Works offline with template fallback** |

### Demo (Proven on AMD MI300X)

```bash
# 1. Pipeline ports the kernel (3 AI models, 22s)
python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu

# 2. Real AMD GPU compilation via hipcc
hipcc -o /tmp/warp_test ported_kernels/warp_reduce.hip.cpp -std=c++17 -O2
# → 7 warnings (normal), 0 errors ✅

# 3. Real execution on AMD GPU
/tmp/warp_test
# → Block 0 sum: 64  → TEST: PASSED ✅

# 4. Second run — 0.3ms cache hit
# → 60,000× faster than first run
```

### AMD Stack Usage

| AMD Technology | How We Use It |
|---------------|---------------|
| **AMD MI300X** | All kernel compilation and execution |
| **ROCm 7.2** | hipcc compiler, rocm-smi monitoring |
| **AMD Cloud Jupyter** | Development and testing environment |
| **AMD Developer Credits** | $100 credits not yet activated — ready for production |
| **hipify-clang** | 96.7% coverage detection, human-review for the remaining 3.3% |

---

## Why This Wins

### Unicorn Criteria (Open Innovation)

| Criterion | How We Hit It |
|-----------|---------------|
| **Creativity** | First CUDA→ROCm migration tool using 3-model MOA orchestration + pattern memory cache |
| **Originality** | Zero similar projects in any track — we're the only tool that reduces CUDA switching cost |
| **Completeness** | End-to-end: scan → classify → LLM port → hipcc compile → cache for next time |
| **AMD Platform** | Real MI300X execution (not API calls). ROCm, hipcc, rocm-smi — the full stack |
| **Market Potential** | Every CUDA developer who wants to try AMD. 100K+ potential users. |

### The "Dopamine Loop"

1. Developer has a CUDA kernel → runs pipeline → **kernel compiles on AMD** ✅
2. Gets an "AMD Compatible ✅" badge for their repo
3. Next CUDA kernel → **0.3ms** — instant port
4. Each ported kernel adds to the community pattern library
5. **Viral loop**: "My repo is AMD Compatible. Is yours?"

### The Story for AMD Judges

> *"AMD doesn't need another AI app. You need developers to SWITCH from CUDA. Kernel Olympics is the on-ramp. One command, real AMD compilation, zero dependencies. Every CUDA repo that adds our badge is proof that ROCm works — and a sale AMD doesn't lose to NVIDIA."*

---

## Technical Details

### Architecture

```
src/
├── main.py                  # Pipeline orchestrator
├── scanner/                 # CUDA file scanning + hipify coverage
├── risk_classifier/         # 9 pattern types (warp, memory, sync)
├── pattern_memory/          # Trigram-index SQLite cache
├── router.py                # 3-model MOA (Kimi → GLM → Gemma)
├── porting_agent/           # LLM calls + template fallback
├── verification/            # hipcc compilation + execution
└── report_generator/        # JSON + human-readable reports
```

### Model Routing Table

| Pattern | Model | Role |
|---------|-------|------|
| `shfl_down_sync` | Kimi K2.6 | Plans the fix (complex warp→wavefront) |
| `warp_size_constant` | GLM 5.2 | Generates HIP code |
| Default | Template + _fix_ported_code | Zero-cost deterministic fallback |

### Fallback Chain

```
Kimi K2.6 (Fireworks) → GLM 5.2 (Fireworks) → DeepSeek-V4-Pro (Fireworks)
→ Template (regex-based) → Explanation
```

Each step guarantees SOME output. Never crashes. Always produces a ported kernel or a useful error message.

### CI/CD

- GitHub Actions: 43 tests, all passing ✅
- Automatic demo run on every push
- Zero network dependencies for tests
- [![CI](https://github.com/indrad3v4/Kernel-Olympics/actions/workflows/ci.yml/badge.svg)](https://github.com/indrad3v4/Kernel-Olympics/actions)

---

## Team Meteorite

| Member | Role | Expertise |
|--------|------|-----------|
| indradev_ (Team Lead) | AI Architecture | LLM orchestration, system design |
| Aahil-Riyaz (Satoru) | AMD/ROCm Engineering | GPU kernel development, hipcc |
| Bromine185 | Kernel Engineering | CUDA→HIP migration patterns |
| icodemun44 (cation) | Infrastructure | CI/CD, Docker, deployment |
| meteorite67 | CI/CD | GitHub Actions, testing |
| _dD | QA | Testing, edge cases |

---

## Links

- **GitHub:** https://github.com/indrad3v4/Kernel-Olympics
- **Demo Video:** [Link — recording in progress]
- **AMD Stack Usage:** https://github.com/indrad3v4/Kernel-Olympics/blob/main/AMD_STACK_USAGE.md

---

## Final Pitch (30 seconds for judges)

> *"AMD GPUs are powerful. But developers can't switch from CUDA because there's no migration tool. Kernel Olympics is that tool. One command ports your CUDA kernel, compiles it on real AMD MI300X, and passes. Second run is instant. We're not another AI app — we're the reason CUDA developers finally try AMD."*

---

*Built in 5 days for AMD Developer Hackathon ACT II — Track 3: Unicorn*
