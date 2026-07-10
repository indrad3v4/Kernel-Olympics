# Kernel Olympics — AMD Developer Hackathon ACT II Submission

**Track:** Track 3 — Unicorn (Full System)
**Team:** Meteorite 🌠
**Team ID:** Team-3793

---

## 📋 Quick Links

| Asset | Link | Status |
|-------|------|--------|
| GitHub Repo | https://github.com/indrad3v4/Kernel-Olympics | ✅ |
| Live Demo | https://endearing-rebirth.up.railway.app | 🚧 Building |
| Demo Video | *Not required by scoring bot* | — |
| Slide Deck | `SUBMISSION_SLIDES.pdf` *(create below)* | 📝 |

---

## 🏆 Project Description

**Kernel Olympics** is a multi-agent CUDA → ROCm migration copilot that ports NVIDIA CUDA kernels to AMD HIP automatically — in minutes, not weeks.

### The Problem

AMD MI300X GPUs outperform NVIDIA on price/performance, but enterprises stay on CUDA because ~20% of custom kernels won't port with `hipify`. The remaining 20% requires manual weeks-long engineering per kernel.

### Our Solution

A 4-LLM agentic loop (DeepSeek → GLM-5.2 → Kimi-K2.7 → Gemma) that:

1. **Scans** CUDA source with hipify-clang (~96.7% coverage)
2. **Classifies** risk by 5 warp/wavefront divergence patterns
3. **Ports** via structured prompting — strip to kernel-only mode for precise offsets
4. **Verifies** through 3 gates: lexical → structural → semantic validator
5. **Compiles** on real AMD hardware (hipcc + MI300X)
6. **Reports** with plain-English summary

### Key Innovations

| Innovation | Impact |
|------------|--------|
| Multi-agent orchestration (4 LLMs in feedback loop) | Auto-refines ported code without human intervention |
| Three-gate validation system | Catches hallucinations before they reach the compiler |
| `_strip_to_kernel_only` pre-processor | Coder sees only kernel code → precise line offsets |
| Pattern memory (trigram cache, 0.2ms) | Skip LLM calls for previously seen patterns |
| Deterministic debug mode | Full artifact trace — append-only, diffable across runs |

---

## 🛠 Tech Stack

| Technology | Usage |
|------------|-------|
| **AMD ROCm 7.2** | GPU verification target — hipcc compilation on MI300X |
| **AMD MI300X** | Hardware verification (via notebooks.amd.com/hackathon) |
| **Fireworks AI API** | Hosted LLM inference on AMD hardware |
| **Python 3.11+** | Pipeline orchestrator (pure stdlib — no framework deps) |
| **FastAPI + Uvicorn** | Web demo frontend (Railway deploy) |
| **Docker** | Containerized deployment |

### LLM Orchestration

| Model | Role | Provider |
|-------|------|----------|
| DeepSeek v4 | Planner — analysis & structural plan | Fireworks |
| GLM-5.2 | Coder — generates HIP source | Fireworks |
| Kimi-K2.7 | Evaluator — validates correctness | Fireworks |
| Gemma | Fallback evaluator | Fireworks |
| DeepSeek v4 Flash | Verdict reporter | Fireworks |

---

## 👥 Team — Meteorite

| Role | Name | GitHub |
|------|------|--------|
| AI Architect (Lead) | indradev_ | https://github.com/indrad3v4 |
| AMD/ROCm Engineering | Aahil-Riyaz (Satoru) | |
| Kernel Engineering | Bromine185 | |
| CI/CD & Testing | meteorite67 | |
| Infrastructure | Icodemun44 | |
| QA | _dD | |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Pipeline runtime (typical kernel) | ~105 seconds |
| Cost per kernel | ~$0.02–0.05 |
| Pattern memory lookup | 0.2ms (~60,000× faster than LLM) |
| Shipable kernels per hour | ~30+ |
| Semantic gate false-positive rate | <1% (31 test cases) |

---

## 🚀 How to Submit

### For lablab.ai form — Copy-Paste Ready

**Project Name:** Kernel Olympics  
**Track:** Track 3 — Unicorn  
**GitHub URL:** https://github.com/indrad3v4/Kernel-Olympics  
**Demo URL:** https://endearing-rebirth.up.railway.app  

---

**Short Description** (1-2 lines, appears in listing):

> Kernel Olympics is a multi-agent AI pipeline that automatically ports NVIDIA CUDA kernels to AMD ROCm/HIP — transforming weeks of manual engineering into ~2 minutes per kernel with 3-stage validation gates and real GPU verification.

---

**Long Description** (for the submission body):

> **Problem**
>
> AMD MI300X GPUs outperform NVIDIA on price/performance, yet enterprises stay on CUDA because ~20% of custom kernels won't port with hipify. The remaining 20% — warp-sensitive logic, shfl operations, library-specific calls — requires manual weeks-long engineering per kernel. This is AMD's #1 adoption blocker, costing enterprises millions in migration friction.
>
> **Solution — Kernel Olympics**
>
> A 4-LLM agentic orchestration pipeline (DeepSeek v4 → GLM-5.2 → Kimi-K2.7 → Gemma) that classifies, ports, validates, and reports on CUDA→ROCm migration in a single automated loop:
>
> 1. **Scan & Classify** — hipify-clang covers ~96.7% syntax; 5 risk-pattern classifiers catch warp/wavefront divergence
> 2. **Smart Porting** — `_strip_to_kernel_only` pre-processor strips host code, giving the coder clean kernel-only source for precise line offsets and reliable output
> 3. **Three-Gate Validation** — lexical gate (rejects prose/markdown), structural gate (braces & symbol preservation), semantic gate (31 test cases, <1% false-positive)
> 4. **Retry & Repair** — failed gates trigger auto-retry with specific error context; pattern memory cache (0.2ms lookup) skips LLM calls for known patterns
> 5. **GPU Verification** — real hipcc compilation on AMD MI300X via ROCm 7.2, binary execution and byte-for-byte diff against CUDA reference output
> 6. **AI Report** — Gemma-generated plain-English summary with timeline, gate outcomes, and next actions
>
> **Key Innovations**
>
> - Multi-agent feedback loop — coder output evaluated and refined without human intervention
> - `_strip_to_kernel_only` — eliminates line offset pollution from host code preamble
> - Three-gate validation — catches LLM hallucinations at lexical, structural, and symbolic levels
> - Pattern memory — trigram index skips LLM for cached patterns (0.2ms vs ~12s LLM)
> - Deterministic debug mode — append-only artifact trace for every pipeline stage
>
> **Performance**
>
> | Metric | Value |
> |--------|-------|
> | Pipeline runtime (typical) | ~105 seconds per kernel |
> | Cost per kernel | ~$0.02–0.05 |
> | Shipable kernels per hour | 30+ |
> | Pattern memory lookup | 0.2ms (60,000× faster than LLM) |
> | Validation gate false-positive | <1% |
>
> **Stack:** Python 3.11+, Fireworks AI API (AMD-hosted LLMs), ROCm 7.2, MI300X GPU, FastAPI web frontend, Docker/Railway.
>
> **Team Meteorite** — 6 engineers. Built in 5 days for AMD Developer Hackathon ACT II.

---

## 📝 Slide Deck Checklist

The scoring bot evaluates the slide deck PDF. Cover:

- [ ] Problem: The $10B AMD adoption gap (20% of CUDA won't port)
- [ ] Solution: 4-LLM agentic loop diagram
- [ ] Architecture: DeepSeek→GLM→Kimi→Gemma flow
- [ ] Three-gate validation (lexical→structural→semantic)
- [ ] Demo flow: Upload .cu → Get HIP code
- [ ] Team slide
- [ ] Live link + QR code

---

*Built in 5 days for AMD Developer Hackathon ACT II — Team Meteorite 🌠*
*"Like a meteorite, we don't arrive quietly."*
