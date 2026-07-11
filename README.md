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
Submit CUDA kernel → Risk classifier (RED/YELLOW/GREEN based on warp/wavefront patterns) → Pattern memory (trigram index, 0.2ms) → 4-model MOA (DeepSeek plans → GLM ports → Kimi validates → Gemma/DeepSeek verifies) → Verify on AMD GPU (real hipcc + run + numeric diff)
```

**Demo:** `python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu`

### What makes it different

| Feature | hipify | Kernel Olympics |
|---------|--------|----------------|
| Coverage | ~80% (syntax only) | **Pattern-aware** — catches warp/wavefront divergence |
| Verification | Manual | **Auto-compile + run + diff** on AMD GPU |
| Memory | Stateless | **Trigram cache** — cache hits skip the LLM (0.2ms). Up to ~60,000× vs a simulated 12s LLM baseline; real ratio measured with a live API key |
| LLM Pipeline | None | **4-model MOA** — DeepSeek(planner) → GLM(coder) → Kimi(evaluator) + Gemma/DeepSeek(verifier) |

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
LLM calls and a few minutes — just to see *why*. Pass `--debug` and every
intermediate artifact of every stage is written to a session directory:

```bash
python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu --debug
# → debug/session_<timestamp>_<kernel>/
```

Read `summary.md` first. It gives the state-machine timeline, the retry history,
the validation outcomes, a probable root cause and a recommended next action —
and it names the artifact backing each claim, so you can disagree with it by
opening one file.

```
debug/session_20260710T093713_warp_reduce/
├── summary.md            human-readable post-mortem  ← start here
├── metrics.json          per-stage timings, tokens, cost, retry counts
├── state_trace.jsonl     every state transition, with elapsed time and reason
├── timeline.jsonl        every retry event, generation, compile and patch
├── manifest.jsonl        append-only index of every artifact written
├── 01_input/             original CUDA, classifier findings, hipify preprocessing
├── 02_planning/          raw planner responses, parsed plans, prompts
├── 03_translation/       every generation, raw + extracted + discarded reasoning
├── 04_extraction/        strategy used, parser confidence, what was thrown away
├── 05_lexical/           prose/markdown/placeholder detection, pass-fail decision
├── 06_structural/        braces, symbol preservation, structural score
├── 07_symbols/           CUDA vs HIP symbol tables and the diff between them
├── 08_static_analysis/   pre-compile findings (residual CUDA, warp-32 hazards)
├── 09_compiler/          exact hipcc argv, environment, version, FULL stdout/stderr
├── 10_evaluation/        raw evaluator responses, parsed diagnostics, confidence
├── 11_patches/           every repair iteration: before, after, unified diff
└── 12_failure/           self-contained failure package
```

Guarantees: **append-only** (no generation, patch or response is ever
overwritten), **never truncated** (compiler output and raw model responses are
written verbatim), **deterministic** (artifacts are numbered by a monotonic
sequence, not a clock, so two runs are diffable), and **provider-independent**.

Debug Mode is off by default and costs the pipeline nothing when disabled. It
can also be enabled with `KERNEL_OLYMPICS_DEBUG=1`, and rooted elsewhere with
`--debug-dir` or `KERNEL_OLYMPICS_DEBUG_DIR`.

> Session directories contain raw prompts, raw model responses and your kernel
> source. They are gitignored — treat them as local diagnostic artifacts.

## Why This Matters

- **AMD** gets enterprise adoption unblocked — the software gap closes
- **Enterprises** cut migration costs from weeks to minutes
- **ROCm ecosystem** grows faster when porting is frictionless

## Team — Meteorite 🌠

| Role | Member |
|------|--------|
| AI Architect | [indradev_](https://github.com/indrad3v4) |
| AMD/ROCm Engineering | Aahil-Riyaz (Satoru) |
| Kernel Engineering | Bromine185 |
| CI/CD & Testing | meteorite67 |
| Infrastructure | Icodemun44 |
| QA | _dD |

**Built in 5 days for AMD Developer Hackathon ACT II — Track 3 (Unicorn).**

---

*"Like a meteorite, we don't arrive quietly."*
