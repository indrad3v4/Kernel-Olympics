# Kernel Olympics 🏆

**CUDA→ROCm Migration Copilot** — Ship AMD-ready code in minutes, not months.

---

[![CI](https://github.com/indrad3v4/Kernel-Olympics/actions/workflows/ci.yml/badge.svg)](https://github.com/indrad3v4/Kernel-Olympics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## The $10B Problem

AMD GPUs (MI300X) outperform NVIDIA on price/performance. Yet enterprises stay on NVIDIA because **20% of CUDA code won't port to ROCm** — custom kernels, warp-sensitive logic, library-specific calls. hipify handles the easy 80%. The remaining 20% is a manual, weeks-long slog per project.

AMD's #1 adoption blocker isn't hardware — it's software migration friction.

## Kernel Olympics

**One kernel at a time.** We scan, classify, auto-port, compile, and verify CUDA→ROCm migration on real AMD hardware.

```
Submit CUDA kernel → Scanner (96.7% coverage) → Risk classifier (5 warp/wavefront patterns)
  → Pattern memory (trigram index, 0.2ms) → Porting agent (Fireworks API or template)
    → Verify on AMD GPU (real hipcc) → Report (Gemma-generated)
```

**Demo:** `python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu`

### What makes it different

| Feature | hipify | Kernel Olympics |
|---------|--------|----------------|
| Coverage | ~80% (syntax only) | **Pattern-aware** — catches warp/wavefront divergence |
| Verification | Manual | **Auto-compile + run + diff** on AMD GPU |
| Memory | Stateless | **Trigram cache** — 2nd kernel is **60,000× faster** (0.2ms vs 12s) |
| Report | None | **Gemma-generated** plain-English summary |

## 🚀 Try it

```bash
git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics
pip install -r requirements.txt
python3 src/main.py --input sample_kernels/cuda/warp_reduce.cu
python3 src/main.py --input sample_kernels/cuda/new_kernel.cu
```

## Why This Matters

- **AMD** gets enterprise adoption unblocked — the software gap closes
- **Enterprises** cut migration costs from weeks to minutes
- **ROCm ecosystem** grows faster when porting is frictionless

## Team — Meteorite 🌠

| Role | Member |
|------|--------|
| AI Architect | [indradev_](https://github.com/indrad3v4) |
| AMD/ROCm Engineering | Satoru |
| Kernel Engineering | Bromine185 |
| ML Pipeline | Aahil-Riyaz |
| CI/CD & Testing | meteorite67 |
| Infrastructure | icodemun44 |
| QA | _dD |
| Full Stack | icodemun44 |

**Built in 5 days for AMD Developer Hackathon ACT II — Track 3 (Unicorn).**

---

*"Like a meteorite, we don't arrive quietly."*
