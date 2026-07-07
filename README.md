# Kernel Olympics — CUDA→ROCm Migration Copilot

**Track:** AMD Developer Hackathon ACT II — Track 3 (Unicorn)
**Team:** Meteorite 🌠

## One-Line Scope

We are NOT building a general CUDA→ROCm migration platform. We ARE building a narrow, verifiable proof-of-concept that: **scans a repo for portability risk, auto-fixes ONE class of dangerous bug (warp/wavefront divergence) with proof of correctness, and shows the system getting smarter across two runs.**

## MVP Demo

1. Take a small CUDA kernel file as input
2. Produce a risk-classified scan report: green/yellow/red
3. For one red-flagged kernel: auto-port it, fixing warp(32)→wavefront(64) divergence
4. Compile + run on AMD Developer Cloud, diff output byte-for-byte against CUDA reference
5. Show pattern memory counter updating as it processes
6. Process a second kernel with similar pattern → faster/higher confidence via cached fix
7. Output portability report with engineer-hours-saved estimate

## Architecture

```
CUDA kernel → Scanner (hipify dry-run) → Risk Classifier (rule-based) 
           → Pattern Memory (Chroma/FAISS) → Porting Agent (Fireworks) 
           → Verification (AMD Cloud Docker) → Report Generator (Gemma ROCm)
```

## Build Order

1. Scanner + risk classifier (no API dependency — fallback if everything else fails)
2. Verification loop on AMD Developer Cloud (highest technical risk — start early)
3. Porting agent (Fireworks) with confidence gating
4. Pattern memory + retrieval + "second kernel is faster" demo
5. Gemma-powered report generator
6. Demo script + pre-recorded fallback

## Project Structure

```
src/
├── scanner/            # hipify-clang dry-run wrapper
├── risk_classifier/    # regex/AST pattern matcher (warp/wavefront)
├── pattern_memory/     # Chroma vector store for verified fixes
├── porting_agent/      # Fireworks API LLM caller
├── verification/       # Docker + AMD Cloud compile/run/diff
├── report_generator/   # Gemma on local ROCm
└── main.py             # Orchestrator
sample_kernels/
├── cuda/               # Input CUDA kernels with warp divergence
├── hip/                # Expected/output HIP kernels
└── reference/          # Known-good CUDA outputs for diff
```
# CI/CD test trigger
