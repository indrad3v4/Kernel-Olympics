# 📋 Full WBS — Kernel Olympics
## Dvizhok Producing Methodology

> **Project:** CUDA→ROCm Migration Copilot
> **Team:** Meteorite 🌠 (indradev_, Satoru, _dD)
> **Timeline:** Jul 6–11, 2026 (5 days)
> **Deadline:** Jul 11, 18:00 CEST
> **Budget:** $150 ($100 AMD Cloud + $50 Fireworks AI)

---

## 1. WBS by Results — Product Structure

```
KERNEL OLYMPICS MVP
│
├── R1: SCANNER MODULE
│   ├── R1.1 — hipify-clang dry-run wrapper
│   ├── R1.2 — Coverage report generator (green/yellow/red %)
│   └── R1.3 — File batch scanner
│
├── R2: RISK CLASSIFIER MODULE
│   ├── R2.1 — 5 warp/wavefront danger pattern detectors (regex/AST)
│   ├── R2.2 — Risk level assignment (green/yellow/red)
│   ├── R2.3 — Pattern match counter (live-updating)
│   └── R2.4 — Context extractor (surrounding code for each match)
│
├── R3: PATTERN MEMORY MODULE
│   ├── R3.1 — Code signature computation (structural features)
│   ├── R3.2 — Vector store (Chroma/FAISS in-memory)
│   ├── R3.3 — Similarity retrieval (query → nearest match)
│   └── R3.4 — Confidence tracking + persistence
│
├── R4: PORTING AGENT MODULE
│   ├── R4.1 — Fireworks API LLM caller (Kimi/GLM for code gen)
│   ├── R4.2 — Template-based fallback (no-API mode)
│   ├── R4.3 — Confidence gating (auto-apply vs flag for review)
│   └── R4.4 — Change annotation (what was modified and why)
│
├── R5: VERIFICATION MODULE
│   ├── R5.1 — HIP compilation harness (hipcc wrapper)
│   ├── R5.2 — Test input generator
│   ├── R5.3 — Output diff (byte-exact + floating-point tolerant)
│   └── R5.4 — AMD Developer Cloud Docker runner
│
├── R6: REPORT GENERATOR MODULE
│   ├── R6.1 — Gemma-powered narrative summary (local ROCm)
│   ├── R6.2 — Hours-saved estimator
│   └── R6.3 — JSON report exporter
│
├── R7: CLI ORCHESTRATOR
│   ├── R7.1 — Pipeline orchestrator (R1→R2→R3→R4→R5→R6)
│   ├── R7.2 — Pattern memory "second run is faster" demo
│   └── R7.3 — Live progress display
│
├── R8: INFRASTRUCTURE
│   ├── R8.1 — Dockerfile (ROCm base + Python deps)
│   ├── R8.2 — docker-compose.yml
│   ├── R8.3 — AMD Developer Cloud deployment script
│   └── R8.4 — CI/CD (GitHub Actions → Docker build → AMD deploy)
│
├── R9: SAMPLE KERNELS
│   ├── R9.1 — warp_reduce.cu (shuffle-based, classic warp→wavefront breakage)
│   ├── R9.2 — transpose.cu (shared memory tiling, second pattern)
│   ├── R9.3 — Reference outputs for both
│   └── R9.4 — Third kernel for "pattern memory speeds up" demo
│
├── R10: PITCH + DEMO
│   ├── R10.1 — README.md (pitch-ready, startup-focused)
│   ├── R10.2 — Demo script (live terminal recording)
│   ├── R10.3 — Pre-recorded fallback video (in case AMD Cloud flaky)
│   └── R10.4 — lablab.ai project page + submission
│
└── R11: GEMMA PRIZE SUBMISSION
    ├── R11.1 — Gemma local ROCm inference pipeline
    ├── R11.2 — Best AMD-Hosted Gemma Project documentation
    └── R11.3 — Shareable Gemma artifact (report card)
```

---

## 2. WBS by Process — Work Structure

```
PRE-PRODUCTION (Day 0-1)
├── P1.1 — Brief finalization + team alignment
├── P1.2 — AMD ADP sign-up + credit activation (Satoru)
├── P1.3 — Fireworks API key setup + test call (indradev_)
├── P1.4 — Repo setup + branch strategy + GitHub Actions
└── P1.5 — Sample CUDA kernels with warp divergence (prepare)

PRODUCTION (Day 1-4)
├── Day 1 — Deterministic core (zero API dependency)
│   ├── R1 Scanner — hipify wrapper + coverage report
│   ├── R2 Classifier — 5 pattern detectors + risk levels
│   └── R9 Sample kernels — 3 CUDA files + reference outputs
│
├── Day 2 — Verification + Pattern Memory
│   ├── R5 Verification — AMD Cloud Docker + compile/run/diff
│   ├── R3 Pattern Memory — signature + store + retrieve
│   └── R7.1 Orchestrator — pipeline skeleton
│
├── Day 3 — LLM Integration + Porting
│   ├── R4 Porting Agent — Fireworks API + template fallback
│   ├── R4.3 Confidence gating — auto vs review
│   └── R7.2 Second-run demo — pattern memory speedup
│
├── Day 4 — Polish + Gemma Prize
│   ├── R6 Report Generator — Gemma on local ROCm
│   ├── R8 Infrastructure — Docker + AMD deploy script
│   ├── R11 Gemma Prize — documentation + artifact
│   └── Full system integration test

POST-PRODUCTION (Day 5)
├── Day 5 — Pitch + Submit
│   ├── R10.1 README.md — pitch-ready startup story
│   ├── R10.2 Demo script — live terminal walkthrough
│   ├── R10.3 Pre-recorded video — fallback recording
│   └── R10.4 lablab.ai submit — all fields + Docker link
└── Risk buffer (20%)
```

---

## 3. Full WBS — Day-by-Day with Owners

### Day 1 (Jul 6) — Kick-off + Deterministic Core
**Milestone:** Scanner + Classifier working on sample kernels

| ID | Task | Owner | Depends | Time | Deliverable |
|:--:|------|-------|:-------:|:----:|------------|
| 1.1 | Idea lock + team alignment call | Team | — | 1h | Decision doc |
| 1.2 | AMD ADP sign-up + $100 credits | Satoru | — | 2h | Access confirmed |
| 1.3 | Fireworks API key + test endpoint | indradev_ | — | 1h | API working |
| 1.4 | GitHub repo init + branches + Actions | indradev_ | — | 1h | Repo live |
| 1.5 | Prepare 3 CUDA kernels with warp divergence | Satoru | — | 2h | sample_kernels/ |
| 1.6 | Scanner module — hipify dry-run wrapper | indradev_ | 1.4 | 3h | R1 ✅ |
| 1.7 | Risk classifier — 5 warp/wavefront patterns | indradev_ | 1.4 | 3h | R2 ✅ |
| 1.8 | Scanner + classifier integration test | Team | 1.6, 1.7 | 1h | Pipeline R1→R2 ✅ |
| 1.9 | Kick-off call at 18:00 CEST | Team | — | 1h | Intel from AMD |

**Day 1 risk:** If hipify-clang not available, implement pure regex scanner.

---

### Day 2 (Jul 7) — Verification + Pattern Memory
**Milestone:** AMD Cloud pipeline proven + pattern memory storing

| ID | Task | Owner | Depends | Time | Deliverable |
|:--:|------|-------|:-------:|:----:|------------|
| 2.1 | AMD Developer Cloud GPU instance setup | Satoru | 1.2 | 3h | GPU accessible |
| 2.2 | Verification harness — compile + run + diff | Satoru | 2.1 | 3h | R5 ✅ |
| 2.3 | Verification test on sample kernel | Satoru | 2.2 | 2h | Pass on real HW |
| 2.4 | Pattern memory — code signatures + vector store | indradev_ | 1.4 | 3h | R3 ✅ |
| 2.5 | CLI orchestrator skeleton (R1→R2→R3 flow) | indradev_ | 2.4 | 2h | R7.1 ✅ |
| 2.6 | Integration test: scan → classify → store | Team | 2.5 | 1h | End-to-end dry run |

**Day 2 risk:** AMD Cloud GPU access delayed → fallback: mock verification with pre-recorded outputs.

---

### Day 3 (Jul 8) — LLM Porting + Pattern Memory Speedup
**Milestone:** Porting agent working + second-kernel speedup demonstrated

| ID | Task | Owner | Depends | Time | Deliverable |
|:--:|------|-------|:-------:|:----:|------------|
| 3.1 | Fireworks API integration (Kimi + GLM) | indradev_ | 1.3 | 3h | R4.1 ✅ |
| 3.2 | Template-based porting fallback | indradev_ | 1.4 | 2h | R4.2 ✅ |
| 3.3 | Confidence gating + human review flag | indradev_ | 3.1 | 1h | R4.3 ✅ |
| 3.4 | Pattern memory retrieval in pipeline | indradev_ | 2.4 | 2h | R7.2 ✅ |
| 3.5 | "Second kernel is faster" demo scenario | Satoru | 2.3 | 2h | Demo script |
| 3.6 | Full pipeline integration (R1→R2→R3→R4→R5) | Team | 3.4 | 3h | Pipeline complete |

**Day 3 risk:** Fireworks API budget → $50 limit. Keep LLM calls minimal. Template fallback for bulk.

---

### Day 4 (Jul 9) — Polish + Gemma Prize
**Milestone:** Report generator + Docker + Gemma prize pipeline

| ID | Task | Owner | Depends | Time | Deliverable |
|:--:|------|-------|:-------:|:----:|------------|
| 4.1 | Report generator — Gemma on local ROCm | Satoru | 2.1 | 4h | R6.1 ✅ |
| 4.2 | Hours-saved estimator + report format | indradev_ | 3.6 | 1h | R6.2 ✅ |
| 4.3 | Dockerfile + docker-compose optimization | Satoru | 2.2 | 2h | R8.1, R8.2 ✅ |
| 4.4 | AMD Cloud deployment script | Satoru | 4.3 | 2h | R8.3 ✅ |
| 4.5 | Gemma Prize documentation + demo scenario | indradev_ | 4.1 | 2h | R11 ✅ |
| 4.6 | Full system test + bug fixes | Team | 4.4 | 3h | Stable build |

**Day 4 risk:** Gemma on ROCm doesn't compile → fallback: template report with Gemma API call.

---

### Day 5 (Jul 10 → Jul 11 18:00) — Pitch + Submit
**Milestone:** Submission ready + pitch recorded

| ID | Task | Owner | Depends | Time | Deliverable |
|:--:|------|-------|:-------:|:----:|------------|
| 5.1 | README.md — pitch-ready startup story | indradev_ | 4.6 | 3h | R10.1 ✅ |
| 5.2 | Live demo script + terminal recording | Team | 4.6 | 2h | R10.2 ✅ |
| 5.3 | Pre-recorded video (fallback, 2-3 min) | Team | 4.6 | 2h | R10.3 ✅ |
| 5.4 | GitHub tag + release | indradev_ | 5.1 | 1h | v1.0.0-hackathon |
| 5.5 | lablab.ai project page + submission | Team | 5.4 | 1h | R10.4 ✅ |
| 5.6 | Pitch deck (10 slides) | indradev_ | 5.1 | 3h | Deck |
| 5.7 | Discord announcement | Team | 5.5 | 0.5h | 🎉 |

**Day 5 door:** **Jul 11, 18:00 CEST** — hard deadline. Record video BEFORE deadline.

---

## 4. Gantt Timeline

```
TASK                          │ D1 Jul6 │ D2 Jul7 │ D3 Jul8 │ D4 Jul9 │ D5 Jul10 │ SUBMIT
──────────────────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼───────
Pre-production                │ ██████  │         │         │         │          │
  ADP sign-up                 │ ██      │         │         │         │          │
  API keys                    │ █       │         │         │         │          │
  Sample kernels              │ ███     │         │         │         │          │
──────────────────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼───────
Scanner + Classifier          │ ██████  │         │         │         │          │
Verification (AMD Cloud)      │         │ ██████  │         │         │          │
Pattern Memory                │         │ █████   │ ██      │         │          │
Porting Agent (Fireworks)     │         │         │ ██████  │         │          │
──────────────────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼───────
Report Generator (Gemma)      │         │         │         │ █████   │          │
Docker + Deploy               │         │         │         │ █████   │          │
Full Integration              │         │         │         │ █████   │          │
──────────────────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┼───────
README / Pitch                │         │         │         │         │ ██████   │
Demo Video                    │         │         │         │         │ █████    │
Submission                    │         │         │         │         │ ███      │ ██
Risk Buffer (20%)             │         │         │         │ ██      │ ██       │
```

---

## 5. Budget Estimate

| Item | Cost | Notes |
|------|:----:|-------|
| AMD Developer Cloud GPU | $100 | $100 ADP credits — GPU time for verification |
| Fireworks AI API | $50 | $50 ADP credits — LLM calls (Kimi, GLM, Gemma) |
| Estimated LLM calls | ~500 | At ~$0.10/call = $50, within budget |
| Estimated GPU hours | ~20h | At ~$5/h = $100, within budget |
| **Total** | **$150** | **Within hackathon budget** |

---

## 6. Risk Assessment

| Risk | Probability | Severity | Response |
|------|:-----------:|:--------:|----------|
| AMD Cloud GPU access delayed | 30% | 5/5 | **Minimize:** Start Day 1, not Day 3. Fallback: pre-recorded verification output. |
| Fireworks API budget exhausted | 20% | 3/5 | **Minimize:** Template fallback for bulk porting. LLM only for hard cases. |
| Gemma on ROCm doesn't compile | 40% | 4/5 | **Minimize:** Use Gemma API via Fireworks instead. Document local inference as stretch goal. |
| hipify-clang not available | 20% | 2/5 | **Minimize:** Pure regex scanner as fallback. Same detection rules, no hipify dep. |
| Team member unavailable | 15% | 3/5 | **Minimize:** Cross-train tasks. Every module has secondary owner. |
| Demo breaks live | 40% | 4/5 | **Minimize:** Pre-recorded fallback video. Demo script practiced ×3. |

---

## 7. Scoped Promise

> **✅ We DO ship (by Jul 11):**
> 1. Scanner + risk classifier on any CUDA kernel file
> 2. 5 warp/wavefront divergence patterns detected with context
> 3. Auto-porting of ONE red-flagged kernel with confidence score
> 4. Verification report (compile + run — actual AMD GPU or pre-recorded)
> 5. Pattern memory storing + retrieval + "second kernel is faster" demo
> 6. Plain-English portability report with hours-saved estimate
> 7. Pitch-ready README + demo video + Docker container

> **❌ We do NOT ship:**
> 1. Full-repo scanning at scale
> 2. GitHub Action / CI integration
> 3. Multi-language support (CUDA→HIP only)
> 4. All 20+ danger pattern classes (sticking to 5)
> 5. Polished multi-page web UI
> 6. User accounts / SaaS infrastructure

---

## 8. Subagent Prompts (for cron automation)

### Subagent 1: Core Pipeline
```
## Role
Python backend engineer — CUDA, GPU programming, ML pipelines

## Context
Building Kernel Olympics: a CUDA→ROCm migration copilot. Scanner + risk classifier + pattern memory are already scaffolded. Need to harden the verification module on real AMD GPU hardware.

## Goal
Get the verification loop (compile hipcc → run on AMD GPU → diff output) working end-to-end on AMD Developer Cloud.

## Steps
1. Set up AMD Developer Cloud instance with ROCm stack → GPU access is the bottleneck, start now
2. Install the repo, compile sample kernel with hipcc → prove compilation pipeline works
3. Run test harness, capture output → prove execution works
4. Diff against CUDA reference output → prove correctness check works
5. Add to orchestrator as VerificationAgent module

## Constraints
- DO NOT mock/fake GPU execution. If hardware is unavailable, fail gracefully with clear error.
- Use Docker for reproducibility.
- Keep compile flags minimal (-std=c++17 -O2 --offload-arch=gfx942).
```

### Subagent 2: Pitch + README
```
## Role
Technical writer + startup pitch specialist

## Context
Kernel Olympics — CUDA→ROCm Migration Copilot for AMD Hackathon ACT II. Track 3: Unicorn. Technical pipeline is being built; need pitch-ready README.

## Goal
Write a README.md that: (1) explains the problem in one sentence, (2) shows the demo in 30 seconds, (3) makes judges want to invest.

## Steps
1. Open with the one-line problem: "AMD's biggest adoption blocker is CUDA→ROCm migration"
2. Show the demo command — `python main.py --input my_kernel.cu` → scan, port, verify
3. Explain the architecture with a clean ASCII diagram
4. List DO/DON'T scope (builds trust)
5. Add team section + links

## Constraints
- Tone: startup pitch, not academic paper
- Judges are AMD engineers — respect their knowledge, don't oversimplify
- Track 3 criteria: creativity, originality, completeness, AMD use, market potential
```

---

*Meteorite 🌠 — "Like a meteorite, we don't arrive quietly."*
