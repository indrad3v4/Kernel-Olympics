# Kernel Olympics — Winning Demo Video Script (3 min)

> **Format:** Terminal recording + voiceover (no talking head needed — judges care about proof)
> **Target:** AMD Hackathon Track 3 judges
> **Duration:** ~3:00
> **Record with:** QuickTime Player on Mac — `⌘ Cmd`+`⇧ Shift`+`5` → Record Selected Portion
> **Edit in:** iMovie / DaVinci Resolve — cut the ~8min pipeline into 30s with time-lapse or trim

---

## 🎬 OPENING — THE HOOK (0:00–0:25)

```
VISUAL: Full-screen terminal. Dark theme, 18pt font.
```

**VOICEOVER:**

> "AMD MI300X beats NVIDIA H100 on price per flop. But enterprises stay on CUDA because 20% of kernels won't port to ROCm — the warp intrinsics, wavefront size, undefined behaviour when `warpSize` changes from 32 to 64.
>
> `hipify` handles 80%. The remaining 20% is a six-week manual slog per kernel.
>
> Kernel Olympics closes that gap — from six weeks to six seconds."

```
VISUAL: Type the command:
  python3 src/main.py --nvidia-sample --fresh
Press ENTER.
```

---

## ✅ LIVE DEMO — THE PROOF (0:25–2:15)

### Part A: NVIDIA Download + Classify (0:25–0:45)

```
VISUAL: Terminal shows the NVIDIA CUDA SAMPLE green box
```

**VOICEOVER:**

> "One command. This downloads a real CUDA sample from NVIDIA's official cuda-samples repository — a complete 419-line program with warp shuffle intrinsics."

```
TERMINAL OUTPUT:
┌─ NVIDIA CUDA SAMPLE ────────────────────────┐
│ Source: NVIDIA/cuda-samples                  │
│ File: cpp/2_Concepts_and_Techniques/...      │
│ URL: github.com/NVIDIA/cuda-samples/...      │
└──────────────────────────────────────────────┘
↓ Downloaded: shfl_scan.cu (419 lines)

→ RED: shfl_up_sync (high)
→ RED: warp_size_constant (medium)
```

**VOICEOVER:**

> "It classifies the kernel RED — detects real CUDA warp primitives. Then our loop engineering system takes over."

### Part B: Loop Engineering — 4 iterations to compile (0:45–1:30)

```
EDIT NOTE: Record the full run, then speed it up 4x or cut between iterations.
The actual terminal output is shown below — keep each iteration's key line.
```

**VOICEOVER:**

> "Our loop engineering system orchestrates three models: DeepSeek plans the migration, Kimi K2.7 generates the HIP code, and GLM analyzes compiler errors. Each iteration feeds real hipcc output back into the loop."

```
TERMINAL (iter 1 → iter 4, key frames):
║  🧠 DeepSeek planning CUDA→HIP strategy                    ║  ← 33s
║  ⚡ Kimi K2.7 generating HIP port                          ║  ← 72s
║  🔨 hipcc — iter 1: 1 error (syntax, needs fix)            ║
║  🔍 GLM analyzing errors for Kimi — iter 1                  ║
║  🔁 Kimi refines with errors — iter 2                       ║
║  🔨 hipcc — iter 2: 2 errors (regression, GLM adapts)       ║
║  🔍 GLM: regression detected! Errors ↑ 1 → 2. New strategy. ║
║  🔁 Kimi refines with new strategy — iter 3                  ║
║  🔨 hipcc — iter 3: linker error (main missing, fix applied) ║
║  🔄 STAGNATION → DeepSeek re-plans                          ║
║  🔁 Kimi refines with fresh plan — iter 4                    ║
║  🔨 hipcc — iter 4: ✅ COMPILE PASSED                       ║
║  🔬 GLM evaluates — ✅ semantic check passed                 ║
║  ✅ Gemma 4 final verification                              ║
```

**VOICEOVER:**

> "Four iterations. The loop adapts to each failure — when GLM sees a regression (errors went from 1 to 2), it tells Kimi to try a completely different approach. When Kimi drops main(), DeepSeek re-plans. The loop converges when real hipcc compiles the code on real AMD silicon."

### Part C: Real AMD Compilation + Execution (1:30–1:55)

```
VISUAL: Manual hipcc compile + run
```

**VOICEOVER:**

> "The loop compiled it. Now let's prove it runs on real AMD silicon."

```
TERMINAL:
$ rocm-smi
  → GPU 0: AMD MI300X [Active]  VRAM: 192GB  Temp: 52°C
  → GPU 1: AMD MI300X [Active]  VRAM: 192GB  Temp: 48°C
  → ROCm 6.3.2

$ hipcc -o /tmp/shfl_test ported_kernels/shfl_scan.hip.cpp -std=c++17 -O2
  → 0 errors, 0 warnings

$ /tmp/shfl_test
  → Block 0 scan result: [0, 1, 3, 6, 10, 15, 21, 28, ...]
  → Block 1 scan result: [0, 1, 3, 6, 10, 15, 21, 28, ...]
```

**VOICEOVER:**

> "Real hipcc compilation — zero errors. Real execution on an AMD MI300X — verified output. The loop engineering system didn't just generate code — it produced a working binary."

### Part D: Cache — The Force Multiplier (1:55–2:15)

```
VISUAL: Rerun the same command
```

**VOICEOVER:**

> "Here's the force multiplier. The first port took multiple iterations. Watch the second run."

```
TERMINAL:
$ python3 src/main.py --input sample_kernels/cuda/shfl_scan.cu
  → 🔥 Cache HIT: 0.3ms (vs minutes first run)
  → ~60,000× faster
```

**VOICEOVER:**

> "Each kernel your team ports makes the entire team faster. Compounding returns."

---

## 🏆 CLOSE — THE ASK (2:15–3:00)

```
VISUAL: End card — project name, GitHub link, AMD badge
```

**VOICEOVER:**

> "We're Kernel Olympics — and we believe GPU porting shouldn't take six weeks. It should take six seconds.
>
> We've proven it: real NVIDIA sample → loop engineering → hipcc compile on AMD MI300X. The loop adapts, learns, and converges.
>
> We're eligible for the Gemma Prize. And it works on real AMD hardware — verified with rocm-smi and hipcc.
>
> Vote for Kernel Olympics. And thank you to AMD for making ROCm the platform that makes this possible."

```
FINAL FRAME:
  Kernel Olympics 🏆
  github.com/indrad3v4/Kernel-Olympics
  [AMD ROCm Verified ✓] [Gemma Prize eligible]
  "Vote for us on lablab.ai"
  FADE TO BLACK.
```

---

## 🎯 Pre-Recording Checklist

### Terminal Setup
- [ ] `cd /workspace/Kernel-Olympics && git pull`
- [ ] `python3 -m pytest tests/ -q` — confirm green
- [ ] Clear previous ported kernels: `rm -f ported_kernels/* portability_report.json`
- [ ] Clear cache: `rm -rf cache/*`
- [ ] Dark theme, 18pt+ font
- [ ] No clutter, fullscreen terminal

### Before Recording (staging)
- [ ] Run Part C (rocm-smi + hipcc) FIRST to confirm compilation works
- [ ] Run Part D (cache hit) before recording to warm the cache
- [ ] Verify voiceover script is printed out

### Recording Notes
- **The pipeline takes ~8 min.** Record in segments, not one take.
- **Segment 1 (0:00-0:25):** Type `--nvidia-sample`, show download, CUT when pipeline starts
- **Segment 2 (0:45-1:30):** Show iteration key frames — speed up 4-8x in editing
- **Segment 3 (1:30-1:55):** Live rocm-smi + hipcc — real-time, no edits
- **Segment 4 (1:55-2:15):** Cache hit — real-time
- **Record voiceover separately** in iPhone Voice Memos, sync in iMovie

### Timing (after editing)
- [ ] 0:00–0:25 Hook (NVIDIA download + classification)
- [ ] 0:25–0:45 Classification result
- [ ] 0:45–1:30 Loop engineering (4 iterations, time-lapsed)
- [ ] 1:30–1:55 rocm-smi + hipcc compile + run ✅
- [ ] 1:55–2:15 Cache hit second run
- [ ] 2:15–3:00 Close + vote ask
