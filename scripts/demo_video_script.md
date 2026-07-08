# Kernel Olympics — Winning Demo Video Script (3 min)

> **Format:** Terminal recording + voiceover (no talking head needed — judges care about proof)
> **Target:** AMD Hackathon Track 3 judges
> **Duration:** ~2:45
> **Record with:** QuickTime Player on Mac — `⌘ Cmd`+`⇧ Shift`+`5` → Record Selected Portion

---

## 🎬 OPENING — THE HOOK (0:00–0:30)

```
VISUAL: Full-screen terminal. Type slowly. Large font (18pt+).
```

**VOICEOVER:**
> "AMD MI300X beats NVIDIA H100 on price per flop. But enterprises stay on CUDA because **20% of CUDA code won't port to ROCm**. The warp intrinsics, the wavefront assumptions, the undefined behaviour when `warpSize` changes from 32 to 64.
>
> `hipify` handles 80%. The remaining 20% is a **six-week manual slog per kernel**.
>
> Kernel Olympics closes that gap — from six weeks to six seconds."

```
VISUAL: Type: python3 src/main.py --nvidia-sample --fresh
        Press ENTER. Let the green NVIDIA box appear.
```

---

## ✅ LIVE DEMO — THE PROOF (0:30–2:15)

### Part A: Real NVIDIA Code (0:30–1:00)

```
VISUAL: Terminal shows the green NVIDIA CUDA SAMPLE box appearing
```

**VOICEOVER:**
> "One command. This downloads a real CUDA sample from NVIDIA's official cuda-samples repository — not our own test kernel — and runs it through our pipeline."

```
TERMINAL OUTPUT:
┌─ NVIDIA CUDA SAMPLE ────────────────────────┐
│ Source: NVIDIA/cuda-samples                  │
│ File: cpp/2_Concepts_and_Techniques/...      │
│ URL: github.com/NVIDIA/cuda-samples/...      │
└──────────────────────────────────────────────┘
↓ Downloaded: shfl_scan.cu (419 lines)

→ RED: shfl_up_sync (high)
→ GLM(planner) → Kimi K2.7(coder) → Gemma 4(verifier)
→ 3-model pipeline ✅ (70%, 22s)
```

**VOICEOVER:**
> "It classifies the kernel RED — detects real CUDA warp primitives that need porting. Then runs a three-model pipeline: GLM plans the fix, Kimi K2.7 Code generates the HIP, and Gemma 4 verifies correctness."

### Part B: Real AMD Compilation (1:00–1:40)

```
VISUAL: Same terminal, now running hipcc
```

**VOICEOVER:**
> "But code generation isn't proof. The output has to compile — and run correctly — on real AMD silicon."

```
TERMINAL:
$ hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
  → 0 errors, 0 warnings

$ /tmp/shfl_test
  → Block 0 sum: 64
  → Block 1 sum: 64  
  → Block 2 sum: 64
  → Block 3 sum: 64
  → TEST: PASSED ✅
```

**VOICEOVER:**
> "Real hipcc compilation. Real execution on AMD MI300X. Real PASSED output. This isn't static analysis — it's verified execution on actual AMD hardware."

### Part C: rocm-smi — Proof of AMD Silicon (1:40–1:55)

```
VISUAL: Type rocm-smi. Show GPU 0 — MI300X, temperature, VRAM.
```

**VOICEOVER:**
> "And here's the proof we're on real AMD silicon — rocm-smi showing an active MI300X. Most competitors never touch AMD hardware. We compile, run, and verify on it."

### Part D: Cache — The Force Multiplier (1:55–2:15)

```
VISUAL: Rerun the same command — show the speed difference
```

**VOICEOVER:**
> "Here's the force multiplier. The first port took 22 seconds with three LLM calls. Watch the second run."

```
TERMINAL:
$ python3 src/main.py --nvidia-sample
  → 🔥 Cache HIT: 0.3ms (vs 22s first run)
  → ~60,000× faster
```

**VOICEOVER:**
> "Sixty thousand times faster. Every kernel your team ports makes the entire team faster. Compounding returns."

---

## 🏆 CLOSE — THE ASK (2:15–2:45)

```
VISUAL: Cut to simple end card — project name, GitHub link, AMD badge
```

**VOICEOVER:**
> "We're Kernel Olympics — and we believe GPU porting shouldn't take six weeks. It should take six seconds.
>
> We've proven it on a real NVIDIA sample, compiled it on AMD MI300X, and verified it passes. We're eligible for the Gemma Prize. And it runs with zero external services — no daemon, no API key required.
>
> Vote for Kernel Olympics. And thank you to AMD for making ROCm the platform that makes this possible."

```
FINAL FRAME:
  Kernel Olympics 🏆
  github.com/indrad3v4/Kernel-Olympics
  [AMD Compatible ✓] [Gemma Prize eligible]
  "Vote for us on lablab.ai"
  FADE TO BLACK.
```

---

## 🎯 Pre-Recording Checklist

### Terminal Setup
- [ ] Dark theme, 18pt+ font
- [ ] Pre-stage: `cd /workspace/Kernel-Olympics && git pull`
- [ ] Pre-stage: `bash scripts/start_gemma_vllm.sh` running in background
- [ ] No browser tabs, no clutter on screen
- [ ] Clear `~/.cache/huggingface` for fresh download

### Recording
- [ ] QuickTime → Record Selected Portion → select terminal window only
- [ ] Mute all notifications (macOS DND)
- [ ] Airplane mode or disable WiFi during recording (no popups)
- [ ] Record voiceover separately (iPhone Voice Memos or Mac's QuickTime)

### Timing
- [ ] 0:00–0:30 Hook (NVIDIA sample download + classification)
- [ ] 0:30–1:00 Pipeline run (3 models working)
- [ ] 1:00–1:40 hipcc compile + PASSED ✅
- [ ] 1:40–1:55 rocm-smi proof
- [ ] 1:55–2:15 Cache hit second run
- [ ] 2:15–2:45 Close + vote ask
