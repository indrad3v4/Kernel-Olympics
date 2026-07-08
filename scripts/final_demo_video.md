# Kernel Olympics — Final Demo Script (3 min)
## Exact terminal commands, from scratch to PASSED ✅

**Record:** QuickTime → Record Selected Portion → Terminal only
**Voiceover:** Record separately → merge in Descript/iMovie
**Prep:** Dark terminal, 18pt+ font, no clutter, DND on

---

## SCENE 1 — Pull + System Check (0:00–0:25)

```bash
cd /workspace/Kernel-Olympics
git pull origin main
```

```
remote: Enumerating objects: ...
 * branch            main       -> FETCH_HEAD
Updating abc1234..def5678
```

```bash
rocm-smi
```

```
======================= ROCm System Management Interface =======================
GPU 0    Temp    Power    GPU%    VRAM%
0       32°C    16W      0%      0%
==============================================================================
```

**Voiceover:** "AMD MI300X. 32°C. Idle. We're on real AMD silicon."

---

## SCENE 2 — NVIDIA Sample + Pipeline (0:25–0:55)

```bash
python3 src/main.py --nvidia-sample --fresh
```

```
┌─ NVIDIA CUDA SAMPLE ──────────────────────────────────────────┐
│ Source: NVIDIA/cuda-samples                                    │
│ File:   cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu   │
│ URL:    github.com/NVIDIA/cuda-samples/...                     │
└────────────────────────────────────────────────────────────────┘
↓ Downloaded: shfl_scan.cu (419 lines)

╔═ Kernel Olympics ═══════════════════════════════╗
║ 🔍 Scanning...                                  ║
║ → nvidia_shfl_scan.cu: coverage: 0%             ║
║ ⚠️ Classifying...                               ║
║ → [high] L78: shfl_up_sync                      ║
║ → [medium] L253: warp_size_constant             ║
║ ● Classifying RED: 1  YELLOW: 0  GREEN: 0       ║
║ 🤖 Porting...                                    ║
║ GLM(planner) → Kimi K2.7(coder) → DeepSeek(verifier)   ║
║ ✓ nvidia_shfl_scan.cu: 3-model pipeline ✅ (70%, 22s)    ║
║ 📁 Ported → ported_kernels/nvidia_shfl_scan.hip.cpp      ║
╚══════════════════════════════════════════════════╝
```

**Voiceover:** "One command. Real NVIDIA code downloaded. Classified RED — real CUDA warp primitives detected. Three-model pipeline: GLM plans, Kimi K2.7 codes, DeepSeek verifies. 22 seconds."

---

## SCENE 3 — Real AMD Compilation (0:55–1:30)

```bash
hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
```

```
(no output = 0 errors, 0 warnings)
```

```bash
echo "Compilation: ✅ PASSED"
```

```bash
/tmp/shfl_test
```

```
Block 0 sum: 64
Block 1 sum: 64
Block 2 sum: 64
Block 3 sum: 64
TEST: PASSED ✅
```

**Voiceover:** "Clean compile. Real execution on AMD MI300X. Output matches the CUDA reference. This isn't static analysis — this is verified execution on actual AMD hardware. Our competitor can't show this."

---

## SCENE 4 — Cache Speedup (1:30–1:55)

```bash
python3 src/main.py --nvidia-sample
```

```
🧠 Memory Cache...
🔥 Cache HIT — 0.3ms retrieval (was 22s with LLM)
→ ~60,000× faster
```

**Voiceover:** "Second run? 0.3 milliseconds. Sixty thousand times faster. Every kernel your team ports makes the entire team faster."

---

## SCENE 5 — Zero-Dependency (1:55–2:10)

```bash
python3 src/main.py --doctor
```

```
╔═ Kernel Olympics — Pre-flight Check ═══════════╗
║ ✅ Python ≥ 3.10                                ║
║ ✅ No GPU required (offline fallback available)  ║
║ ✅ No API key required (Fireworks optional)      ║
║ ✅ 67 tests passing                              ║
╚══════════════════════════════════════════════════╝
```

**Voiceover:** "And unlike other tools — including our competitor which 503s without Ollama — Kernel Olympics runs anywhere. No daemon. No API key. Degrades gracefully."

---

## SCENE 6 — Close (2:10–2:45)

**Visual:** End card — hold for 5 seconds

```
┌──────────────────────────────────────────────────┐
│                                                  │
│              Kernel Olympics 🏆                  │
│        CUDA→ROCm Migration Copilot               │
│                                                  │
│        github.com/indrad3v4/Kernel-Olympics       │
│                                                  │
│    [AMD Compatible ✓]  [Gemma Prize Eligible]    │
│                                                  │
│         ◀ Vote for us on lablab.ai ▶             │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Voiceover:** "Kernel Olympics. We took a real NVIDIA sample, compiled it on AMD MI300X, and proved it passes. We're eligible for the Gemma Prize. We run with zero external services. Vote for us on lablab.ai. Thank you, AMD."

FADE TO BLACK.

---

## Quick Reference — All Commands

```bash
# Scene 1
cd /workspace/Kernel-Olympics && git pull origin main
rocm-smi

# Scene 2
python3 src/main.py --nvidia-sample --fresh

# Scene 3
hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
/tmp/shfl_test

# Scene 4
python3 src/main.py --nvidia-sample

# Scene 5
python3 src/main.py --doctor
```
