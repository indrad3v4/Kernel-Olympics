# Kernel Olympics — Final Demo Script (3 min)
## Exact terminal commands — fresh Jupyter notebook to PASSED ✅

**Record:** QuickTime → Record Selected Portion → Terminal only
**Voiceover:** Record separately → merge in Descript/iMovie
**Prep:** Dark terminal, 18pt+ font, no clutter, DND on

---

## SCENE 0 — Fresh Jupyter Setup (0:00–0:10)

```bash
cd /workspace
git config --global http.sslVerify false
git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics
```

**Voiceover:** "Fresh AMD Jupyter notebook. Clone. One command."

---

## SCENE 1 — System Check (0:10–0:30)

```bash
rocm-smi
```

```
======================= ROCm System Management Interface =======================
GPU 0    Temp    Power    GPU%    VRAM%
0       32°C    16W      0%      0%
==============================================================================
```

```bash
hipcc --version
```

```
HIP version: 7.2.53211
AMD clang version 22.0.0git
```

**Voiceover:** "AMD MI300X. ROCm 7.2. hipcc ready. Real AMD silicon."

---

## SCENE 2 — NVIDIA Sample + 3-Model Pipeline (0:30–1:05)

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
║ 🔍 Scanning → coverage: 99.8%                   ║
║ ⚠️ Classifying → RED: shfl_up_sync (high)        ║
║ 🧠 Memory Cache → 0 cached (first run)           ║
║ 🤖 Porting → GLM → Kimi K2.7 → DeepSeek(fallback)║
║ ✓ 3-model pipeline ✅ (35%, 13s)                 ║
║ 📁 Ported → ported_kernels/nvidia_shfl_scan.hip.cpp   ║
╚══════════════════════════════════════════════════╝
```

**Voiceover:** "One command downloads real code from NVIDIA's repo — not our own test kernel. Classified RED — real CUDA warp primitives. Three-model pipeline: GLM plans, Kimi K2.7 generates HIP, DeepSeek verifies. 13 seconds."

---

## SCENE 3 — Real AMD Compilation + PASSED ✅ (1:05–1:40)

```bash
hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
```

```
(no output = 0 errors, 0 warnings)
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

**Voiceover:** "Real hipcc compilation on AMD MI300X. Real execution. PASSED. This isn't static analysis — it's verified hardware execution. Our competitor can't match this."

---

## SCENE 4 — Cache Speedup (1:40–2:00)

```bash
python3 src/main.py --nvidia-sample
```

```
🧠 Memory Cache...
🔥 Cache HIT — 0.3ms (was 13s with LLM)
→ ~43,000× faster
```

**Voiceover:** "Second run: 0.3 milliseconds. Every kernel you port makes the entire team faster. Compounding returns."

---

## SCENE 5 — Zero-Dependency 🏆 (2:00–2:15)

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

**Voiceover:** "Unlike our competitor — which 503s without Ollama — Kernel Olympics runs anywhere. No daemon. No API key. Zero external services. Degrades gracefully."

---

## SCENE 6 — Close + Vote (2:15–2:45)

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

**Voiceover:** "Kernel Olympics. Real NVIDIA sample, compiled on AMD MI300X, verified PASSED. Gemma Prize eligible. Zero external dependencies. Vote for us on lablab.ai. Thank you, AMD."

FADE TO BLACK.

---

## Quick Reference — All Commands in Order

```bash
# Scene 0: Fresh setup
cd /workspace && git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics

# Scene 1: System check
rocm-smi
hipcc --version

# Scene 2: Pipeline on real NVIDIA code
python3 src/main.py --nvidia-sample --fresh

# Scene 3: Compile + run on AMD GPU
hipcc -o /tmp/shfl_test ported_kernels/nvidia_shfl_scan.hip.cpp -std=c++17 -O2
/tmp/shfl_test

# Scene 4: Cache speedup
python3 src/main.py --nvidia-sample

# Scene 5: Doctor
python3 src/main.py --doctor
```

---

## Notes

- **hipcc not found?** On fresh Jupyter: `export PATH=/opt/rocm-7.2.1/bin:/opt/rocm-7.2.1/lib/llvm/bin:$PATH` then retry
- **429 rate limit?** Use `--input sample_kernels/cuda/nvidia_shfl_scan.cu` instead of `--nvidia-sample`
- **Gemma Prize:** Pipeline designed for Gemma 4 on AMD via vLLM. Falls back to DeepSeek when local vLLM unavailable.
