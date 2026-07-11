# Demo Recording — SDLC Loop

Terminal recording script for the **Kernel Olympics AMD Track 3** submission video (~3–5 min).

Shows the full pipeline: **CUDA kernel → pipeline flags → firecrawl agent → auto-compile with hipcc → proof on real AMD GPU**.

---

## Quick Start (on notebooks.amd.com Jupyter terminal)

```bash
cd /workspace/Kernel-Olympics && git pull origin main
```

### Run the demo live

```bash
bash scripts/demo_recording.sh run
```

Paced with ~3–5 sec pauses between phases — takes about 3 minutes total.

### Record for video

```bash
bash scripts/demo_recording.sh record
```

Auto-detects recording tool:
- **asciinema** → saves `.cast` to `/tmp/kernel-olympics-demo.cast`
- **script(1)** → saves timing + session files
- Falls back to live run if no recorder found

---

## Demo Phases

| Phase | Content | Est. time |
|-------|---------|-----------|
| 0 | Hardware + repo info | 5s |
| 1 | Original CUDA kernel (`__shfl_up_sync`) | 30s |
| 2 | Pipeline runs → detects needs manual compile | 30s |
| 3 | Flagged kernel with `saved for manual hipcc` marker | 15s |
| 4 | Firecrawl agent detects the flag | 10s |
| 5 | Auto-generated proof harness source | 45s |
| 6 | **hipcc compile** for gfx1100 (RDNA3) | 15s |
| 7 | **Run on real AMD GPU → PASS** 🚀 | 15s |
| 8 | Summary table — full SDLC loop | 15s |

---

## Converting to Video

If you want a GIF or MP4 from the asciicast:

```bash
# If agg is installed:
agg /tmp/kernel-olympics-demo.cast /tmp/demo.gif --idle-time-limit 2

# Or upload to asciinema.org for an embeddable player:
asciinema upload /tmp/kernel-olympics-demo.cast
```

---

## What's Being Proved

The demo verifies that **NVIDIA's `__shfl_up_sync` warp-level prefix scan** compiles and runs correctly on **AMD RDNA3** hardware (gfx1100, wavefront 32) via:

1. **Pipeline** auto-generates a HIP version from the CUDA source
2. Detects host-code SDK symbols that can't be auto-port
3. Flags the kernel with `saved for manual hipcc`
4. **Firecrawl agent** detects the flag, generates a device-only proof harness
5. Compiles with `hipcc` targeting gfx1100
6. Launches `<<<1, 256>>>`, verifies every lane's prefix sum against expected values
7. **PASS** confirms the port is functionally correct

---

## Files

| Path | Purpose |
|------|---------|
| `scripts/demo_recording.sh` | Main recording/run script |
| `scripts/firecrawl_auto_compile.sh` | Firecrawl auto-compile agent |
| `ported_kernels/manual_hip_direct.hip.cpp` | Working proof (reference) |
| `sample_kernels/cuda/nvidia_shfl_scan.cu` | Original CUDA kernel |
