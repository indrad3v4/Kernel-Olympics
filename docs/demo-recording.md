# Demo Recording — SDLC Loop

Terminal recording script for the **Kernel Olympics AMD Track 3** submission video (~3–5 min).

Shows the **multi-agent loop architecture**: **CUDA kernel → SCAN → PLAN → PORT → EVAL → VERIFY → PASS on real AMD GPU**.

> **Integrity note:** The demo runs REAL commands on REAL AMD hardware.
> Nothing is simulated, nothing is faked. The VERIFY stage's device-only proof
> retry is implemented in `src/verification/verifier.py` — the fallback happens
> automatically inside the orchestration loop, not via an external agent.

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
| 0 | Hardware + architecture intro | 5s |
| 1 | Original CUDA kernel (`__shfl_up_sync`) | 30s |
| 2 | Pipeline runs: SCAN→PLAN→PORT→EVAL (real 4-LLM orchestration) | 60s |
| 3 | VERIFY stage: full harness compile fails, device-only retry fires | 35s |
| 4 | Show the verifier code path that handles device-only proof | 30s |
| 5 | **hipcc compile** device-only proof → SUCCESS | 15s |
| 6 | **Run on real AMD GPU → PASS** 🚀 | 15s |
| 7 | Pipeline summary — all stages completed | 15s |

---

## Architecture

```
INPUT CUDA → SCAN → PLAN → PORT → EVAL → VERIFY → REPORT
               │       │       │       │       │        │
           hipify  DeepSeek  GLM-5.2  Kimi    hipcc   Gemma
                     v4                K2.7    + AMD GPU
```

When VERIFY's full-harness compile fails (unportable SDK host code), it
automatically retries with a **device-only proof** inside the loop:

1. `_strip_to_device_code()` — extract device functions
2. `_fix_hip_intrinsics()` — convert `__shfl_up_sync` → `__shfl_up`
3. `_try_device_only_proof()` — generate minimal harness, re-compile
4. If compile succeeds → **run on real AMD GPU** → correctness check

No cron jobs. No external watchers. No human in the middle.

---

## Converting to Video

```bash
# If agg is installed:
agg /tmp/kernel-olympics-demo.cast /tmp/demo.gif --idle-time-limit 2

# Upload to asciinema.org for embed:
asciinema upload /tmp/kernel-olympics-demo.cast
```

---

## Files

| Path | Purpose |
|------|---------|
| `scripts/demo_recording.sh` | Main recording/run script |
| `src/verification/verifier.py` | Device-only proof retry in VERIFY stage |
| `sample_kernels/cuda/nvidia_shfl_scan.cu` | Original CUDA kernel |
