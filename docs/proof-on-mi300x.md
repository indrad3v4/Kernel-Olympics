# Proof: Pipeline Verifies a Ported Kernel on a Real AMD MI300X

**Status:** ⏳ **PENDING — verification attempted but no AMD GPU/ROCm stack available in current CI environment.**

> **Attempted verification:** `scripts/verify_on_amd_gpu.py` was run on
> `2026-07-11` in the standard Hermes CI environment (Linux x86_64 cloud VM).
> Both `rocm-smi` and `hipcc` reported "not found" — this environment does not
> have access to AMD ROCm hardware or toolchain. See
> `scripts/verify_on_amd_gpu.py` for the exact tooling required and instructions
> for running on **AMD Developer Cloud** (`notebooks.amd.com`) where an MI300X
> is available.

## Status summary

GPU verification is queued but **not yet recorded**. The current submission
environment (Hermes CI, AMD Developer Cloud, sandboxed runners used during
authoring) does **not** ship `hipcc`, so the ported HIP kernel cannot be
compiled and executed on real AMD silicon from this repo on its own.

Until that run is captured, every numeric claim in `docs/nvidia-cuda-sample-proof.md`
and `docs/proof-warp64.md` that mentions "runs on AMD GPU" / "compiled with
hipcc" / "executed on MI300X" refers to the *capability* of the pipeline and
the *expected* behavior of the ported HIP kernel — not to an in-repo record
of that run.

## How the proof is produced

The repo already contains the orchestration script that creates this file's
content. Run it from a Jupyter terminal on **AMD Developer Cloud**
(notebooks.amd.com), where `rocm-smi`, `hipcc`, and an MI300X device are
available:

```bash
export FIREWORKS_API_KEY=...   # any key; the run uses GLM/Kimi/DeepSeek
git clone https://github.com/indrad3v4/Kernel-Olympics.git
cd Kernel-Olympics
python3 scripts/verify_on_amd_gpu.py
```

What that script does, line by line:

1. `rocm-smi` — confirms an AMD GPU (MI300X / `gfx942`) is visible.
2. `hipcc --version` — confirms ROCm toolchain is installed.
3. Clones this repo into `/workspace/kernel-olympics`.
4. Runs the full pipeline (`src/main.py`) on `sample_kernels/cuda/warp_reduce.cu`,
   producing `/tmp/ko_report.json` (the classification → port → cache flow).
5. Writes a small self-contained HIP kernel to `/tmp/test_kernel.hip.cpp` —
   the same warp-reduce pattern that is the highest-risk CUDA→HIP migration
   case (warp=32 vs wavefront=64).
6. `hipcc -o /tmp/test_kernel /tmp/test_kernel.hip.cpp --offload-arch=gfx942`
   on the AMD host.
7. Runs `/tmp/test_kernel` on the device and asserts the per-block sum is
   `256.0 ± 0.001`.
8. Writes `AMD_GPU_PROOF.json` summarising: ROCm version, hipcc version,
   kernel name, compilation result, execution result, and pipeline report
   path. The script then `git add`s and commits it.

## What this file becomes after a successful run

Once `scripts/verify_on_amd_gpu.py` has been executed on AMD Developer Cloud,
this section is filled in from `AMD_GPU_PROOF.json`:

| Field           | Value (filled at run time) |
|-----------------|---------------------------|
| Verified on     | AMD GPU (e.g. MI300X)     |
| ROCm version    | e.g. 7.2                  |
| hipcc version   | e.g. 7.2                  |
| Kernel          | warp_reduce               |
| Compile         | passed / failed           |
| Execute         | passed / failed           |
| Pipeline report | `/tmp/ko_report.json`     |

…and the row in `docs/nvidia-cuda-sample-proof.md` flips from
`⏳ Pending` to `✅ Verified on MI300X`.

## Honest boundary

This document deliberately does **not** claim "we ran this on an MI300X and
it passed" because, at the time of writing, it has not been. The script,
the sample HIP kernel, the AMD Developer Cloud target, and the
`AMD_GPU_PROOF.json` schema are all real and present in the repo. The
recorded numeric output is the only part still missing, and that gap is
filled by the one command above.
