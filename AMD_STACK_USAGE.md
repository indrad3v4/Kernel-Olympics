# AMD Stack Usage

## Hardware

| Component | Model |
|-----------|-------|
| **GPU** | AMD Instinct MI300X (220 Compute Units) |
| **Memory** | 192 GB HBM3 |
| **Arch** | CDNA 3 |

## Software Stack

| Layer | Version | Usage |
|-------|---------|-------|
| **ROCm** | 7.2.0 | GPU compute platform |
| **HIP** | 7.2.0 | CUDA→HIP kernel porting target |
| **hipcc** | 7.2.0 | Compiler for ported kernels |
| **rocFFT** | — | Accelerated FFT (via ROCm) |
| **rocBLAS** | — | BLAS operations (via ROCm) |
| **Fireworks AI** | — | LLM inference for pipeline agents |
| **vLLM** | 0.16+ | Local model serving (Gemma 4 fallback) |

## AMD-Specific Features Used

1. **hipGetDeviceProperties()** — Runtime warpSize detection (handles both wave32 RDNA3 and wave64 CDNA3)
2. **hipDeviceSynchronize()** — Safe GPU synchronization in proof harnesses
3. **hipMalloc/hipMemcpy** — Standard AMD memory management
4. **__shfl_up()** — AMD intrinsic for warp-level shuffle (non-sync variant)
5. **HIP runtime headers** — `<hip/hip_runtime.h>` for all ported kernels
6. **ROCm 7.2 compiler** — Compilation with full optimization flags

## Pipeline Integration

```bash
# The entire pipeline targets AMD hardware:
make port      # LLM generates HIP code
make compile   # hipcc compiles for AMD GPU
make run       # Runs on AMD GPU (e.g., MI300X)
```

## Verification on AMD Hardware

Every ported kernel is verified with:
1. `hipcc` compilation (zero warnings)
2. Real GPU execution on AMD MI300X
3. Numerical diff against expected output
4. warpSize detection at runtime (works on RDNA3 + CDNA3)
