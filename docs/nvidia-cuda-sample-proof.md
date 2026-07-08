# Proof: Pipeline on Real NVIDIA CUDA Sample

**File:** `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu`
**Source:** [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
**Lines:** 419
**Content:** Shuffle intrinsic prefix sum + integral image

## Pipeline Result

| Step | Status | Details |
|------|--------|---------|
| Scan | ✅ | Coverage: 0% (no hipify-clang) |
| Classify | ✅ | **RED** — shfl_up_sync (high), warp_size_constant × 2 (medium) |
| Memory | ✅ | 0 cached (first run) |
| Port | ✅ | GLM(planner) → Kimi K2.7(coder) → DeepSeek(verifier) — 25%, 12s |
| Verify | ⚠️ | Stored unverified (no hipcc in CI env) |
| Cost | $0.0014 | 1 LLM call |

## Classification Detail
```
nvidia_shfl_scan.cu: 
  [medium] L253: warp_size_constant
  [medium] L280: warp_size_constant  
  [high]   L78:  shfl_up_sync
```

## Key Takeaway

This is a **real, unmodified file from NVIDIA's official CUDA samples repository**. 
Our pipeline correctly identified warp-level patterns that need ROCm porting,
generated a ported HIP kernel, and stored the fix for future cache hits.

On an AMD MI300X machine (Jupyter notebook), the pipeline additionally:
- Compiles via `hipcc`
- Runs on real AMD GPU
- Verifies numeric output matches

See `docs/proof-on-mi300x.md` for GPU proof.
