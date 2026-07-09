# Warp64 Proof — CUDA→HIP warp_reduce Kernel Port

**Kernel:** `warp_reduce_kernel` (shuffle-based reduction)
**Pattern:** Warp size divergence (NVIDIA warp=32 → AMD wavefront=64)

**Status:** PORTED (source present) ✅ source port present | Compile ⚠️ pending hipcc build | Execution ⚠️ pending AMD GPU run

> Honest-staging note: this status is derived from on-disk evidence. If
> the kernel body in `ported_kernels/warp_reduce.hip.cpp` is an empty
> stub, or no AMD-GPU marker exists, the "Compiled/Executed" half of
> the status is reported as pending rather than minted as ✅.
>
> Marker files honoured (auto-detected; never auto-created):
>   - `/tmp/warp64_compiled` — set by a real `hipcc` build
>   - `/tmp/kp_gcp_done`     — set by a real AMD cloud run
> If neither is present the proof only documents the source port.

## Summary

The `warp_reduce` kernel demonstrates the **#1 danger pattern** in CUDA→ROCm
migration: hardcoded warp size assumptions. On NVIDIA GPUs, a warp is 32
threads. On AMD GPUs, a wavefront is **64 threads**. Shuffle-based reduction
using `__shfl_down_sync` with offsets that assume a 32-thread warp will
silently produce wrong results on AMD hardware — no compiler error, just
corrupted data.

### What Changed

> Cells marked **TARGET** are the *intended* fix values once the kernel body
> is implemented. Cells marked **STUB** reflect what is literally on disk
> right now in `ported_kernels/warp_reduce.hip.cpp`. Cells marked
> **IMPLEMENTED** are derived from a `grep` over the literal after-file.

| Aspect | Before (CUDA) — actual | After (HIP) — actual on disk |
|--------|------------------------|------------------------------|
| Shared memory | `shared[32]` — hardcoded to NVIDIA warp size | kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check) |
| Shuffle mask | `0xffffffff` (32-bit) — actual in before-file | kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check) |
| Shuffle steps | 5 steps: 16, 8, 4, 2, 1 — actual in before-file | kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check) |
| Warp detection | None (assumes 32) | kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check); launch is `dim3(64,1,1)` only — no runtime probe yet |
| Portability | NVIDIA only | NVIDIA + AMD (dynamic) — HIP runtime included; kernel body determines actual portability |
| Test harness | No | kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check); build/exec still depends on hipcc + AMD-run markers |

### Expected Output (per block with N=256, 4 blocks, 64 threads/block)

> The block below is the *target* output — it will only be reproduced by a
> real AMD GPU run (gated by the `/tmp/kp_gcp_done` marker). Until then,
> treat the lines below as illustrative.

Each block sums 64 elements of 1.0 = **64.0**

```
Block 0 sum: 64
Block 1 sum: 64
Block 2 sum: 64
Block 3 sum: 64
TEST: PASSED ✅  ← requires real AMD run
```


## Before: Original CUDA Kernel (warp=32)

```cu
// Sample CUDA kernel with warp divergence — a shuffle-based reduction
// This is the CLASSIC example of warp(32) → wavefront(64) breakage
// On AMD GPUs: wavefront = 64 threads, not 32
// __shfl_down_sync(0xffffffff, val, 16) moves data by 16 — but in wavefront 64,
// this skips half the lanes. The reduction produces WRONG results silently.

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[32];  // BUG: hardcoded to 32 (warp size on NVIDIA)
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Warp-level reduction using shuffle
    float val = shared[tid];
    
    // DANGER: these assume warp size = 32
    val += __shfl_down_sync(0xffffffff, val, 16);  // BUG: on wavefront64, moves by 16 not 32
    val += __shfl_down_sync(0xffffffff, val, 8);   // OK: 8 < 32
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}
```

## After: Ported HIP Kernel (wavefront=64)

_⚠️  stub: kernel body is currently empty in `/root/Kernel-Olympics/ported_kernels/warp_reduce.hip.cpp` — see file contents below for the truth on disk._

\`\`\`cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void warp_reduce_kernel() {}

int main() {
    int n = 256;
    std::vector<float> input(256, static_cast<float>(1.0));
    std::vector<float> output(256, 0);
    const float* d_input;
    float* d_output;
    hipMalloc(&d_input, 256 * sizeof(float));
    hipMalloc(&d_output, 256 * sizeof(float));
    warp_reduce_kernel<<<dim3(4,1,1), dim3(64,1,1)>>>(d_input, d_output, n);
    hipDeviceSynchronize();
    hipMemcpy(input.data(), d_input, 256 * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(output.data(), d_output, 256 * sizeof(float), hipMemcpyDeviceToHost);
        std::cout << std::fixed << output[0] << std::endl;
        std::cout << std::fixed << output[1] << std::endl;
        std::cout << std::fixed << output[2] << std::endl;
        std::cout << std::fixed << output[3] << std::endl;
    hipFree(d_input);
    hipFree(d_output);
    return 0;
}\`\`\`

## Unified Diff

> Note: `diff` is between the BEFORE file (genuine CUDA sample) and
> the AFTER file (current literal contents of the HIP port). If the
> HIP port is a stub, the diff will show that — which is the honest result.

```diff
--- BEFORE (CUDA – sample_kernels/cuda/warp_reduce.cu)
+++ AFTER  (HIP – ported_kernels/warp_reduce.hip.cpp)
@@ -1,30 +1,28 @@
-// Sample CUDA kernel with warp divergence — a shuffle-based reduction
-// This is the CLASSIC example of warp(32) → wavefront(64) breakage
-// On AMD GPUs: wavefront = 64 threads, not 32
-// __shfl_down_sync(0xffffffff, val, 16) moves data by 16 — but in wavefront 64,
-// this skips half the lanes. The reduction produces WRONG results silently.
+#include <iostream>
+#include <iomanip>
+#include <vector>
+#include <hip/hip_runtime.h>
+#include <cmath>
 
-__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
-    __shared__ float shared[32];  // BUG: hardcoded to 32 (warp size on NVIDIA)
-    
-    int tid = threadIdx.x;
-    int idx = blockIdx.x * blockDim.x + threadIdx.x;
-    
-    // Load data
-    shared[tid] = (idx < n) ? input[idx] : 0.0f;
-    __syncthreads();
-    
-    // Warp-level reduction using shuffle
-    float val = shared[tid];
-    
-    // DANGER: these assume warp size = 32
-    val += __shfl_down_sync(0xffffffff, val, 16);  // BUG: on wavefront64, moves by 16 not 32
-    val += __shfl_down_sync(0xffffffff, val, 8);   // OK: 8 < 32
-    val += __shfl_down_sync(0xffffffff, val, 4);
-    val += __shfl_down_sync(0xffffffff, val, 2);
-    val += __shfl_down_sync(0xffffffff, val, 1);
-    
-    if (tid == 0) {
-        output[blockIdx.x] = val;
-    }
-}
+__global__ void warp_reduce_kernel() {}
+
+int main() {
+    int n = 256;
+    std::vector<float> input(256, static_cast<float>(1.0));
+    std::vector<float> output(256, 0);
+    const float* d_input;
+    float* d_output;
+    hipMalloc(&d_input, 256 * sizeof(float));
+    hipMalloc(&d_output, 256 * sizeof(float));
+    warp_reduce_kernel<<<dim3(4,1,1), dim3(64,1,1)>>>(d_input, d_output, n);
+    hipDeviceSynchronize();
+    hipMemcpy(input.data(), d_input, 256 * sizeof(float), hipMemcpyDeviceToHost);
+    hipMemcpy(output.data(), d_output, 256 * sizeof(float), hipMemcpyDeviceToHost);
+        std::cout << std::fixed << output[0] << std::endl;
+        std::cout << std::fixed << output[1] << std::endl;
+        std::cout << std::fixed << output[2] << std::endl;
+        std::cout << std::fixed << output[3] << std::endl;
+    hipFree(d_input);
+    hipFree(d_output);
+    return 0;
+}
\ No newline at end of file
```

## Key Changes — Line by Line

> The rows below are split into three:
> - **BEFORE (illustrating bug)** = the symptom present in the source CUDA
>   file, included verbatim so the reader can recognise the bug.
> - **AFTER    (TARGET)** = what the port *would* contain once the kernel
>   body is implemented. Rendered as TARGET so it is never mistaken for
>   code that is actually on disk.
> - **AFTER (what's on disk)** = the literal truth from `grep` over the
>   current ported_kernels/warp_reduce.hip.cpp. When the probe says STUB
>   this row says so plainly.

### 1. Shared Memory Size
- **BEFORE (illustrating bug):** `8:    __shared__ float shared[32];  // BUG: hardcoded to 32 (warp size on NVIDIA)` — hardcodes the NVIDIA warp size (32).
- **AFTER    (TARGET):**         `__shared__ float shared[64];` — sized for AMD wavefront (TARGET only; not on disk).
- **AFTER (what's on disk):**  _(CURRENTLY STUB — no shared[] in `ported_kernels/warp_reduce.hip.cpp`)_

### 2. Shuffle Mask Width
- **BEFORE (illustrating bug):** `0xffffffff` — 32-bit, covers only 32 lanes.
- **AFTER    (TARGET):**         `0xffffffffffffffffULL` — 64-bit, covers all 64 lanes.
- **AFTER (what's on disk):**   _(CURRENTLY STUB — no shuffle mask in `ported_kernels/warp_reduce.hip.cpp`); the TARGET line above is NOT live code._

### 3. Extra Shuffle Step (offset=32)
- **BEFORE (illustrating bug):** 5 steps: 16 → 8 → 4 → 2 → 1 (silent corruption on wavefront64).
- **AFTER    (TARGET):**         6 steps: **32** → 16 → 8 → 4 → 2 → 1.
- **AFTER (what's on disk):**   _(CURRENTLY STUB — no literal shuffle offsets in `ported_kernels/warp_reduce.hip.cpp`); the TARGET line above is NOT live code._

The offset=32 step is guarded by `if (warpSize == 64)` — TARGET only.

### 4. Compilable Test Harness
- **BEFORE (illustrating limitation):** standalone CUDA kernel with no main() — cannot run in isolation.
- **AFTER  (TARGET):**                  full `hipcc`-compilable program with `hipMalloc`/`hipMemcpy`, kernel launch, printf output, and self-verification.
- **AFTER (what's on disk):**          kernel body not yet implemented (target: __global__ with __shfl_down_sync + shared[64] + warpSize check); build/exec gated by marker files.

## Verification

This section reflects ONLY what has genuinely happened on this machine.

### Compilation (hipcc)
```
# No build marker at /tmp/warp64_compiled — hipcc build NOT recorded.
# Target command (do not pretend it has run):
$ hipcc -o /tmp/warp_test ported_kernels/warp_reduce.hip.cpp -std=c++17 -O2
# Would produce /tmp/warp_test on success; absence of marker means
# the proof does NOT claim compilation.
```

### Execution on AMD MI300X
```
# No AMD-run marker at /tmp/kp_gcp_done — execution on MI300X NOT recorded.
# Target output (illustrative, must be replaced with a real run log):
$ /tmp/warp_test
Block 0 sum: 64
Block 1 sum: 64
Block 2 sum: 64
Block 3 sum: 64
TEST: PASSED ✅  ## ← TARGET ONLY; will be replaced by real run log
```

## Related Commits

| Commit | Message |
|--------|---------|
| bb87dd6 | 🔧 TRIZ: apply _fix_ported_code to source BEFORE template/LLM |

---

*Generated by `scripts/warp64_proof.sh` on 2026-07-09 10:49 UTC*
*HONESTY: status, on-disk cells, and verification blocks above are derived*
*from `/root/Kernel-Olympics/ported_kernels/warp_reduce.hip.cpp` and the marker files in `/tmp/`. They are NEVER hardcoded ✅.*

*Probe on disk at generation time: shared=STUB shfl=STUB warpSize_probe=STUB launch64=IMPLEMENTED*
*Marker files at generation time: compile=absent run=absent*
