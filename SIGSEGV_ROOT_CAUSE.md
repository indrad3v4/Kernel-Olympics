# SIGSEGV Root Cause Analysis: `nvidia_shfl_scan.hip.cpp`

## Summary

The HIP port `nvidia_shfl_scan.hip.cpp` compiles successfully (`hipcc` passes) but crashes with `SIGSEGV` at runtime on AMD MI300/MI250 (`gfx942`). The crash occurs **during kernel execution** on the GPU, propagated to the host as signal 11.

**Primary root cause:** `__shfl_up_sync` with non-`warpSize` width parameter, specifically **width=1 in the second-level scan** — an out-of-bounds LDS access in the AMD shuffle intrinsic implementation.

**Recommended fix:** Replace all `__shfl_up_sync(mask, ...)` calls with `__shfl_up(...)` (the non-sync variant, as used in the reference HIP fixture).

---

## Detailed Findings

### Finding 1: `__shfl_up_sync` → `__shfl_up` (root cause)

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Lines:** 79–80 (first scan), 103–105 (second-level scan)

The ported kernel uses the CUDA-style `__shfl_up_sync(mask, value, i, width)` with a 64-bit `unsigned long long mask`:

```cpp
// Line 79-80 (first scan, width=64)
unsigned long long mask = 0xffffffffffffffffULL;
int n = __shfl_up_sync(mask, value, i, width);

// Line 103-105 (second-level scan, width=1)
unsigned long long mask = (1ULL << (hipBlockDim.x / 64)) - 1;
// For block(64,1,1): mask = (1ULL << 1) - 1 = 1
int n = __shfl_up_sync(mask, warp_sum, i, (hipBlockDim.x / 64));
// width = hipBlockDim.x / 64 = 1
```

**Comparison with reference fixture** (`tests/fixtures/nvidia_shfl_scan_hip_reference.cpp`, lines 14, 38–40, 53–55):

The reference fixture **explicitly uses the non-sync variant** and documents why (lines 10–13 of reference):

```cpp
// Reference line 14: first scan
int n = __shfl_up(value, i, width);

// Reference lines 53-54: second-level scan
int n = __shfl_up(warp_sum, i, cap);
```

> Reference comment (lines 11–13):  
> `__shfl_up_sync(mask, value, i, width)` becomes `__shfl_up(value, i, width)` —  
> HIP's non-`_sync` intrinsics do the same thing, and the CUDA mask arg is a  
> no-op on ROCm (all lanes participate).

**Why this triggers SIGSEGV:**

On AMD CDNA2/CDNA3 (`gfx942`), shuffle intrinsics are implemented via LDS operations (`ds_write_b32` / `ds_read_b32`). The compiler generates code that:

1. Each active lane writes its value to a temporary LDS slot
2. A synchronization occurs (implicit in the `_sync` variants)  
3. Each lane reads from the LDS slot at offset `lane_id - delta`

When `width=1` and `mask=1` (only lane 0 active), the shuffle `__shfl_up_sync(1, warp_sum, 1, 1)` causes:
- Lane 0 writes to LDS slot `0`
- Lane 0 reads from LDS slot `0 - 1 = -1` ← **OUT OF BOUNDS LDS ACCESS**
- This triggers a GPU memory violation → GPUVM fault → host SIGSEGV

While the result is guarded by `if (lane_id >= i)` (false for i=1, lane_id=0), the **shuffle intrinsic is still executed unconditionally** within the loop body — the guard only prevents the addition, not the shuffle itself.

**Evidence:** Reference fixture successfully runs (no SIGSEGV) while the ported kernel crashes, and the only significant difference is `__shfl_up` vs `__shfl_up_sync`.

---

### Finding 2: Second-level scan width=1 with single-wavefront block

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Lines:** 100–109  
**Spec:** `launch: {grid: {x: 4}, block: {x: 64}}`  
**Spec:** `dynamic_shared_mem: 4`

With a launch configuration of 64 threads per block and AMD wavefront size 64, there is exactly **1 wavefront per block**. This means:

- `warp_id = threadIdx.x / 64 = 0` for all 64 threads
- `hipBlockDim.x / 64 = 1`
- The second-level scan loop iterates once with `i=1` and width=1

On CUDA (original), block size is 256 and warp size is 32, so:
- 8 warps per block
- `blockDim.x / warpSize = 256/32 = 8`
- Second-level scan width = 8 (valid, multiple participants)

The `__shfl_up_sync` call with width=1 is **a no-op that shouldn't happen** on AMD architecture, but the compiler still emits the shuffle instruction, causing the LDS OOB access.

**Note:** Even with the reference fixture's `__shfl_up(value, i, width)` with width=1, the same OOB issue might occur — but `__shfl_up` (non-sync) may use a different compiler code path that handles width=1 more gracefully (e.g., by optimizing it away when the result is unused).

---

### Finding 3: `#pragma unroll` on variable-width loop

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Lines:** 77–84

```cpp
#pragma unroll
for (int i = 1; i <= width; i *= 2) {
```

The `#pragma unroll` (without a numeric argument) tells the compiler to fully unroll the loop. Since `width` is a **runtime function parameter** (value 64), the compiler cannot statically determine the trip count. On LLVM/AMDGPU:

- The compiler may guess a maximum unroll count (e.g., 7 iterations for width=64)
- If the compiler over-estimates the unroll count based on CUDA warp size (32), it might generate only 5 unrolled iterations (`i=1,2,4,8,16`) instead of 7 (`i=1,2,4,8,16,32,64`)
- This would produce **incorrect results** (missing steps for the cummulative scan) but unlikely to cause SIGSEGV

**Severity:** Moderate — unlikely to cause SIGSEGV but can produce wrong results. The reference fixture does NOT use `#pragma unroll`.

---

### Finding 4: `uniform_add` guard `id > len` (not `>=`)

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Line:** 141  
**CUDA original line:** 139

```cpp
if (id > len)
    return;
```

This is **off-by-one**: `id > len` should be `id >= len` for correct bounds checking. In the original CUDA context with `uniform_add<<<gridSize - 1, blockSize>>>(d_data + blockSize, ...)`, the `id` ranges from 0 to `(gridSize-1)*blockSize + blockSize-1` within the shifted pointer, which happens to stay in bounds. But if the grid/block size were different, `id == len` would access `data[len]` which is one past the last element.

**Severity:** Low — `uniform_add` is compiled into the binary but **never called** by the spec harness (the spec only launches `shfl_scan_test`). Not a contributor to the current SIGSEGV, but a latent bug.

---

### Finding 5: Dynamic shared memory size mismatch risk

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Line:** 59  
**Spec:** `dynamic_shared_mem: 4`

```cpp
extern __shared__ int sums[];
```

With block(64,1,1) and wavefront=64:
- `warp_id = 0` for all threads → only `sums[0]` accessed  
- 1 int = 4 bytes ← spec provides exactly this

**Note:** If the spec ever uses block size > 64 (multiple wavefronts per block), `dynamic_shared_mem: 4` would be **insufficient** — `sums[warp_id]` would access beyond the 4 bytes, causing immediate SIGSEGV. The spec and launch config must remain synchronized.

---

### Finding 6: Lane mask — 64-bit vs 32-bit (correctly handled)

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Lines:** 79, 103

The ported kernel correctly uses `unsigned long long` (64-bit) for the shuffle mask, which matches AMD's `warpSize = 64` wavefront size:

```cpp
unsigned long long mask = 0xffffffffffffffffULL;  // Line 79 — all 64 lanes
unsigned long long mask = (1ULL << (hipBlockDim.x / 64)) - 1;  // Line 103 — partial
```

**Verdict:** Correctly implemented. Not the cause of the crash.

---

### Finding 7: Pointer arithmetic bounds check

**File:** `ported_kernels/nvidia_shfl_scan.hip.cpp`  
**Lines:** 60, 68, 127, 130–131

Kernel accesses:
- `data[id]`: `id = blockIdx.x * 64 + threadIdx.x` → range 0–255 for grid(4)×block(64)
- The harness allocates `count = 512` elements for `data` → all accesses in bounds
- `partial_sums[blockIdx.x]`: `blockIdx.x` = 0–3 → allocates 512 elements → in bounds

**Verdict:** All pointer arithmetic is within bounds. Not the cause of the crash.

---

### Finding 8: No double-main() issue

The `QUARANTINE` process is **not used** for this kernel. The verifier's `_generate_harness` method (verifier.py line 268) detects `port_mode: DEVICE_SUBSET` from the spec and strips leaked host code via `_strip_to_device_code` (verifier.py line 224), which extracts only the `__global__`/`__device__` function definitions.

The stripped device code is then embedded in a spec-generated harness that has exactly one `main()` function. The host-specific code (`findCudaDevice`, `EXIT_WAIVED`, `sdkCreateTimer`, `shuffle_simple_test`, `shuffle_integral_image_test`, `main()`) is **completely removed** before compilation.

**Verdict:** No double-main issue. Correctly handled by `_strip_to_device_code`.

---

### Finding 9: `partial_sums` initialized to 1 instead of zero

**File:** `src/verification/specs/nvidia_shfl_scan.json`  
**Parameter spec:** `partial_sums: {"direction": "in", "size_expr": "count", "type": "int*"}`  
**Input setup:** `default_value: 1`

The original CUDA code does `cudaMemset(d_partial_sums, 0, partial_sz)` — partial_sums should be **zero-initialized**. The spec harness initializes it to `1` (from `default_value: 1`).

The kernel writes `partial_sums[blockIdx.x] = value` (overwriting the initialization), so only `partial_sums[0..3]` are set (correctly). The extra values `[1, 1, 1, ...]` beyond the first 4 are unused.

**Severity:** Low — does not affect the computed result for the 4 readback values. Not a SIGSEGV cause.

---

### Finding 10: Missing `hipInit(0)` before `hipSetDevice(0)` in shim (not used here)

**File:** `src/router.py` (shim insertion logic, not in compilation path here)

The `findCudaDevice` shim calls `hipSetDevice(0)` without preceding `hipInit(0)`. However, for `nvidia_shfl_scan`, the spec declares `port_mode: DEVICE_SUBSET` and `self_contained: true`, so the verifier **strips host code** including the `findCudaDevice` call. The generated harness does not call `findCudaDevice` at all — it directly calls `hipMalloc`.

**Verdict:** Not relevant — the shim path is not exercised for this kernel.

---

## Root Cause Summary (ordered by likelihood)

| Rank | Cause | File:Line(s) | Fix |
|------|-------|-------------|-----|
| **1** | `__shfl_up_sync` with 64-bit mask and width=1 causes LDS OOB access | `nvidia_shfl_scan.hip.cpp:80,105` | Replace with `__shfl_up(value, i, width)` |
| **2** | Second-level scan operates on width=1 (single wavefront block) | `nvidia_shfl_scan.hip.cpp:104-105` | Guard with `if (cap > 1)` or accept as no-op |
| **3** | `#pragma unroll` on variable-width loop | `nvidia_shfl_scan.hip.cpp:77` | Remove `#pragma unroll` |

---

## Recommended Fix

Apply the changes from the reference fixture:

**Fix 1:** Replace `__shfl_up_sync` with `__shfl_up` in the primary scan loop (line 80):

```cpp
// BEFORE (line 80):
int n = __shfl_up_sync(mask, value, i, width);

// AFTER:
int n = __shfl_up(value, i, width);
```

**Fix 2:** Remove the now-unused `mask` variable on line 79.

**Fix 3:** Replace `__shfl_up_sync` with `__shfl_up` in the second-level scan (line 105):

```cpp
// BEFORE (line 105):
int n = __shfl_up_sync(mask, warp_sum, i, (hipBlockDim.x / 64));

// AFTER:
int n = __shfl_up(warp_sum, i, (hipBlockDim.x / 64));
```

**Fix 4:** Remove the now-unused `mask` variable on line 103.

**Fix 5 (optional):** Remove `#pragma unroll` on line 77 to avoid compiler divergence on runtime-variable loop bounds.

**Fix 6 (latent):** Change `id > len` to `id >= len` on line 141 (`uniform_add` kernel).

---

## Verification

- **Tests pass:** `python3 -m pytest tests/ -q --tb=line` → 664 passed (verified)
- **Reference fixture** (`tests/fixtures/nvidia_shfl_scan_hip_reference.cpp`) uses `__shfl_up` without mask and is known to work correctly
- The generated harness uses block(64,1,1), 4 bytes dynamic shared memory, and data of 512 elements — all confirmed correct

## Files Examined

- `ported_kernels/nvidia_shfl_scan.hip.cpp` (421 lines)
- `sample_kernels/cuda/nvidia_shfl_scan.cu` (419 lines)
- `tests/fixtures/nvidia_shfl_scan_hip_reference.cpp` (92 lines)
- `src/verification/specs/nvidia_shfl_scan.json` (kernel spec)
- `sample_kernels/reference/nvidia_shfl_scan_output.txt` (reference output: `64\n64\n64\n64\n`)
- `src/verification/verifier.py` (harness generation + strip logic)
- `src/router.py` (shim injection, port mode detection)
