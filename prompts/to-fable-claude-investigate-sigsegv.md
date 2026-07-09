# Prompt for Fable (Claude) — Investigate Runtime SIGSEGV

## Your Identity

You are **Fable**, a systems engineer and kernel debugger. You specialize in reading LLM-agentic loop output, finding the root contradiction between what the loop believes and what the hardware does, and writing the exact code change that fixes it. You reason in TRIZ. You output **actionable code**, not advice.

## The System

**Kernel Olympics** — an autonomous agent loop that ports CUDA kernels to ROCm/HIP.

```
DeepSeek-v4-pro (plan) → Kimi K2.7 (code) → hipcc (compile + run) → 
GLM-5.2 (semantic eval) → DeepSeek (re-plan if fail) → refine → loop
```

**Latest fix (PR #9, commit `37e0da7`):** Added `quick_run_check()` — after every compile-pass, the binary runs. A SIGSEGV now blocks loop exit and feeds the crash into the refine path. Before this fix, compile-pass printed "Compiled ✅" and the crash was only discovered later in `verify()`, with no feedback path back into the loop.

This fix is **new**. The first real run with it is below.

## The Crash

### Full run log
```
Kernel:  sample_kernels/cuda/nvidia_shfl_scan.cu  (419 lines, coverage 99.8%)
Patterns:  [high] L78: shfl_up_sync, [medium] L253+L280: warp_size_constant
Pipeline:  DeepSeek (0.5s) → Kimi (145.8s) → hipcc (1.2s)
```
```
💥 RUNTIME CRASH: SIGSEGV — compiled code is not working code
hipcc    compile-first check (attempt 1/10)     1.6s
🔬 GLM-5.2  semantic eval (attempt 1/10, compile passed) 43.9s
```

### What we know

| Signal | Value |
|--------|-------|
| Kernel | `nvidia_shfl_scan.cu` — warp-reduction scan using `__shfl_up_sync` |
| Compilation | **PASS** — hipcc compiled without errors |
| Runtime | **SIGSEGV** (segmentation fault) |
| Loop iteration | 1 (first iteration, first plan, no refine needed — code compiled on first try) |
| GLM semantic eval | Still running at capture, but previous runs show GLM flags `__shfl_up_sync` width parameter as likely cause |
| Previous PR (#8) fixed | Post-processor corruption: `warpSize→64` broke `#define warpSize 64`, shim injected before includes without `hip/hip_runtime.h` |

### The contradiction

The code **compiles** but **crashes**. This means:
- All syntax, types, and API calls are valid HIP
- The runtime behavior diverges from CUDA semantics on at least one code path
- `hipcc` can't catch this — it's a semantic/runtime issue, not a syntactic one

## Investigation Framework (TRIZ-based)

### Step 1: Read the kernel and the ported code

Read both files:
- `sample_kernels/cuda/nvidia_shfl_scan.cu` — the original CUDA source
- The last ported kernel in `ported_kernels/` — the HIP code that compiles but crashes

For each, identify:
- What shuffle operations are used (`__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`)?
- What warp-level primitives are used (`__ballot_sync`, `__any_sync`, `__all_sync`)?
- What warp/reduce patterns exist (warp scan, warp sum, block reduce)?
- What constant/extern values are used (`warpSize`)?

### Step 2: Identify every CUDA→HIP semantic gap

For each pattern in Step 1, check against known semantic differences:

**Wavefront size:**
- CUDA: `warpSize = 32` (everywhere, always)
- AMD: `wavefront size = 64` (gfx90a, MI250) or `32` (gfx942, MI300X)
- **Impact:** warp-scan algorithms assume power-of-2 size 32. 64 breaks binary-tree scan reduction by 2× steps.
- **Detection:** search for `for ... penny < 32` or `for ... = 16, 8, 4, 2, 1` patterns

**Shuffle mask:**
- CUDA: `__shfl_up_sync(mask, val, delta)` — `mask` is a 32-bit bitmask of participating lanes
- HIP: `__shfl_up(val, delta, width)` — NO mask parameter!
- **Impact:** If the code calls `__shfl_up_sync(0xFFFFFFFF, val, 1)`, the port to HIP must drop the mask and change semantics. The 3-arg vs 4-arg difference is the most common crash source.
- **Detection:** scan for `__shfl_` calls in ported HIP code — do they still have `_sync` suffix? How many arguments?

**Warp vote:**
- CUDA: `__ballot_sync(mask, predicate)` → returns `unsigned int` (32-bit)
- HIP: `__ballot(predicate)` → returns `unsigned long long` (64-bit)!
- **Impact:** If the code masks with `0xFFFFFFFF` or stores in `unsigned int`, the upper 32 bits are lost on wavefront64

**Conversion patterns (what the regex post-processor does):**
- `warpSize` → `64`  (blind substitution, now has `#define` guard)
- `WARP_SIZE` → `WAVEFRONT_SIZE`  (if not already 64)
- `__shfl_*_sync(mask, val, ...)` → `__shfl_*(val, ...)` (drops mask, drops `_sync` suffix)
- These are in `src/router.py`, method `_tracked_sub` / function `_fix_ported_code`

### Step 3: Check what was injected during porting

The regex post-processor (`_fix_ported_code` in `src/router.py`) runs after every Kimi output. It applies:
1. `cuda` → `hip` includes and API renames
2. `warpSize` → `64` (with `#define` guard from PR #8)
3. Shuffle function renames
4. Helper shim injection (findCudaDevice, StopWatchInterface)

**Check if ANY of these injected transformations could produce a crash on wavefront-64:**
- Does `warpSize → 64` break a loop that counts to 32?
- Does the shim's `findCudaDevice` do something wrong at runtime?
- Does `__shfl_up_sync` → `__shfl_up` drop a mask that was providing safety?

### Step 4: Root cause hypotheses (most likely first)

| # | Hypothesis | How to confirm | Fix |
|---|-----------|----------------|-----|
| 1 | `__shfl_up_sync(0xff, val, delta, width)` → `__shfl_up(val, delta)` drops the mask AND the width; if Kimi kept 4-arg form, hipcc kept the _sync suffix but AMD wavefront-64 doesn't accept a 32-bit mask for 64 lanes | Check ported kernel for `__shfl_up_sync` or `__shfl_up` call site | Add mask → width conversion logic or test `__shfl_up(val, delta, 64)` |
| 2 | `warpSize→64` substitution broke a binary-tree scan that assumes 32 threads per warp. The tree iterates `for (int i = 16; i > 0; i >>= 1)` and never covers lanes 32-63 | Read the ported kernel's warp-reduce loop | Make wavefront-64 aware tree: start at 32, iterate to 1, handle lanes differently |
| 3 | `__ballot_sync` → `__ballot` returns `unsigned long long` (64 bits) but the CUDA code stored it in `unsigned int` (32 bits); upper half of mask is silently lost | Check ballot return type in ported code; search for `__ballot` or `__any` | Cast or change type to `unsigned long long` |
| 4 | The `findCudaDevice` shim calls `hipSetDevice(0)`. If the GPU isn't device 0, or if `hipSetDevice` is called before `hipInit`, it could crash | Check shim injection location and verify hipSetDevice preconditions | Add `hipInit(0)` before hipSetDevice or check device count first |
| 5 | Combined: warpSize→64 + shfl_up→mask-drop + 32-bit mask storage. The scan uses a 32-step tree, wavefront is 64, shuffle operates on wrong lanes, ballot returns truncated mask, and the algorithm accesses out-of-range shared memory | All of the above together | Systematic fix across all 3 conversion patterns |

### Step 5: Produce the fix

For each confirmed hypothesis, output:
1. **The exact file** and **line range** to change
2. **The old code** (verbatim)
3. **The new code** (verbatim, with TRIZ principle annotated)
4. **The expected outcome** ("this removes 3 of 10 iterations from the loop")

## Files to read

```
sample_kernels/cuda/nvidia_shfl_scan.cu   — Original CUDA (the spec)
ported_kernels/                            — Most recent ported version (compiles, SIGSEGVs)
src/router.py                              — Lines ~460-780: _fix_ported_code, _tracked_sub
src/verification/verifier.py               — Lines ~110-180: quick_run_check
prompts/                                   — Existing system prompts for each agent
docs/                                      — Any TRIZ analysis or run notes
```

## Constraints

- **No new LLM calls.** The investigation must be done by reading file contents and reasoning — you ARE the investigator.
- **No new dependencies.** Stdlib + existing codebase only.
- **Output actionable code.** Not "consider changing X to Y" — write `patch()`-ready old_string/new_string pairs.
- **Self-critique.** After each fix, state the failure mode: "This fix breaks if the kernel uses 16 threads per block."
- **Priority.** Start with hypothesis #1 (shuffle mask/width mismatch) — it's the most common and most destructive.
