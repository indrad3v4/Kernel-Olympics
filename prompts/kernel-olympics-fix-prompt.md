# Mission: Fix the Kernel Olympics CUDA→HIP Pipeline

**Target repo:** `indrad3v4/Kernel-Olympics`
**Failing kernel:** `sample_kernels/cuda/nvidia_shfl_scan.cu`
**Current result:** `FAILED` (wall-clock timeout, 177.4s, 0 verified patterns)
**Required result:** `VERIFIED` (kernel compiles under hipcc), existing sample kernels still pass, logs accurately reflect execution.

---

## How to read this document

This is a three-part plan with a strict ordering rule:

> **Do NOT begin the architectural rewrite until Part A is fixed and proven.**

The failing kernel is not a model-quality problem and it is not (primarily) an orchestration problem. It is a **specific data contradiction between two existing components**. If you start with a grand state-machine redesign, it will pass review and this kernel will still fail, because none of the architectural items address the actual defect. Fix the bug, prove it compiles, *then* harden.

Part A = the surgical root-cause fix (makes the benchmark pass).
Part B = timeout/budget re-allocation (makes real GPU runs survive).
Part C = architecture hardening (the long-term robustness goals).

---

## PART A — The root-cause bug (fix first, verify, commit)

### A.1 What actually fails — the exact chain

`nvidia_shfl_scan.cu` is a **complete NVIDIA sample program**, not a bare kernel. Structure:

| Line | Symbol | Kind |
|------|--------|------|
| 55  | `__global__ void shfl_scan_test(int* data, int width, int* partial_sums)` | device kernel (portable) |
| 134 | `__global__ void uniform_add(int* data, int* partial_sums, int len)` | device kernel (portable) |
| 207 | `bool shuffle_simple_test(int argc, char** argv)` | host driver (uses `checkCudaErrors`, events, timers) |
| 310 | `bool shuffle_integral_image_test()` | host driver (needs `shfl_integral_image.cuh`) |
| 383 | `int main(int argc, char** argv)` | host driver (calls `findCudaDevice`, the two tests above) |

The program `#include`s `helper_cuda.h`, `helper_functions.h`, and `"shfl_integral_image.cuh"` — **none of which are vendored in this repo**. So the host driver and `main()` cannot be built here; only the two `__global__` kernels are portable.

Two components then make locally-correct but **mutually contradictory** decisions:

1. **`router.ModelRouter._unsatisfied_main_calls()` correctly refuses to restore `main()`.**
   Restoring it would call `shuffle_integral_image_test()` (missing header) and produce an unfixable link error. So the pipeline logs:
   ```
   ⚠ MAIN NOT RESTORED: driver needs code this port dropped
   ```
   This is the *right* call.

2. **`verification/verifier.py::_generate_harness()` reads `specs/nvidia_shfl_scan.json` → `"self_contained": true` and returns the ported code verbatim as the test file**, on the assumption that the code contains its own `main()`:
   ```python
   spec = self.load_spec(kernel_name)
   if spec is not None and spec.get("self_contained", False):
       return ported_kernel_source, 1, len(ported_kernel_source.splitlines())
   ```

**The contradiction:** Component 1 guarantees the code has *no* driver. Component 2 assumes the code *provides* its own driver and therefore does *not* synthesize one. Net effect: the translation unit handed to `hipcc` has **neither the original driver nor a synthesized harness**.

### A.2 The cascade that produces the observed log

3. The coder prompt (in `DEVICE_SUBSET`-style situations) currently still says *"Do NOT strip `main()` — it drives the full test."* So the model tries to reproduce the whole 419-line program, including the parts that cannot compile.
4. Under the token/time budget the model truncates → unbalanced braces →
   ```
   🧱 STRUCTURAL REJECT: REJECTED: unbalanced braces (depth +1)
   ```
5. Because iteration 1 is a **structural reject**, the pre-compile structural gate **skips hipcc every iteration**:
   ```
   🧱 STRUCTURAL GATE (iter 1): hipcc skipped — see structural feedback
   ```
   So `hipcc` never actually runs inside the loop, and the log line
   `refining with compile errors (iter 1→2)` **is false** — there are no compile errors, only structural feedback.
6. The loop burns the budget (plan 22s + codegen 90s + refine 43s ≈ 156s), hits
   `24s remain — no room to retry`, and times out.
7. The final `test_nvidia_shfl_scan.cpp:242:12: no matching function` and `:303:42: expected '}'` errors come from the **authoritative post-loop verify** compiling the mangled best-attempt — that is the first and only time hipcc actually ran, and it compiled a truncated file.

### A.3 The fix — a "portable subset" contract

When a self-contained program's `main()` has **unsatisfiable dependencies**, switch that kernel from *"port the whole program"* to *"port only the device code and verify with a synthesized harness."*

#### A.3.1 One decision, computed once, shared by everyone

Introduce a single portability decision derived from the **original CUDA source**, e.g. an enum `PortMode ∈ {WHOLE_PROGRAM, DEVICE_SUBSET}`:

```
DEVICE_SUBSET  when  _is_self_contained(original) is True
                     AND ( _unresolved_local_headers(original) is non-empty
                           OR _unsatisfied_main_calls(original_main, device_code, original) is non-empty )
WHOLE_PROGRAM  otherwise
```

- Compute this **once**, early in `route()`, before the coder prompt is built.
- **Persist it into the spec JSON** as a new field `"port_mode"`.
- **Replace** the verifier's `spec.get("self_contained")` branch with `spec.get("port_mode") == "WHOLE_PROGRAM"`.

> **Why the existing `self_contained` flag is the bug's origin:** it answers *"does the source have a `main()`?"* but the verifier uses it to answer a different question — *"should the harness expect the ported code to provide a `main()`?"*. Those are only the same question when the driver is portable. `port_mode` separates them.

#### A.3.2 Behaviour in `DEVICE_SUBSET` mode

- **Coder prompt** must instruct: port ONLY the `__global__` / `__device__` functions (`shfl_scan_test`, `uniform_add`) plus any `__device__`/`__constant__` helpers they reference. **Drop** `main()`, `shuffle_simple_test`, `shuffle_integral_image_test`, and anything needing the missing headers. Do **not** emit a host driver.
- **`_ensure_main_preserved`** must not attempt restoration. Make this decline **mode-driven and authoritative**, not an emergent side effect of dependency analysis.
- **`_generate_harness`** must take the `_harness_from_spec` path (synthesize a fresh `main()` from the spec's params/launch), NOT the return-as-is path.

#### A.3.3 Fix the spec so the synthesized harness is correct

`specs/nvidia_shfl_scan.json` currently has bugs that will make even a correct port fail:

- `"output_readback": { "element_type": "float", "format": "float_per_line" }` is **wrong** — this kernel operates on `int`. Change to `int` / `int_per_line`.
- Confirm `kernel_function`, `params`, `launch`, and dynamic shared memory match the DEVICE signature the harness will call:
  `shfl_scan_test(int* data, int width, int* partial_sums)` launched with dynamic shared memory (`extern __shared__ int sums[]`).
- Add `"port_mode": "DEVICE_SUBSET"` (or let the auto-gen decision write it).

#### A.3.4 The actual porting work (this is the one thing the LLM is for)

The kernel uses `__shfl_up_sync(mask, value, i, width)` and `warpSize`. On AMD **wavefront64** the lane arithmetic and the `width` argument differ from CUDA's 32. The coder must:

- Preserve the `width`-scoped shuffle semantics (do not silently treat width as 32).
- Size shared memory by the wavefront size, not by a hard-coded 32.
- Keep `warpSize` correct for the target (64 on most AMD GPUs).

Make the coder prompt say this explicitly — it is the only part of the job a regex/hipify pass cannot do.

### A.4 Prove Part A before touching anything else

Add tests (all must pass):

1. **Unit — portability decision:** for `nvidia_shfl_scan`, assert `port_mode == DEVICE_SUBSET`; assert `_ensure_main_preserved` does not restore; assert `_generate_harness` returns a synthesized harness whose `main` is present and whose reported kernel-line range excludes that `main`.
2. **Unit — spec correctness:** assert readback type is `int`, kernel signature matches, launch config present.
3. **Integration — real compile:** mock the LLM with a **correct hand-written DEVICE_SUBSET port** of the two kernels (see A.5) and assert the verifier compiles it clean with `hipcc`. If `hipcc` is unavailable in CI, gate that assertion behind a capability check but keep the path runnable locally on the GPU box.
4. **End-to-end on the GPU box:** run the real pipeline and confirm `RESULT: VERIFIED` (compiled), not FAILED.

### A.5 Deliverable fixtures to create

- Corrected `src/verification/specs/nvidia_shfl_scan.json`.
- A reference DEVICE_SUBSET HIP port of `shfl_scan_test` + `uniform_add` (wavefront64-correct) to serve as the integration-test fixture / golden.

---

## PART B — Timeout & budget re-allocation

The 180s budget is exhausted before a single real compile→repair cycle runs.

Observed timeline (from the trace):

| Stage | Cap | Used |
|-------|-----|------|
| Plan | 36s | 22.2s |
| Codegen | ~132s (remaining − 25s reserve) | 89.9s |
| Refine 1 | remaining − 25s | 43.4s |
| Left | — | ~24s → "no room to retry" |

Two problems:

1. **Initial generation is allowed to eat ~130s.** Cap the *initial* generation smaller and protect a real **repair reserve** (~120–140s of a larger or re-apportioned budget). Implement per-stage budgets where **no stage may borrow the repair reserve**:
   ```
   Planning ~20s · Generation ~70s · Validation ~10s · Compile ~15s · Repair ~120–140s · Verify ~20s · Reserve ~25s
   ```
   (Exact numbers depend on `MAX_PIPELINE_SECONDS`; keep the *ratios* and the hard reserve.)

2. **Adaptive stop on repeated identical failure.** If the same structural (or compile) error signature repeats twice, switch strategy instead of regenerating the whole file again. After Part A this stops occurring for `nvidia_shfl_scan`, but keep the guard so future kernels don't burn the clock repeating a doomed generation.

> Do **not** fix this by simply raising the global timeout. Re-allocate it.

---

## PART C — Architecture hardening (only after A + B pass)

These implement the long-term robustness goals. They sit *on top of* the Part A fix.

### C.1 Separate the three repair paths
Lexical repair (markdown/prose/placeholders/truncation), structural repair (braces/missing functions/includes/symbols), and compiler repair (hipcc diagnostics) must be independent. **Compiler-repair code can only run after hipcc actually executed.**

### C.2 Honest logging
Fix the log so it names the real feedback source: *"Refining from lexical feedback"* / *"…structural feedback"* / *"…compiler diagnostics"*. Never print "refining with compile errors" when hipcc was skipped. (This was the misleading line in the failing trace.)

### C.3 Explicit stage contracts
Each validator returns a typed result:
```
{ passed: bool, confidence: float, reason: str, repair_mode: str, diagnostics: [...] }
```
The orchestrator consumes structured objects — **never parses console strings**. `IterationState` already exists; extend it rather than inventing a parallel structure.

### C.4 Validate after every mutation
Every refine and every retry output routes through extraction → lexical → structural → symbol → syntax **before any compile**. `_postprocess_port` is already the choke point; add a test proving the refine and retry paths both go through it.

### C.5 Symbol validation over line counts
Diff original **device** symbols vs ported device symbols. **Critical:** in `DEVICE_SUBSET` mode the expected symbol set is the device functions only. The diff must **not** flag the intentionally-dropped host driver (`main`, `shuffle_*_test`) as "missing" — a naive diff would re-introduce the exact bug Part A fixes. Add a test for this.

### C.6 Cache only verified artifacts
Never cache quarantined or timed-out attempts. Confirm the quarantine path cannot poison pattern memory. Assign confidence scores; reject low-confidence memories.

### C.7 Syntax validation before hipcc
Insert a lightweight syntax-only check (braces, semicolons, malformed declarations/templates) between structural validation and hipcc, so invalid code is rejected before an expensive compile.

### C.8 Debug & Replay
Keep unique per-run session directories; never overwrite artifacts. Persist original CUDA, planner output, raw responses, extracted code, all validation reports, compiler diagnostics, patches, retry history, state transitions, and metrics. Support replay from persisted artifacts **without consuming new LLM calls**.

---

## Guardrails (do not violate)

- **Do NOT make `MAIN NOT RESTORED` a hard-fatal error.** For this kernel the decline is *correct*. The fix is to change **mode**, not to abort. (An architecture proposal that makes it fatal would make this kernel fail faster, not pass.)
- **Do NOT raise the global timeout as the primary fix.** Re-allocate the budget.
- **Do NOT let the symbol diff treat dropped host code as missing** in `DEVICE_SUBSET` mode.
- **Every change ships with a test.**
- **Do NOT regress the other sample kernels** (`conv2d`, `warp_reduce`, `softmax`, `transpose`, `histogram`, `nvidia_simple_atomics`, `new_kernel`).

---

## Definition of done

1. `nvidia_shfl_scan.cu` reaches a **VERIFIED / compiled** result on the GPU box.
2. All existing sample kernels still pass.
3. `port_mode` is written to the spec and read **identically** by router and verifier.
4. `DEVICE_SUBSET` symbol diff does not flag dropped host code as missing.
5. At least one full **compile → patch → recompile** cycle fits within the wall clock.
6. Logs match actual execution — no "refining with compile errors" when hipcc did not run.
7. Compiler-repair code provably cannot run before a real compile.

---

## Self-review checklist (answer each with evidence: a test or a real run)

- [ ] No validator can be bypassed; no invalid generation reaches hipcc.
- [ ] No compiler-repair code runs before a real compile.
- [ ] The portability decision is single-source and shared (router == verifier).
- [ ] `DEVICE_SUBSET` symbol diff excludes intentionally-dropped host code.
- [ ] Repair budget is protected; a full compile→patch→recompile cycle can run.
- [ ] Every mutation is revalidated through `_postprocess_port`.
- [ ] Logs name the real feedback source (lexical / structural / compiler).
- [ ] The corrected spec (`int` readback, correct signature, `port_mode`) is committed.
- [ ] A wavefront64-correct reference port exists as an integration-test fixture.

---

## Appendix — Files you will touch

| File | Change |
|------|--------|
| `src/router.py` | `PortMode` decision; mode-driven `_ensure_main_preserved`; coder/refine prompts for `DEVICE_SUBSET`; budget re-allocation; honest logging; symbol-diff scope |
| `src/verification/verifier.py` | `_generate_harness` reads `port_mode`, not `self_contained`; always synthesize harness in `DEVICE_SUBSET` |
| `src/verification/specs/nvidia_shfl_scan.json` | Fix `element_type`→`int`, `format`→`int_per_line`; add `port_mode`; verify signature/launch |
| `src/verification/symbols.py` | Scope symbol diff to device symbols in `DEVICE_SUBSET` |
| `tests/` | Unit + integration tests described in A.4 and C.4/C.5 |

*Note: the wavefront64 correctness of the reference port must be verified with the ROCm toolchain on the GPU box — that step is intentionally part of Part A's definition of done, not something that can be confirmed offline.*
