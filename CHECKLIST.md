# CHECKLIST — de-roast program (Fable report)

Companion to [`ROAST.md`](ROAST.md). Items 1–5 are **implemented in this change set** (deterministic only — no new LLM calls, no new dependencies, per constraints). Items 6+ are specified with exact pointers for the next pass. 10B impact scale per the tasking doc: 1X = shfl-class kernels, 10X = all warp/wavefront kernels, 100X = all CUDA kernels, 1000X+ = general pipeline.

## Implemented (verify on the notebook before anything else)

### ✅ 1. `import threading` — the crash
- **Change:** `src/main.py` imports block.
- **Route:** none (CLI harness).
- **TRIZ:** #9 preliminary anti-action — plus item 5, the tripwire so this class never costs a GPU run again.
- **Convergence:** restores the entire verification phase; every full run currently crashes without it.
- **10B impact:** gate for all of it — 0X by itself, but everything is 0X while runs crash at verify.

### ✅ 2. Shim self-sufficiency — the col-56 killer
- **Change:** `src/router.py` `_fix_ported_code`: shim now carries `#include <hip/hip_runtime.h>` itself; `#pragma once` removed (meaningless in a main file).
- **Route:** fixer→hipcc (the non-LLM edge everyone forgot is a route).
- **TRIZ:** #24 intermediary made self-sufficient; the instrument must not depend on the product it modifies.
- **Convergence:** removes the injected `use of undeclared identifier 'hipSetDevice'` that consumed all 6 iterations. Regression test reproduces the failure before the fix.
- **10B impact:** 100X — every self-contained CUDA sample passes through this shim.

### ✅ 3. `#define warpSize` corruption guard
- **Change:** `src/router.py`: `(?<!#define )\bwarpSize\b` → `64`.
- **Route:** fixer→hipcc.
- **TRIZ:** #11 beforehand cushioning.
- **Convergence:** removes the injected `macro name must be an identifier` (29:9). Test-proven (`#define 64 64` reproduced, then fixed).
- **10B impact:** 10X — `warpSize` appears in essentially every warp-aware kernel; Kimi emitting the `#define` form is common.

### ✅ 4. Source lines at error locations — the hidden resource
- **Change:** `src/verification/verifier.py` `quick_compile_check` returns `error_context` (3 source lines per error, extracted from the exact compiled text, capped 6×160 chars); `src/router.py` appends it to Kimi's `evaluator_feedback` and passes it into `_build_glm_error_analysis_prompt`, where it explicitly counts as "found in the code" for GLM's honesty rule.
- **Route:** hipcc→Kimi and hipcc→GLM (new deterministic channel).
- **TRIZ:** #24 intermediary / hidden-resource audit.
- **Convergence:** converts "count to line 9 in a 15k blob" into "here is the broken line." Would have exposed all three corruption vectors on iteration 1 of the roasted run. Also directly attacks the `1 fixes, 0 includes, 0 APIs` plateau — GLM can now analyze errors past its 3000-char code window.
- **Failure mode (self-roast):** context is deliberately NOT folded into the `errors` strings — new/resolved diffing and cycle-detection hash those; folding it in would mark every error "new" forever. Do not "simplify" this later.
- **10B impact:** 1000X+ — this channel is kernel-agnostic and model-agnostic.

### ✅ 5. AST import tripwire
- **Change:** `tests/test_imports_resolve.py` — walks every `src/*.py` AST; any use of a known stdlib module name without an import anywhere in the file fails in <1s. Proven: fails on the pre-fix `main.py`, passes after.
- **TRIZ:** #23 feedback — moved from "22-minute GPU run" to "unit test".
- **10B impact:** infrastructure — protects the journey, not one kernel.

## Specified, not implemented (next pass)

### ☐ 6. Fixer output must be self-validating (the real Tier-1)
The pattern behind three incidents: `_fix_ported_code` has no check that its own output is sane. Add a post-fix invariant pass (pure Python, no LLM): (a) no `#define <digit>`; (b) every identifier the shim *uses* is declared before use in the emitted text; (c) balanced braces/parens delta vs. input. On violation, **return the unfixed code and log loudly** — a missed fix is recoverable by the loop; an injected defect is not. `src/router.py:_fix_ported_code` tail, ~30 lines. TRIZ #9/#11. **10B: 1000X+** — this is the class fix; items 2–3 are instances.

### ☐ 7. Blame attribution for post-processor lines
`_fix_ported_code` already returns a changelog; the shim/defines have known markers. When an error's context line (item 4) matches injected text, tag it `origin: "post_processor"` in `error_origins` and *skip the GLM analyst + Kimi refine for that error* — no model can fix it; only item 6's revert can. Extends the existing `harness`/`link`/`ported_code` origin taxonomy in `verifier.py:_classify_error_origin`. TRIZ #10. **10B: 100X.**

### ☐ 8. Re-plan budget spent only on model-origin errors
This run spent 5/5 informed re-plans on post-processor errors. Gate `GLM→DeepSeek` re-plan on `any(origin == "ported_code")`. `src/router.py` re-plan trigger block. TRIZ #22 trimming. **10B: 10X** (budget efficiency, not capability).

### ☐ 9. GLM display shows content, not counts
`router.py` prints `💡 GLM: N fixes…` — print the first `exact_fix[:60]` under it. One line; converts the operator's view from "GLM said something" to "GLM said *this*", which is how the flat-counts pathology stayed invisible for 5 iterations. **10B: observability.**

### ☐ 10. Tasking-doc items downgraded with cause
- *"Discard re-plans >15% longer"* (doc Tier-1 #2): *rejected as written* — plan inflation was a symptom of an unsolvable input; a length gate would have discarded plans while the injector kept injecting. Revisit only if inflation persists after items 6–8.
- *"Kimi ignores the plan"* (doc Tier-1 #3): premise disproven — the error set was invariant because the corruption was re-applied post-hoc, not because Kimi ignored plans. Close.
- *Cycle hard-abort* (doc Tier-2 #4): abort fired at iter 6/10 this run via plateau+stagnation. Acceptable; tighten only with data from post-fix runs.
- *Cross-run pattern memory, adaptive A2A routing, PromptCompiler, confidence-weighted channels* (doc Tier-3): all deferred past July 11 — each adds a new failure surface to a loop whose current blocker was three deterministic bugs. The 10B journey is currently gated on the fixer, not on learning.

## Verification

```bash
PYTHONPATH=src python -m pytest tests/ -q        # 152 passed (149 + 3 new)
```

On the notebook:
```bash
cd /workspace/Kernel-Olympics && git pull
python3 src/main.py --input sample_kernels/cuda/nvidia_shfl_scan.cu --fresh
```
Check, in order: (1) no crash at the Verifying phase; (2) no `error` at column 56 / no `macro name must be an identifier`; (3) Kimi's feedback and GLM's analysis now contain `SOURCE AT ERROR LOCATIONS` blocks — read them, they are the ground truth this whole report was reconstructed from; (4) if errors remain, they should finally be *in Kimi's own code* — which is the loop working as designed for the first time on this kernel.
