# Fable: KernelOlympics PR #13 Post-Mortem — Gap Analysis

**Prompt Version:** v3.0.0
**Created:** 2026-07-09
**Target:** 180s budget → successful compile + verify in one pass
**Fable Mode:** `claude -p "..." --append-system-prompt "$(cat prompts/to-fable-post-pr13-gap-analysis.md)" --allowedTools Read,Edit,Write,Bash --max-turns 25 --effort high`

---

## Context: What PR #13 Already Ship

The pipeline now has:
- HIPIFY mechanical preprocessor (0.0s, deterministic CUDA→HIP)
- 180s hard timeout (activates correctly — run was 181.2s)
- Stagnation detection (Δ+0, new:0 detected immediately)
- Phase budgets (20% plan, 55% code reserve, 25s compile reserve)
- Prompt versioning (`prompts/CHANGELOG.md`, `data/prompt_changelog.json`)

**Result:** Run completed in 181.2s, $0.03, but **FAILED** — `nvidia_shfl_scan.cu` hit TIMEOUT with "undefined symbol: main".

---

## The Run — Line by Line Gap Analysis

### Phase: Scanning & Classifying (PASS) ✅
```
✓ nvidia_shfl_scan.cu: coverage: 99.8%
→ nvidia_shfl_scan.cu: [medium] L253: warp_size_constant, [medium] L280: warp_size_constant, [high] L78: shfl_up_sync
```
No issues. Classifier correctly identifies warp_size_constant (x2) and shfl_up_sync.

### Phase: Memory Cache (PASS) ✅
```
● Memory Cache 0 cached patterns (first run)
```
Expected for first run. No fix needed.

### Phase: Porting — pre-processor (PASS) ✅
```
● pre-processor mechanical CUDA→HIP translation        0.0s
🔨 hipcc    compile-check hipify baseline          0.5s
```
HIPIFY works. Baseline compiles in 0.5s. **But the pipeline proceeds to DeepSeek anyway,** even though the HIPIFY baseline compiles and the only issue is `undefined symbol: main` (a harness problem, not a code problem).

**GAP #1:** When HIPIFY baseline compiles successfully, the pipeline should check: "is the failure just a missing main() wrapper?" If yes, DIAGNOSE it in < 1s instead of burning 38s on a DeepSeek plan.

### Phase: Porting — DeepSeek plan (WASTE) ⚠️
```
🧠 DeepSeek-v4-pro planning CUDA→HIP strategy             38.2s
```
38.2s = 21% of the 180s budget. The plan was meaningful (3197 chars), but **what did it plan for?** The HIPIFY draft already has the mechanical translation done. DeepSeek should only plan wavefront semantics — which for this kernel means fixing the `__shfl_up_sync` width. That's a 1-line change, not 38s worth of planning.

**GAP #2:** DeepSeek still plans the ENTIRE porting strategy, not just the wavefront-semantics delta. When `hipified_source` is provided, DeepSeek's prompt should be ~5 lines: "Here's what's already done. Only plan the remaining wavefront issues."

### Phase: Porting — Kimi code (WASTE) ⚠️
```
⚡ Kimi K2.7 generating HIP port from plan          58.1s
```
58.1s = 32% of budget. For what? The HIPIFY baseline already compiles. Kimi should be EDITING the draft, not regenerating from scratch. The preprocessed_source prompt was supposed to narrow this to "edit the drafted HIP code", but it still took 58s.

**GAP #3:** Kimi's "edit" prompt is too verbose. It should be: "Here's the HIPIFY draft that compiles. Fix the N remaining wavefront issues. Do not rewrite the file."

### Phase: hipcc compilation (FAIL) ❌
```
🔨 hipcc    in-loop compilation check              0.7s
│  ⚠ ld.lld: error: undefined symbol: main
│  ⚠ clang++: error: linker command failed with exit code 1
│  📊 → 2 errs (Δ+0, new:0)
```
**2 errors, both linker errors for `main()`.** This is NOT a compilation error — it's a LINKER error. The kernel is self-contained (`nvidia_shfl_scan.cu` has its own `int main(...)`). But Kimi's port either:
- Stripped the `main()` function (most likely — regression of T1.4)
- Or the HIPIFY preprocessor dropped `main()` (unlikely — HIPIFY just renames APIs)

**GAP #4:** The pipeline treats "undefined symbol: main" as a CODE error and tries to fix it via LLM. It should detect this as a HARNESS error: the source is self-contained, Kimi dropped main(), so just re-insert main() from the original. This is a 2-line fix, not a 180s loop.

### Phase: GLM analysis (PARTIAL) ✅
```
🔍 GLM-5.2  analyzing compile errors for Kimi (iter 1) 12.9s
│  💡 GLM: 2 fixes, 0 includes, 0 APIs
```
GLM identified 2 fixes in 12.9s. Both are "fix the undefined reference to main" — but GLM doesn't know this is a harness issue, not a code issue. GLM's prompts need context about self_contained detection.

**GAP #5:** GLM's analysis prompt doesn't include the self_contained flag. When a kernel has its own `main()`, GLM needs to know so it can tell Kimi "restore main() from the original source" instead of generic "fix the linker errors."

### Phase: DeepSeek re-plan (WASTE) ⚠️
```
🔄 DeepSeek-v4-pro informed re-plan after GLM (iter 1)    38.1s
│  🧠 DeepSeek-v4-pro re-plan landed (3197 chars)
```
Another 38s on planning. For what? The only fix needed is: "restore main() from the original source." DeepSeek doesn't need to re-plan the entire porting strategy.

**GAP #6:** Re-plan should be short-circuited for LINKER errors. If the error is "undefined symbol: main", skip DeepSeek entirely and go straight to: "restore main() from original CUDA source, append it to the HIPIFY draft."

### Phase: Kimi refine → TIMEOUT ❌
```
│  ⏱ TIMEOUT: 180s budget spent — returning best attempt
🔁 Kimi K2.7 refining with compile errors (iter 1→2) 31.1s
```
Kimi started refining (31.1s spent) but the 180s budget expired during the call. The refine was cut off in the middle — no output, no fix applied.

**GAP #7:** The 180s budget doesn't account for in-flight LLM calls. When a Kimi refine is in progress and the budget expires, the budget should extend slightly to let the in-flight call finish, then check its output. Or the timeout should check remaining budget BEFORE dispatching a new LLM call.

### Phase: Final verification (BROKEN) ❌
```
✅ Gemma 4  final verification                     0.0s
→ nvidia_shfl_scan.cu: TIMEOUT wall-clock budget spent — best attempt kept
⚠️  test_nvidia_shfl_scan.cpp:55:39: error: use of undeclared identif
📦 Quarantined best attempt (iter 0) — resume only, not served
```
Gemma 4 was used for final verification (was this switched from GLM?). The verifier created a test harness that also failed (undeclared identifiers in test_nvidia_shfl_scan.cpp). The code was correctly quarantined.

**GAP #8:** Gemma 4 replaced GLM-5.2 for final verification. Verify the model catalog aligns with expectations. Also, the verifier's test harness has its own compilation errors — this is a separate bug.

### Budget Analysis
```
DeepSeek plan:      38.2s  (21%)
Kimi code:          58.1s  (32%)
hipcc compile:       1.4s   (0.8% — should be ~25s budget)
GLM analysis:       12.9s   (7%)
DeepSeek re-plan:   38.1s  (21%)
Kimi refine:        31.1s  (17%) — cut off by timeout
Final verify:        ~1s   (0.5%)
Overhead/other:      0.4s
Total:             181.2s
```
**LLM total: 178.4s (98.5% of budget). hipcc: 1.4s (0.8%).** The budget is 99% LLM, 1% compiler. This is the core problem. The LLM budget needs to be compressed.

---

## Root Cause Summary

| # | Gap | Budget Wasted | Fix |
|---|-----|--------------|-----|
| 1 | Self-contained `main()` dropped | ALL of it | Detect self-contained → preserve main() from original source |
| 2 | DeepSeek plans 100% instead of 20% (wavefront delta) | 38s | Shrink plan prompt; skip planner entirely when error is linker-only |
| 3 | Kimi rewrites from scratch instead of editing HIPIFY draft | 58s | Shrink code prompt; "edit this draft, don't rewrite" |
| 4 | Linker error "undefined symbol: main" treated as compile error | 38s + 31s | Short-circuit: skip LLM loop, just restore main() |
| 5 | GLM doesn't know about self_contained flag | 12.9s (minor) | Pass self_contained to GLM prompt |
| 6 | Re-plan fires for linker errors | 38s | Skip DeepSeek re-plan when error type is "linker: main" |
| 7 | Timeout kills in-flight LLM call | 31s wasted refine | Extend budget for in-flight calls, or check budget before dispatch |
| 8 | Gemma 4 verification confusion | ~1s (minor) | Verify model config is consistent |

---

## Required Fixes (Priority Order)

### 🔴 P0: Self-contained main() preservation (saves 100% of loop)
The single fix that makes this kernel work. When `_is_self_contained(source)` is True:
1. Extract `main()` from the original CUDA source (regex or save it before HIPIFY)
2. Append it to the HIPIFY draft BEFORE any LLM call
3. If hipcc fails with "undefined symbol: main", restore main() from saved copy — no LLM needed

**Implementation:** Add a `_preserve_main()` method + main() extraction in the route loop before llm calls. Check hipcc output for "undefined symbol: main" and short-circuit.

### 🔴 P0: Linker error short-circuit (saves ~88s)
When hipcc error contains "undefined symbol: main" or "linker command failed":
- Skip DeepSeek re-plan entirely
- Skip GLM analysis entirely (unless there are REAL compile errors too)
- Just restore main() and re-compile
- If that fixes it, skip the LLM loop entirely

### 🟡 P1: Shrink DeepSeek plan for hipified source (saves ~30s)
When `hipified_source` is provided, DeepSeek's prompt should be a short wavefront-only plan. Target: < 5s for the plan call (use smaller model? or shorter prompt).

### 🟡 P1: Shrink Kimi code for hipified source (saves ~40s)
When `preprocessed_source` is provided, the Kimi prompt should be:
```python
"EDIT this HIP draft. Do NOT rewrite from scratch. Fix only the issues below.\n"
"Changes already done: header swaps, API renames, shuffle masks, checkCudaErrors.\n"
"Remaining work: wavefront semantics only.\n```hip\n{draft[:2000]}\n```\n"
```
Target: < 15s for Kimi code generation.

### 🟢 P2: Budget-aware LLM dispatch
Before calling any LLM, check `deadline.remaining()`. If it's less than the estimated call time + compile_reserve_seconds, skip and return best attempt.

### 🟢 P2: GLM model config verification
Verify that GLM-5.2 (not Gemma 4) is used for error analysis. If there's a model fallback chain, document it.

---

## Constraints

- No `shell=True` in subprocess calls
- All changes must pass: `python -m pytest tests/ -x -q`
- Every code change must include a version bump and changelog entry
- Keep the HIPIFY preprocessor — it works and costs 0.0s
- The 180s hard timeout works — keep it, but add budget-aware dispatch

---

## Done Definition

1. `python -m pytest tests/ -x -q` passes
2. `nvidia_shfl_scan.cu` or equivalent self-contained kernel compiles + verifies in < 180s
3. Self-contained main() is preserved through the pipeline
4. "undefined symbol: main" is detected as a linker error and short-circuits the LLM loop
5. DeepSeek plan with hipified_source is < 10s
6. Kimi code with preprocessed_source is < 20s
7. Budget is checked before LLM dispatch (no more cut-off in-flight calls)
8. Prompt version is bumped to v3.0.0 with changelog entry
