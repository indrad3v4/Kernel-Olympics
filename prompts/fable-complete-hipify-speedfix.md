# Fable Prompt: Complete HIPIFY Preprocessor + Speed Fix

**File:** `/root/Kernel-Olympics/src/router.py`  
**Tests:** `/root/Kernel-Olympics/tests/test_demo_budget.py`, `/root/Kernel-Olympics/tests/test_router.py`  
**211 tests currently pass.** Changes must preserve all passing tests.

---

## Problem

Pipeline is **too slow** (238s per kernel). The 180s `MAX_PIPELINE_SECONDS` budget is consumed entirely by iteration 0:
- DeepSeek plan: **86.9s** (48%)
- Kimi code: **93.5s** (52%)
- Total: **180.4s → TIMEOUT → 0 compile iterations → FAIL**

**Hidden bottleneck:** nvidia_shfl_scan.cu (15k chars, 419 lines, **42 cuda API calls**, self-contained `main()`). But HIPIFY's mechanical translator handles ALL 42 API renames + 5 `#include` swaps in **~0.01s**. It's never called.

---

## Already Deployed (verify on disk)

Subagents already added these to `router.py`:

| Change | Location | Impact |
|--------|----------|--------|
| `_hipify_source()` — 60+ regex patterns, pure Python, 0.01s | line ~899 | Handles all mechanical CUDA→HIP transforms |
| Compile-first fast-path — compiles hipify output before ANY LLM | line ~1980 | If hipify compiles → skip entire pipeline, save 180s |
| `_compute_adaptive_max_tokens()` — scales Kimi's token budget | line ~626 | Small kernels (~4k chars) get 2000 tokens instead of 16384 |
| Preprocessed source passed to Kimi initial port | line 2023 | Kimi edits mechanical draft instead of full rewrite (saves ~50s) |
| Shortened plan context 4000→1500 chars | line ~1446 | Smaller prompt = faster LLM response |
| Reduced checklist 10→4 items with preprocessed source | line ~1429 | Kimi only handles warp→wavefront semantics |

Tests already fixed in `test_demo_budget.py` and `test_router.py`. All pass.

---

## What's NOT Done (Must Fix)

**The refine loop** at line ~2989 does NOT pass `preprocessed_source` to `_build_kimi_refine_prompt()`. Every Kimi refine iteration re-explains mechanical transforms instead of editing the existing draft. Also doesn't pass `max_tokens_override`.

The parameter `preprocessed_source` already exists on `_build_kimi_refine_prompt()` (line 1529) — it just never gets passed from the call site. The variable `hipified_source` is already in scope (defined at line 1962, before the for-loop). Same for `adaptive_tokens` (line 2027).

### Exact Edits

**Edit 1 — router.py line ~2989:** Add `preprocessed_source=hipified_source`:

```python
refine_prompt = self._build_kimi_refine_prompt(
    kernel_source, result["ported_code"],
    evaluator_feedback, patterns,
    deepseek_plan=deepseek_plan_output,
    iteration=iteration + 1,
    checklist_override=evolved.checklist,
    stagnation_count=stagnation_count,
    regex_changelog=result.get("regex_changelog"),
    frozen_base_code=(frozen_base_code if run_crashed_this_iter else ""),
    preprocessed_source=hipified_source,  # ADD THIS
)
```

**Edit 2 — router.py line ~3002:** Add `max_tokens_override=adaptive_tokens`:

```python
refine = self._call_model(
    "kimi27", refine_prompt,
    system_prompt=SYSTEM_PROMPTS.get("kimi27", ""),
    max_tokens_override=adaptive_tokens,  # ADD THIS
)
```

**Edit 3 — router.py line ~3024:** Same for the retry:

```python
retry_refine = self._call_model(
    "kimi27", refine_prompt,
    system_prompt=SYSTEM_PROMPTS.get("kimi27", ""),
    max_tokens_override=adaptive_tokens,  # ADD THIS
)
```

### Verification

```bash
cd /root/Kernel-Olympics
python3 -c "import ast; ast.parse(open('src/router.py').read()); print('Syntax OK')"
python3 -m pytest tests/ -q --tb=short
```

Must pass all tests. If any fail, the mock `quick_compile_check` on the failing test's `MagicMock` verifier needs a hipify fail entry prepended to its `side_effect` list:
```python
{"compile_success": False, "errors": ["hipify not enough"], "output": "", "error_context": []}
```

---

## Expected Speedup

| Scenario | Before | After |
|----------|--------|-------|
| Small kernel (<3k chars): warp_reduce.cu | ~30s | HIPIFY→compile passes→skip LLM → **~5s** |
| Outlier (15k chars, 42 API calls): nvidia_shfl_scan.cu | **180s TIMEOUT** | HIPIFY→narrow plan(60s)+Kimi edits draft(50s)→**110s** ↔ 70s budget left for iteration |
| Each refine iteration | ~90s | **~50s** (Kimi edits existing draft) |
