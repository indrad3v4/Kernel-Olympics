# TRIZ Analysis: Pipeline Budget Contradiction

**File:** `/root/Kernel-Olympics/src/router.py`
**Constants:** `MAX_PIPELINE_SECONDS = 180` (line 290), `MIN_LLM_TIMEOUT_SECONDS = 5` (line 294)
**Per-model timeouts:** deepseek=120, kimi27=180, glm=120, gemma4=60 (MODEL_CATALOG, lines 215-263)
**Deadline class:** line 311, `Deadline(0)` = unlimited, `clamp_timeout()` at line 349
**State:** Converged loop (TRIZ #13 compile-first mode) — but budget kills it before 2nd iteration.

---

## 1. System Definition

### Main Function (Гл.Ф)
Port CUDA kernels to HIP within a bounded wall-clock window.

### Key Elements

| Element | Role | Time Budget |
|---------|------|------------|
| `Deadline` | Global wall-clock budget enforcer | 180s total |
| `_call_model` | LLM request dispatch + timeout clamping | Per-model: 60-180s |
| DeepSeek Phase 1 | Strategic planning | ~87s (observed) |
| Kimi Phase 2 | Initial code generation | ~94s (observed) |
| GLM error analysis | Compile error translation (~30s) | DeepSeek-informed |
| Kimi refine (iter N) | Code refinement (~90-180s) | Dwindling budget |
| hipcc compile check | Deterministic verification (~5s) | N/A |
| `clamp_timeout()` | Shrinks per-request timeout to remaining budget | Budget-dependent |

### Cost-per-iteration (compile-fail path)

| Step | Time (s) | Cumulative |
|------|---------|------------|
| Kimi refine | ~90-180 | ~90-180 |
| hipcc compile | ~5 | ~95-185 |
| GLM error analysis | ~30 | ~125-215 |
| **Total per iter** | **~125-215** | **—** |

With 180s budget and initial DeepSeek+Kimi consuming ~180s, **iteration 0 never begins** — the budget is exhausted before the refine loop even starts.

---

## 2. Contradictions

### АП (Administrative)
"The pipeline needs to iterate to converge on working HIP code, but it runs out of time after the very first LLM calls."

### ТП-1 (Technical)
| Improves | Worsens |
|----------|---------|
| Larger budget → more iterations can converge | Longest run is unbounded — kills demo viability |
| Shorter budget → bounded run time | At most 1 iteration — convergence impossible |

### ТП-2 (Technical)
| Improves | Worsens |
|----------|---------|
| `clamp_timeout()` protects the budget | A Kimi refine call (needs ~90s) gets clamped to 30s left → guaranteed timeout |
| Generous per-model timeout | Kimi call consumes the entire budget solo |

### ФП (Physical)
The budget must be **bounded** (demo/sync) AND **unbounded** (convergence needs 3+ iterations).

> Same resource (the 180s wall clock) must be both large enough for many iterations and small enough to finish on schedule.

### ФП-2 (Physical)
`clamp_timeout` must shrink timeouts to protect the deadline BUT the shrunk timeout must still be large enough for a real LLM response.

> `MIN_LLM_TIMEOUT_SECONDS = 5` is not enough for any model. The floor is 5, but hipcc needs ~5s alone — leaving no room for any LLM call.

---

## 3. Resources (ВПР)

| Resource | Status | Current Use |
|----------|--------|------------|
| Time (180s budget) | **Over-constrained** | Cannot fit 2 Kimi calls + 2 GLM calls + compile |
| Time (per-model timeout) | Underutilized for fast paths | Kimi=180 even when compile passes on iter 0 |
| Kernel complexity signal | **Unused** | All kernels get same 180s budget |
| Compile-pass signal | Late-bound | Budget already spent when compile passes |
| Early-exit on compile-pass | Exists (line 2650) but too late | Budget may already be consumed by initial calls |
| Iteration count | Max 10 | Hard-limited by budget to 0-1 |
| `max_seconds` override param | Exists (line 1626) | **Unused by callers** — always None → falls through to MAX_PIPELINE_SECONDS |

---

## 4. Proposed Resolutions

### Proposal 1: Dynamic Budget Based on Kernel Complexity

**TRIZ Principle:** #15 (Dynamics) + #3 (Local Quality)

**Contradiction:** All kernels get 180s — simple kernels with trivially ported code waste budget; complex kernels can't converge.

**Resolution:** Compute budget based on kernel source characteristics:

```python
def compute_budget(kernel_source: str, budget_s: int = 180) -> int:
    """Scale budget by kernel complexity — simple kernels need less time."""
    lines = kernel_source.count('\n')
    api_calls = len(re.findall(r'\b(cuda[Mm]alloc|cuda[Mm]emcpy|cuda[Ll]aunchKernel'
                               r'|__syncthreads|__shfl|__ballot)\b', kernel_source))
    unique_includes = len(set(re.findall(r'#include\s+<([^>]+)>', kernel_source)))

    # Complexity score: lines × 0.3 + API calls × 15 + includes × 10
    complexity = lines * 0.3 + api_calls * 15 + unique_includes * 10

    # Scale budget: min 120s for trivial kernels, max 600s for complex ones
    scaled = max(120, min(600, int(budget_s * (complexity / 50 + 0.5))))
    return scaled
```

**Integration point:** Replace line 1626:
```python
# BEFORE:
budget = MAX_PIPELINE_SECONDS if max_seconds is None else max_seconds
# AFTER:
budget = compute_budget(kernel_source) if max_seconds is None else max_seconds
```

**Pitfall:** Must cap at a configurable maximum to prevent unbounded runs.

---

### Proposal 2: Progressive Timeout

**TRIZ Principle:** #19 (Periodic Action) + #10 (Preliminary Action)

**Contradiction:** First call needs full timeout (180s for Kimi to generate from scratch); subsequent refines need less time (the input is smaller, the model has context).

**Resolution:** Tiered timeouts that shrink per-iteration:

```python
# In MODEL_CATALOG, add a "progressive_timeout" field:
"kimi27": {
    ...
    "timeout": 180,
    "progressive_timeout": {  # tiered by iteration
        0: 180,   # initial generation: generous
        1: 120,   # first refine: less to generate
        2: 90,    # second refine: incremental changes only
        3: 60,    # third refine: tiny fixes
    },
}
```

**Integration point:** In `_call_model` (line 2846), resolve timeout from iteration:
```python
# BEFORE:
model_timeout = model_info.get("timeout", 90)
# AFTER:
progressive = model_info.get("progressive_timeout", None)
if progressive:
    # Use current iteration from Agent's context
    model_timeout = progressive.get(self._current_iteration, model_info["timeout"])
else:
    model_timeout = model_info.get("timeout", 90)
```

**Second layer: `_call_model` tracks its own retry state:**
```python
# When retrying (attempt=1), use 1.5x the current tier's timeout
attempt_timeout = deadline.clamp_timeout(model_timeout * (attempt + 1))
```

**Benefit:** First Kimi call gets 180s guaranteed. Second refine gets 120s (fits within remaining budget). Third gets 90s (incremental fix). The progressive curve acknowledges that later iterations have less to generate.

---

### Proposal 3: Early-Exit on Compile Success (Before the Loop)

**TRIZ Principle:** #11 (Beforehand Cushioning) + #20 (Continuity of Useful Action)

**Contradiction:** The loop exists to fix compile errors, but we only check for compile errors inside the loop — after burning DeepSeek+Kimi budget.

**Resolution:** Compile the **original CUDA source** before any LLM calls. Wait, that doesn't make sense — CUDA needs nvcc. Instead:

**Sub-resolution: Compile check between DeepSeek plan and Kimi code generation**

The current order is: DeepSeek (plan) → Kimi (code) → compile check (inside loop).  
Move the hipcc compile to run on a simple regex-based CUDA→HIP transform BEFORE any LLM call:

```python
# BEFORE any LLM call — attempt a fast regex port + compile check
# If the regex port happens to compile, we saved TWO LLM calls.
fast_port = regex_port_only(kernel_source)  # ~0.1s
quick_cc = verifier.quick_compile_check(fast_port, kernel_name=kernel_name)
if quick_cc["compile_success"]:
    result["ported_code"] = fast_port
    result["changes"].append("[fast-path] Regex-only port compiled — skipping LLM pipeline")
    # Skip DeepSeek+Kimi entirely, go straight to GLM semantic eval
```

**More important early-exit: Check budget BEFORE starting the loop**

```python
# After initial Kimi call (phase 2), check if enough budget remains for 1 iteration:
REQUIRED_PER_ITER = MODEL_CATALOG["kimi27"]["timeout"] * 0.7  # ~126s
if deadline.remaining() < REQUIRED_PER_ITER:
    result["changes"].append(
        f"[budget] Only {deadline.remaining():.0f}s remain, need ~{REQUIRED_PER_ITER}s "
        f"for one refine iteration — skipping loop, returning initial port")
    result["timed_out"] = False  # This is intentional, not a timeout
    result["iterations_used"] = 0
    return result
```

**Even tighter early-exit with compile-gate:**
```python
# Initial code compiled on first try? Don't enter the loop at all.
if compile_passed:
    # Still run GLM semantic eval
    evaluator = self._call_model("glm", eval_prompt, ...)
    if evaluator.success and parsed.get("pass", False):
        result["iterations_used"] = 0
        return result  # Done — code compiles AND passes semantic check
```

This saves 200s+ of refining code that already works.

---

### Proposal 4: Budget-Aware Iteration Scheduling

**TRIZ Principle:** #25 (Self-Service) + #23 (Feedback)

**Contradiction:** The loop uses the same expensive operations every iteration even when the remaining budget can't support them all.

**Resolution:** Schedule operations based on remaining budget:

```python
# Before each iteration, compute the best action for remaining budget:
def _schedule_iteration(deadline: Deadline) -> str:
    """Return the optimal action given remaining budget."""
    rem = deadline.remaining()
    if rem is None:
        return "full_iteration"   # unlimited budget
    if rem >= 180:
        return "full_iteration"   # Kimi refine + GLM + compile
    if rem >= 90:
        return "kimi_only"        # Kimi refine only, skip GLM, try compile
    if rem >= 30:
        return "compile_only"     # Don't call any LLM — just re-compile existing code
    return "abort"                # Too little budget, return best attempt
```

**Inside the loop (line 1838):**
```python
action = _schedule_iteration(deadline)

if action == "abort":
    result["timed_out"] = True
    break

if action == "compile_only":
    # Just re-compile existing code — maybe the compile check itself is flaky
    if verifier and hasattr(verifier, 'quick_compile_check'):
        cc = verifier.quick_compile_check(result["ported_code"], kernel_name=kernel_name)
        if cc["compile_success"]:
            compile_passed = True
            break
        else:
            # Compile still fails, no budget to fix it
            break
    break

if action == "kimi_only":
    # Skip GLM error analysis — send raw compile errors directly to Kimi
    # Saves ~30s per iteration
    evaluator_feedback = build_raw_compile_feedback(compile_errs)
    refine = self._call_model("kimi27", refine_prompt, ...)
    # After refine, compile check — but if it fails, NO second chance
    # This gives one last shot at convergence when budget is tight
```

**Budget schedule table:**

| Remaining Budget | Optimal Action | What Runs | Saves |
|-----------------|---------------|-----------|-------|
| 180-300s | Full iteration | Kimi + GLM + compile + DeepSeek re-plan | — |
| 120-179s | Kimi + compile only | Kimi refine + compile | GLM (30s) |
| 60-119s | Compile-only pass | Re-compile existing code | Kimi (90-180s) |
| 30-59s | Compile + check budget | One more compile, then abort | Everything |
| < 30s | Abort | Return best attempt | Prevents doomed call |

---

## 5. Combined Resolution (IFR)

**Ideal Final Result:** The pipeline converges in exactly the time available — no second wasted, no iteration cut prematurely.

### Implementation Priority

| # | Proposal | Effort | Impact | Payback |
|---|----------|--------|--------|---------|
| 1 | Dynamic budget (kernel complexity) | Low (1 function) | High: complex kernels get 600s | Every kernel gets fair time |
| 2 | Progressive timeout | Low (dictionary per model) | High: early refines need less time | 30-60s saved per iteration 2+ |
| 3 | Early-exit on compile success | Medium (add check points) | Very High: skip LLM calls entirely for working code | 180-600s saved on first-pass kernels |
| 4 | Budget-aware scheduling | Medium (scheduler logic) | High: adapts to remaining budget | Prevents doomed calls from burning budget |

### Integration Order

```
1. Add Proposal 3 (early-exit) — zero risk, saves most time
   → Integration: check compile before entering loop, exit on success

2. Add Proposal 1 (dynamic budget) — configurable, backward compatible
   → Integration: replace MAX_PIPELINE_SECONDS in route() with scaled value

3. Add Proposal 4 (budget-aware scheduling) — no harm with early-exit already in place
   → Integration: _schedule_iteration() at loop head (line 1842)

4. Add Proposal 2 (progressive timeout) — fine-tuning after structural fixes
   → Integration: progressive_timeout dict in MODEL_CATALOG, resolved in _call_model
```

### Current Code That Already Supports This (No-ops to Verify)

| Feature | Location | Status |
|---------|----------|--------|
| `max_seconds` parameter on `route()` | line 1626 | EXISTS but callers always pass None |
| Early-exit on GLM pass + compile pass | line 2650 | EXISTS but budget is often already exhausted |
| Budget-check at loop boundary | line 1846 | EXISTS but has no scheduler — just "run or die" |
| `clamp_timeout` with per-model timeout | line 2938 | EXISTS but always shrinks, never grows |

---

## 6. TRIZ Principle Reference

| Principle | Application |
|-----------|------------|
| #1 Segmentation | Per-iteration timeouts (Proposal 2) |
| #3 Local Quality | Budget scales to kernel (Proposal 1) |
| #10 Preliminary Action | Check compile feasibility before loop (Proposal 3) |
| #11 Beforehand Cushioning | Early-exit path for working code (Proposal 3) |
| #15 Dynamics | Schedule adapts to remaining budget (Proposal 4) |
| #19 Periodic Action | Progressive timeout shrinks per-iteration (Proposal 2) |
| #20 Continuity | Don't waste budget on doomed iterations (Proposal 4) |
| #23 Feedback | Remaining budget feeds scheduling (Proposal 4) |
| #25 Self-Service | Scheduler chooses optimal action from budget (Proposal 4) |
