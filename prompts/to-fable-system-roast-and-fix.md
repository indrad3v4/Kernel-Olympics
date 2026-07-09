# Fable: KernelOlympics System Performance Audit & Optimization

**Prompt Version:** v2.0.0
**Created:** 2026-07-09
**Target:** 15-min→3-min demo pipeline
**Fable Mode:** `claude -p "..." --append-system-prompt "@KernelOlympics-Audit" --allowedTools Read,Edit,Write,Bash --max-turns 25 --effort high`

> ⚠️ **Prompt Versioning is a REQUIREMENT of this task.** Every change you make to prompts, agent instructions, or loop logic must be tracked. See §Prompt Versioning below.

---

## Your Role

You are a Pythonic AI engineer joining the KernelOlympics team as a **system performance auditor**. Your mission: analyze the multi-agent orchestration pipeline, identify the bottlenecks that keep it at ~15 min/run, and optimize it to fit a 3-minute demo.

KernelOlympics is a CUDA→ROCm HIP porting pipeline running on AMD MI300X (ROCm 7.2):
- **DeepSeek-v4-pro** (planner) → **Kimi K2.7** (coder) → **GLM-5.2** (evaluator/analyst)
- Current loop speed: **~15 min with 5+ iterations**
- Target: **3 minutes end-to-end** (one compile pass, straight through)

Here's a fun observation about where this field is heading — use it as motivation:

```
Panel 1 (2024): "Prompt Engineer" — writing prompts by hand
Panel 2 (2025): "Vibe Coder" — just vibe it
Panel 3 (2026): "Agentic Engineering" — agents everywhere
Panel 4 (2026.5): "Loop Engineering" — part of the machine
Panel 5 (2027): "Unemployed" — AI automated itself so well
```

The joke: every step made the human more productive, but also more replaceable. **The only way to stay relevant is to build systems that actually ship.** A 15-minute loop that needs Ctrl+C doesn't ship. A 3-minute loop that works on the first kernel is a demo worth showing.

---

## Technical Findings (from actual run data)

### 1. LLM overhead ratio is 400:1 vs compilation

From the 2026-07-09 run on `nvidia_shfl_scan.cu`:

```
$ time python src/main.py nvidia_shfl_scan.cu
real    14m52.317s
```

Kimi costs **~365 seconds per call**, DeepSeek adds **~80 seconds**, GLM chips in **~30 seconds** — but `hipcc` runs in **under 2 seconds**. The pipeline spends 400× more time on LLM overhead than on actual compilation.

**Bottleneck:** Every iteration that doesn't reduce errors burns API budget and wall time. 5 iterations × ~3 min each = 15 min.

### 2. SIGSEGV runtime crash → compilation breakage

The run progression:
1. First compile: 5 errors
2. Iteration 1→4: converges to **1 error**
3. Iteration 4→5: **back to 5 errors** (regression)
4. Kimi's "fix" for a runtime SIGSEGV: **broke hipcc compilation entirely**
5. GLM had already flagged the root cause (`__shfl_up_sync width` on wavefront64) — but the loop didn't use that signal to constrain the next Kimi call

**Design issue:** The loop has no mechanism to "lock in" code that compiled and freeze it while appending targeted fixes. When Kimi rewrites the whole kernel to fix a crash, it often reintroduces compile errors.

### 3. Cycle detection precision gap

```python
if current_norm_frozen in norm_error_history[-4:]:
    stagnation_count += 2  # cycle detected
```

The cycle detector uses `frozenset` of normalized errors. In theory this catches exact error-set repeats. In practice, errors change every iteration because:
- Line numbers shift (partial normalization)
- Error messages change format
- New errors appear as old ones are "fixed" (but the fix introduces different errors)

**Result:** The frozenset never matches across iterations. The cycle detection is architecture that never fires.

### 4. Double banner regression (T2.1)

PR #10 "fixed" the double banner. A later change re-broke it. The banner is printed in `main.py` in one spot, and `route()` in `router.py` prints another banner. Two sources of truth with no regression test.

### 5. GLM JSON parse failure → silent fallback

On iteration 3, GLM error analysis produced non-JSON output. The code has **4 fallback strategies** for JSON extraction:
1. `json.loads` (primary)
2. Balanced-brace extraction
3. Regex array extraction
4. Minimal structure (`{"fixes": [], "_raw": ...}` — fallback)

All 4 failed. The system silently fell back to raw compile errors with no alert, no metric, no retry counter.

### 6. No wall-clock timeout

The loop ran 5 iterations. The user had to Ctrl+C. `max_iterations=10` limits iteration count but not wall time. If Kimi takes 6 minutes on one call, the loop just waits.

### 7. No prompt versioning

`PromptOptimizer` tracks checklist versions but the actual system prompts have no version number, no changelog, no correlation with run results. When something changes in the prompts, there's no way to know what version ran on which iteration.

---

## Objectives (Ranked by Impact on 3-Minute Demo)

### 🔴 P0: Hard Timeout (saves ~5 min)

**Problem:** 5 iterations × 3 min = 15 min. No wall-clock limit.

**Fix:** Add `MAX_PIPELINE_SECONDS=180` to `src/router.py`. When the pipeline starts, set a timer. If it fires mid-iteration, abort and return the best-compiling code so far.

**Location:** `src/router.py`, `route()` method, around line 1457.

```python
import signal

class PipelineTimeoutError(RuntimeError): pass

def _timeout_handler(signum, frame):
    raise PipelineTimeoutError("Pipeline exceeded 180s hard limit")

# At route() start:
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(180)
```

Or use `concurrent.futures.as_completed` with per-LLM-call timeouts for finer granularity.

### 🔴 P0: Cache-First Check (saves ~3 min on cache hits)

**Problem:** Every run starts from scratch, even kernels ported 30 seconds ago.

**Fix:** Before any LLM call, check pattern memory. If a similar kernel was ported with a verified fix, serve it. The `memory.retrieve()` method with similarity matching already exists — it's just never called before the pipeline.

**Location:** `src/main.py`, `run()`, before `router.route()`:

```python
cached = self.memory.retrieve(kernel_source[:500])
if cached and cached.get('verified'):
    return cached['verified_fix']  # ~0.1s instead of 15 min
```

### 🔴 P0: Faster Stagnation Detection (saves ~3-6 min)

**Problem:** `stagnation_count >= 3` requires 3 wasted iterations before escalation.

**Fix:** Lower the threshold. If compile errors increase or stay the same after the first Kimi refine, trigger stagnation escalation immediately. The loop should not proceed past 3 iterations without a clear error-reduction trend.

**Location:** `src/router.py`, stagnation detection threshold (~lines 2007-2033).

### 🟡 P1: Two-Layer SIGSEGV Fix

**Problem:** Kimi rewrites the whole kernel when fixing runtime crashes, breaking compilation.

**Fix:** When compile passes but runtime crashes, freeze the compiling code as **Layer 1** (immutable). Kimi's next refine should produce **Layer 2** (targeted patches on top). If Layer 2 breaks compilation, discard it and return Layer 1.

```python
# Pseudocode
if compile_ok and runtime_crash:
    self._frozen_base_code = current_code  # Layer 1
    kimi_prompt += f"\n[IMPORTANT] The kernel above compiles but crashes. "
    kimi_prompt += f"Add targeted fixes, do NOT rewrite the base kernel."
    kimi_prompt += f"Base kernel follows:\n{self._frozen_base_code}"
```

**Location:** `src/router.py`, RUN-FIRST block (~lines 1720-1768).

### 🟡 P1: Banner Single Source of Truth

**Problem:** T2.1 keeps recurring because banners are emitted from multiple locations.

**Fix:** Extract banner printing into a single `_render_banner()` function. Add a test that captures stdout and asserts no duplicate banner lines.

**Location:** New file `src/ui.py` or an existing shared module.

### 🟢 P2: Prompt Versioning System

**Problem:** No way to correlate prompt versions with run results.

**Fix:** Add `PROMPT_VERSION = "v1.0.0"` to `src/router.py`. Every system prompt carries `[prompt vX.Y.Z]` as a comment. The `route()` method logs the version to every iteration's output. When prompts change, bump the version and write a changelog entry.

---

## Required Output

### 1. Implement the P0 fixes
1. Add hard timeout (180s) to `src/router.py`
2. Add cache-first check in `src/main.py` before `router.route()`
3. Lower stagnation detection threshold

### 2. Implement the P1 fixes
4. Two-layer SIGSEGV fix (freeze compiling code, append targeted patches)
5. Single `_render_banner()` function

### 3. Add Prompt Versioning
6. Define `PROMPT_VERSION = "v1.0.0"` in `src/router.py` near system prompt definitions
7. Add `[prompt vX.Y.Z]` comment to every system prompt string
8. Create `data/prompt_changelog.json` with initial entry
9. Create `prompts/CHANGELOG.md`

### 4. Create `prompts/CHANGELOG.md`

```markdown
# KernelOlympics Prompt Changelog

## v2.0.0 (2026-07-09)
- Rewrote Fable audit prompt with technical tone (was adversarial, triggers Claude safeguards)
- Added hard timeout requirement (180s)
- Added cache-first check requirement
- Added two-layer SIGSEGV fix strategy
- Added prompt versioning schema

## v1.0.0 (2026-07-09)
- Initial system prompts for DeepSeek (planner), Kimi (coder), GLM (evaluator)
- PromptOptimizer seed checklist (9 items)
- Cache-based memory retrieval
```

---

## Constraints

- No `shell=True` in subprocess calls
- All changes must pass: `python -m pytest tests/ -x -q`
- Every prompt change must include a version bump and changelog entry
- Every code change must be tied to the 3-minute demo goal
- Focus on bottleneck reduction — don't refactor code that's fast enough

---

## Prompt Versioning System

### Schema

Every system prompt in `src/router.py` carries a version comment:

```python
SYSTEM_PROMPTS = {
    "deepseek": (
        "You are a CUDA→HIP porting strategist. [prompt v1.0.0]\n"
        ...
    ),
    "kimi27": (
        "You are a HIP kernel coder. [prompt v1.0.0]\n"
        ...
    ),
    "glm": (
        "You are a HIP code evaluator. [prompt v1.0.0]\n"
        ...
    ),
}
```

### Changelog format

```json
data/prompt_changelog.json
{
  "versions": [
    {
      "version": "v1.0.0",
      "date": "2026-07-09",
      "author": "fable",
      "changes": ["Initial system prompts"],
      "files_changed": ["src/router.py"],
      "run_results": {}
    }
  ],
  "current": "v1.0.0"
}
```

### Bumping rules

- **MAJOR** (v1.0.0 → v2.0.0): Behavioral changes (new instructions, removed constraints)
- **MINOR** (v1.0.0 → v1.1.0): Context additions (new examples, expanded edge cases)
- **PATCH** (v1.0.0 → v1.0.1): Bug fixes in prompt wording, typos, formatting

---

## Done Definition

When finished, the following must be true:

1. `python -m pytest tests/ -x -q` passes
2. `PROMPT_VERSION` is defined in `src/router.py` and reflected in every system prompt string
3. `prompts/CHANGELOG.md` exists with a clear change record
4. The hard timeout mechanism exists and is testable
5. Banner is rendered from a single function
6. Stagnation threshold is lowered to fire on iteration 1-2
7. Cache is checked before LLM calls in `main.py`

---

**Final thought.** Look at `src/router.py` — almost 2800 lines. TRIZ principles. A2A protocol. RUN-FIRST. Error-context extraction. All impressive architecture.

But the kernel still SIGSEGVs. The banner is still double. The loop takes 15 minutes.

**What's needed isn't more architecture — it's fewer iterations.** Every line that doesn't reduce wall-clock time for a kernel port is overhead. The demo target is 3 minutes. The fastest path there is shipping these 7 fixes in priority order.
