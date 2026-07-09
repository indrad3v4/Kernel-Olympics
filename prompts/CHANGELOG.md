# KernelOlympics Prompt Changelog

Every system prompt in `src/router.py` carries a `[prompt vX.Y.Z]` tag matching
`router.PROMPT_VERSION`. Bump the version and add an entry here on **any** prompt
edit, then update `data/prompt_changelog.json` to match — `tests/test_demo_budget.py`
asserts the three stay in sync.

**Bumping rules**

- **MAJOR** (v1.0.0 → v2.0.0): Behavioral changes (new instructions, removed constraints)
- **MINOR** (v1.0.0 → v1.1.0): Context additions (new examples, expanded edge cases)
- **PATCH** (v1.0.0 → v1.0.1): Bug fixes in prompt wording, typos, formatting

---

## v1.0.0 (2026-07-09)

Initial versioned baseline. The prompt text is unchanged from what shipped before
this entry — only the `[prompt v1.0.0]` tags and this changelog are new, so runs
before and after are directly comparable.

- Initial system prompts for DeepSeek (planner), Kimi (coder), GLM (evaluator)
- `glm_error_analyst` role lifted out of its inline call site in `route()` into
  `SYSTEM_PROMPTS` so every system prompt carries a version tag from one place
- Kimi refine prompt gained the two-layer SIGSEGV constraint: when a kernel
  compiles but crashes at runtime, the compiling kernel is passed as a frozen
  Layer 1 baseline and Kimi is instructed to patch rather than re-port it
- Stagnation notice in the refine prompt now reports the real threshold
  (`STAGNATION_ABORT_THRESHOLD`) instead of a hardcoded `3`
- PromptOptimizer seed checklist (9 items)
- Cache-based memory retrieval

---

## Prompt versions vs. run results

`route()` returns `prompt_version` in its result dict and records the wall-clock
budget in `changes`, so a run's transcript identifies the prompt text that
produced it. Populate `run_results` in `data/prompt_changelog.json` when a version
is exercised against a real MI300X run.
