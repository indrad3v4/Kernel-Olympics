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

## v3.0.0 (2026-07-10)

MAJOR: the planner's instructions changed behaviorally, and the error analyst is
now told when the program it is judging owns a `main()` it cannot see. Fixes the
PR #13 post-mortem (`prompts/to-fable-post-pr13-gap-analysis.md`), where a run
spent 181.2s and shipped nothing because `nvidia_shfl_scan.cu` lost its driver.

- DeepSeek system prompt: gains an explicit exception — when handed a mechanically
  translated HIP draft, plan **only** the wavefront64 semantics delta and keep it to
  a terse checklist. Restating header swaps and API renames the draft already applies
  is work whose output the coder discards.
- DeepSeek plan prompt (`_build_deepseek_plan_prompt`): a `hipified_source` argument
  switches it to a delta prompt that embeds the HIP draft and **no CUDA original at
  all** (15,555 → 4,965 chars on `nvidia_shfl_scan.cu`). Output is capped at
  `PLAN_DELTA_MAX_TOKENS` (640) instead of the 2048 that a 38.2s plan was made of.
- GLM error-analyst prompt (`_build_glm_error_analysis_prompt`): a `self_contained`
  flag states that the original CUDA source owns a `main()` which may sit past the
  3000-char excerpt, so a missing driver must not be reported as a defect. If a
  linker error does name it, the only correct fix is "restore `main()` from the
  original source" — never "write a new `main()`" or "add a test harness".

Non-prompt changes shipped alongside (same commit):

- `_extract_main` / `_ensure_main_preserved` / `_postprocess_port`: the coder dropping
  `main()` from a self-contained program is repaired with a brace-matched extraction
  from the original, mechanically hipified, **before the first compile**. No model is
  in that loop. `main()` is cut from the hipified *whole* source, never hipified in
  isolation — the latter re-injects the helper shims and `#define WAVEFRONT_SIZE`,
  trading one link error for a dozen redefinition errors.
- `_is_linker_only` / `_is_missing_main_error`: when every hipcc diagnostic is a link
  failure, the GLM analyst and both DeepSeek re-plans are skipped. Neither can supply
  a symbol that is absent; on 2026-07-09 they cost 38.2 + 12.9 + 38.1s to say nothing.
- Budget-aware refine dispatch: a Kimi refine (and its 1.5x retry) is not started
  unless the clock holds the call plus `COMPILE_RESERVE_SECONDS`. The old code began a
  refine with ~31s left and the deadline killed it mid-flight, returning nothing.

---

## v2.0.0 (2026-07-10)

MAJOR: the coder's instructions changed behaviorally. It is now handed a
mechanically hipified draft and told to **edit** it rather than re-port from
scratch. See `_hipify_source` in `src/router.py`.

- Kimi code prompt: when a `preprocessed_source` draft is supplied, the checklist
  drops the 5 mechanical items (headers, `checkCudaErrors`, `cuda*→hip*`,
  `WAVEFRONT_SIZE`) and keeps only the 4 wavefront64 semantics items a regex
  cannot do. The draft is embedded as "HIP DRAFT TO EDIT"; the CUDA original is
  demoted to "reference only — do not port this again".
- Kimi refine prompt: gains a "MECHANICAL PASS ALREADY APPLIED" note, and names
  any `cuda*` symbols that crept back into the previous output. The draft itself
  is deliberately NOT re-embedded — `previous_code` is already the working copy,
  and re-sending it would grow the prompt this change exists to shrink.
- Output-token budget is now sized to the kernel (`_compute_adaptive_max_tokens`)
  instead of always requesting kimi27's 16384-token ceiling.

Rationale: on `nvidia_shfl_scan.cu` the mechanical pass rewrites 53 `cuda*` API
calls, 2 CUDA includes, 2 NVIDIA helper headers and 38 `checkCudaErrors` sites in
~19ms. The coder was spending ~90s of LLM time reproducing them one token at a
time, and a full rewrite is also what reintroduced compile errors on 2026-07-09.

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
