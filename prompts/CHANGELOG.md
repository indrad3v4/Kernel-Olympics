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
