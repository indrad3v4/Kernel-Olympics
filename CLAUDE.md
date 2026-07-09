# CLAUDE.md

This file tells you what this repo is, what I'm doing in it, and how to help me without wasting cycles.

---

## What this is

Kernel-Olympics: a hackathon project. GPU kernel benchmarking + LLM-assisted analysis pipeline. ROCm/HIP stack. Not production, but should be clean enough to demo without embarrassment.

I'm a **contributor**, not the owner. All changes go through PRs. Do not push to main.

---

## Current work surface: P0 → P1 → P2

These are the only tasks that matter right now. Do them in priority order. P0 first, always.

### P0 — The tool lies about its own results   [IMPLEMENTED on branch p0-pipeline-integrity]

```
T0.1  COMPILED label is contradicted two lines later
      main.py:335 sets the tag from the in-loop compile_ok on the bare kernel
      verifier then compiles kernel + generated harness at main.py:74 and fails
      Two compilers, two verdicts, one screen
      Fix: one source of truth
      The in-loop compile must use the same harness the verifier uses,
      or the label must read "kernel compiles, harness pending" — never bare "COMPILED"

T0.2  Confidence is fiction when compile fails
      main.py:336 header shows 47%
      main.py:522 cache line shows 8.5%
      Reality is "does not compile"
      Fix: verification-gate confidence
      If compile_success is false, clamp confidence to ~0
      Report says FAILED, not 47%

T0.3  Exit status does not reflect failure
      Tool prints "Report Generated", saves a report, exits clean on a kernel that never compiled
      Fix: non-zero exit + top-line "RESULT: FAILED" when nothing verified

T0.4  Cache poisoning — non-compiling code is stored into the live cache
      main.py:510–535 stores best_attempt_code on failure with confidence=0.085
      fallback path stores at 0.10
      There is no verified flag and no confidence floor on retrieval
      Next similar kernel can get a fake cache hit on broken code
      Fix: add verified boolean column
      Retrieval filters to verified-only, or at least verified + thresholded
      Unverified resume attempts live in a separate quarantine store

T0.5  Pattern count celebrates garbage
      main.py:156 reports "Patterns: 0 → 1 stored" even when the stored artifact is broken
      Fix: count verified patterns separately from quarantined attempts
```

### P1 — The 14-minute, $0.11 bonfire

```
T1.2  No time or cost budget
      router.py:1458 sets max_iterations=10
      At ~150s/iter that is a ~25-minute worst case, uncapped
      Fix: add --max-seconds and --max-cost
      Lower default iterations and enforce early-exit
```

Pruned 2026-07-09 — verified already fixed on main @ f3affa4:
- T1.1 (stagnation) now aborts via runtime_stagnation / harness_origin /
  kimi_plateau paths (router.py ~1763/1815/1926), not just a counter.
- T1.3 (shim) now has a real SHIM FIX success path + clean abort, not
  instant-fail theater (router.py ~1903–1934).
- T1.4 (self-contained) guard at verifier.py:196 fires for self_contained
  specs, so the verify path no longer wraps a second main().

### P2 — It fumbles its own hello

```
T2.1  Double banner + silently-dropped --silent flag  [STILL LIVE 2026-07-09]
      main.py:172 constructs Display(silent=silent)
      main.py:186 immediately overwrites it with Display()
      Result: banner prints twice and --silent does nothing
      Fix: construct Display once and pass silent through
      (Reviewer thought this landed in PR #8/#9 — it did NOT; origin/main
       @ f3affa4 still double-constructs Display. Verified against code.)

T2.2  Box borders overflow
      Long lines and emoji-heavy status text blow past the ║ right border
      Padding is not width-aware
      Fix: use wcwidth-aware padding

T2.3  Error snippets truncate mid-word
      Example: "use of undeclared i"
      Fix: truncate on a word boundary or widen the truncation window

T2.4  Stale phase timer
      Header shows "● Porting 0.0s" while sub-steps show ~145s
      main.py:319–324 captured the timer before the work
      Fix: stamp the timer after the phase actually completes
```

---

## Conventions

- One task per commit. Commit message = task ID + one line. e.g. `T0.4: add verified flag and quarantine failed cache writes`
- Python: UTF-8 everywhere, no exceptions
- Don't refactor things outside task scope. Scope creep is noise.
- If something looks broken but isn't in my task list, note it in a comment, don't fix it silently

---

## What I don't need

- Explanations of what UTF-8 is
- Suggestions to "consider" things
- Partial fixes with "you may also want to..."

Just find the things, fix the things, show me the diff.
