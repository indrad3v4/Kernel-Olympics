---
name: port-a-kernel
description: Run the CUDA→ROCm/HIP porting pipeline over one or more .cu kernels and read the result honestly. Use when asked to "port a kernel", "run the pipeline", "try it on <file>.cu", or to reproduce a run.
---

# Port a Kernel

Drives `src/main.py`, the DeepSeek (plan) → Kimi K2.7 (code) → GLM-5.2 (evaluate) loop, over
one or more CUDA kernels. Produces `ported_kernels/<name>.hip.cpp`, a run log under
`runs/<kernel>_<uuid>/`, and `portability_report.json`.

## Prerequisites

- `FIREWORKS_API_KEY` in `.env` at the repo root. `main.py` loads it at import time.
- `hipcc` for real verification. Without it the loop still runs; the verifier reports
  `hipcc_available: false` and no kernel can reach a PASSED verdict.
- Run from the repo root. `src/main.py` writes `ported_kernels/` and `runs/` relative to cwd.

## Steps

Check the environment first — this is cheap and it is the only way to know whether a
FAILED verdict means "bad port" or "no toolchain":

```bash
python src/main.py --doctor
```

Then port:

```bash
python src/main.py --input sample_kernels/cuda/nvidia_shfl_scan.cu --fresh
```

- `--fresh` clears the pattern-memory cache first. Use it when measuring a cold run;
  omit it to let a verified cached fix short-circuit the LLM calls (~0.1s instead of minutes).
- `--input` takes several files.
- `--demo` runs the "second kernel is faster" cache demo; `--reset` clears memory first.
- `--daemon --watch <dir>` polls a directory for new `.cu` files.

On Windows, prefix with `PYTHONIOENCODING=utf-8` or the box-drawing characters in the
display raise `UnicodeEncodeError` from `cp1252`.

## Reading the result

The loop has a wall-clock budget (`MAX_PIPELINE_SECONDS`, default 180s). Four outcomes,
distinguished by `abort_reason` in the result and by the changes log:

| Signal | Meaning |
|---|---|
| `fast_path_used: true` | mechanical hipify compiled **and** ran — zero LLM calls, `model_used: hipify` |
| `orchestrator_passed` | compile **and** run **and** GLM all passed |
| `abort_reason: pipeline_timeout` | budget spent; `ported_code` is the best **compiling** attempt |
| `abort_reason: layer2_rejected` | a refine broke a build that worked; the frozen kernel was returned |
| `abort_reason: runtime_stagnation` | 3 consecutive compile-pass-but-crash iterations |
| `abort_reason: hard_stagnation` | no error reduction after the re-plan budget ran out |

An in-loop "compile passed" is **not** the verdict. The Verifying phase compiles, runs, and
diffs against `sample_kernels/reference/<name>_output.txt`, and it owns the real
PASSED/FAILED. A binary that compiles and SIGSEGVs is a failure.

## Troubleshooting

- **`UnicodeEncodeError: 'charmap'`** — set `PYTHONIOENCODING=utf-8`.
- **`--doctor` reports `sqlite: write test FAILED` on Windows** — a false alarm, and not a
  reason to stop. The check opens a `NamedTemporaryFile` and asks sqlite to open it again;
  Windows forbids the second handle. Reproducible with three lines of stdlib and no project
  code. The real database, `data/pattern_memory.db`, opens fine.
- **Everything FAILS with `hipcc_available: false`** — no ROCm. `--doctor` says so. The ports
  are still written to `ported_kernels/` for manual compilation.
- **The run ends immediately with `timed_out: true` and no kernel** — the budget expired during
  the first Kimi call. Raise it: `MAX_PIPELINE_SECONDS=600 python src/main.py ...`
- **A run takes far longer than the budget** — one in-flight LLM call can overrun by its own
  clamped timeout. That is expected; the budget bounds the loop, not a syscall already in flight.

## Anti-patterns

- **Do not read a green in-loop compile as success.** `T0.1` exists because that lie shipped
  once. The verifier is the only authority. This applies to the fast path too: a mechanical
  port that compiles but SIGSEGVs is not a port, so `_hipify_source` output is only accepted
  when the binary also *runs*.
- **Do not widen the fast path to kernels using `__shfl`/`warpSize`.** `_needs_wavefront_semantics`
  skips them on purpose. A regex can rename the symbols; it cannot re-derive lane arithmetic,
  so the result compiles and crashes — and you paid a hipcc compile to learn what the source
  already said.
- **Do not raise `max_iterations` to fix a stagnating run.** Iterations are not the constraint;
  wall time is. A run that stagnates at iteration 3 will stagnate at iteration 10, for ~80s of
  DeepSeek each.
- **Do not disable the wall-clock budget** (`max_seconds=0`) outside a debugging session. It is
  the difference between a 3-minute demo and a run that needs `Ctrl+C`.
- **Do not commit `ported_kernels/`, `runs/`, or `data/pattern_memory.db`.** All are gitignored;
  `data/prompt_changelog.json` is the one tracked file under `data/`.
- **Do not add `shell=True` to a subprocess call.** `verifier.py` resolves binaries with
  list-form `subprocess.run` and `shutil.which`. See `bump-prompt-version` for the prompt rules.
