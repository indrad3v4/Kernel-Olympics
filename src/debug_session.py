"""Deterministic Debug Mode — every pipeline decision, persisted and replayable.

The problem this solves
-----------------------
A translation failure used to be reconstructible only from console scrollback.
The raw model response that produced a rejected port was never written down;
the extraction strategy that chose one code block over another was never
recorded; the exact hipcc invocation lived in a subprocess argv that no longer
exists. Diagnosing a failure meant re-running the pipeline — three LLM calls
and ~3 minutes — to see a different failure.

Debug Mode persists every intermediate artifact of every stage into a single
session directory, so a failed translation can be fully reconstructed offline
from that directory alone.

Contract
--------
* **Optional.** Off by default. ``DebugSession.disabled()`` is a null object
  whose every method is a cheap no-op returning ``None``; call sites never
  branch on ``if debug:``.
* **Non-destructive, append-only.** No artifact is ever overwritten. Every
  generation, patch and compile gets its own file, numbered by a monotonic
  sequence counter. Traces are JSONL, opened in append mode.
* **Deterministic.** Filenames come from the sequence counter, never from a
  clock, so two runs of the same pipeline produce the same *tree*. Wall-clock
  times are recorded as *content*, and the clock is injectable for tests.
* **Machine- and human-readable.** Every artifact has a JSON form; ``finalize()``
  additionally renders ``summary.md`` for a human.
* **Provider-independent.** Nothing here knows DeepSeek from Kimi from GLM. A
  stage is a string; a model is a string.
* **Never fatal.** A debug sink that raises would turn an observable failure
  into an unobservable one. Every write is guarded; failures are recorded in
  ``errors.jsonl`` and the pipeline continues.

Layout
------
``debug/session_<timestamp>_<kernel>/``::

    manifest.jsonl        append-only index: one line per artifact written
    errors.jsonl          failures of the debug sink itself
    state_trace.jsonl     every state-machine transition
    timeline.jsonl        every retry / pipeline event
    01_input/             CUDA source, classifier, patterns, preprocessing
    02_planning/          raw planner responses, parsed plans, validation
    03_translation/       every generation, raw and extracted, never overwritten
    04_extraction/        extraction reports (strategy, confidence, discarded)
    05_lexical/           lexical validation reports
    06_structural/        structural validation reports
    07_symbols/           CUDA vs HIP symbol tables and diffs
    08_static_analysis/   pre-compile findings
    09_compiler/          exact command, env, version, full stdout/stderr
    10_evaluation/        raw + parsed evaluator (GLM) responses
    11_patches/           every repair iteration: before, after, unified diff
    12_failure/           failure snapshot package
    metrics.json          per-stage timings, token usage, retry counts
    summary.md            human-readable post-mortem
"""

from __future__ import annotations

import difflib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Configuration ───────────────────────────────────────────────────────────

#: Environment variables that turn Debug Mode on. Any truthy value works.
ENV_FLAGS = ("KERNEL_OLYMPICS_DEBUG", "KERNEL_DEBUG_MODE")

#: Where sessions land unless overridden.
ENV_DEBUG_DIR = "KERNEL_OLYMPICS_DEBUG_DIR"

DEFAULT_ROOT = "debug"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}

# Stage directory names. Ordered by their position in the pipeline so an `ls`
# reads like the execution itself.
STAGE_INPUT = "01_input"
STAGE_PLANNING = "02_planning"
STAGE_TRANSLATION = "03_translation"
STAGE_EXTRACTION = "04_extraction"
STAGE_LEXICAL = "05_lexical"
STAGE_STRUCTURAL = "06_structural"
STAGE_SYMBOLS = "07_symbols"
STAGE_STATIC = "08_static_analysis"
STAGE_COMPILER = "09_compiler"
STAGE_EVALUATION = "10_evaluation"
STAGE_PATCHES = "11_patches"
STAGE_FAILURE = "12_failure"

_ALL_STAGES = (
    STAGE_INPUT, STAGE_PLANNING, STAGE_TRANSLATION, STAGE_EXTRACTION,
    STAGE_LEXICAL, STAGE_STRUCTURAL, STAGE_SYMBOLS, STAGE_STATIC,
    STAGE_COMPILER, STAGE_EVALUATION, STAGE_PATCHES, STAGE_FAILURE,
)


def debug_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve whether Debug Mode is on.

    An explicit ``True``/``False`` (from a CLI flag) always wins. Otherwise the
    environment decides. Absent both, Debug Mode is off — the pipeline must pay
    nothing for a feature nobody asked for.
    """
    if explicit is not None:
        return bool(explicit)
    for var in ENV_FLAGS:
        val = os.environ.get(var, "").strip().lower()
        if val in _TRUTHY:
            return True
    return False


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of *obj* into something ``json.dumps`` accepts.

    Dataclasses (``ExtractionResult``, ``LexicalResult``, ``ValidationResult``,
    ``Finding``) become dicts; sets become sorted lists; anything else that
    resists becomes its ``repr``. Never raises — a debug dump of a weird object
    is worth more than an exception.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _jsonable(asdict(obj))
        except Exception:
            return {k: _jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return repr(obj)


def _slug(text: str, limit: int = 48) -> str:
    """A filesystem-safe, deterministic slug. Empty input yields ``unnamed``."""
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip("_")
    return (cleaned[:limit] or "unnamed")


def _dur(ms: float) -> str:
    """Render a duration in a unit a reader can act on.

    A 0.4ms parser stage printed as ``0.00s`` tells nobody anything, and the
    parser stages are precisely the ones a reader is checking are cheap.
    """
    if ms < 1:
        return f"{ms * 1000:.0f}µs"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


_NAMESPACE_DEF = re.compile(r'^[ \t]*namespace[ \t]+([A-Za-z_]\w*)[ \t]*\{', re.MULTILINE)
_USING_NAMESPACE = re.compile(r'^[ \t]*using[ \t]+namespace[ \t]+([\w:]+)[ \t]*;', re.MULTILINE)


def _namespace_report(cuda_source: str, hip_source: str) -> Dict:
    """Which namespaces the original declared, and whether the port kept them.

    A port that silently drops ``namespace cg = cooperative_groups;`` or moves a
    kernel out of its enclosing namespace produces link errors far from the
    cause. Reported, never gated: a namespace can legitimately disappear when
    the construct that needed it is gone.
    """
    def _ns(text: str) -> Dict[str, List[str]]:
        return {
            "declared": sorted(set(_NAMESPACE_DEF.findall(text))),
            "using": sorted(set(_USING_NAMESPACE.findall(text))),
        }

    src, prt = _ns(cuda_source), _ns(hip_source)
    # Namespace *brace* balance is deliberately absent: the structural validator
    # already balances every brace in the file, and a second, weaker check here
    # would either duplicate it or contradict it.
    return {
        "original": src,
        "generated": prt,
        "dropped_declarations": sorted(set(src["declared"]) - set(prt["declared"])),
        "added_declarations": sorted(set(prt["declared"]) - set(src["declared"])),
        "dropped_using": sorted(set(src["using"]) - set(prt["using"])),
        "preserved": set(src["declared"]) <= set(prt["declared"]),
    }


def discarded_text(raw: str, code: str) -> str:
    """Everything in *raw* that extraction did NOT keep as *code*.

    This is the model's reasoning, its markdown fences, and its closing
    commentary — the text a failed port is usually explained by, and the text
    that is otherwise thrown away the instant the extractor returns.

    When *code* is not a literal substring of *raw* (the extractor may have
    sanitized fences or a truncation marker out of it), we cannot subtract the
    two reliably, so the whole response is reported as discarded-adjacent and
    labeled as such. Over-reporting here is safe; silently reporting ``""``
    would be a lie.
    """
    if not raw:
        return ""
    if not code or not code.strip():
        return raw
    idx = raw.find(code)
    if idx < 0:
        # Fall back to the first and last lines of the extract as anchors.
        lines = [l for l in code.splitlines() if l.strip()]
        if lines:
            start = raw.find(lines[0])
            end = raw.rfind(lines[-1])
            if start >= 0 and end > start:
                return (raw[:start] + "\n<<< extracted code omitted >>>\n"
                        + raw[end + len(lines[-1]):]).strip()
        return raw
    prefix, suffix = raw[:idx], raw[idx + len(code):]
    if not prefix.strip() and not suffix.strip():
        return ""
    return (prefix + "\n<<< extracted code omitted >>>\n" + suffix).strip()


# ── Records ─────────────────────────────────────────────────────────────────

@dataclass
class StageTiming:
    """Accumulated wall time and call count for one named stage."""
    name: str
    calls: int = 0
    total_ms: float = 0.0
    failures: int = 0

    def to_dict(self) -> Dict:
        return {
            "stage": self.name,
            "calls": self.calls,
            "total_ms": round(self.total_ms, 2),
            "mean_ms": round(self.total_ms / self.calls, 2) if self.calls else 0.0,
            "failures": self.failures,
        }


@dataclass
class TokenUsage:
    """Token and cost accounting for one model."""
    model: str
    calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "tokens": self.tokens,
            "cost": round(self.cost, 6),
            "latency_ms": round(self.latency_ms, 2),
            "mean_latency_ms": round(self.latency_ms / self.calls, 2) if self.calls else 0.0,
        }


# ── The null object ─────────────────────────────────────────────────────────

class _NullSession:
    """Debug Mode, off.

    Every public method of :class:`DebugSession` exists here as a no-op so that
    instrumented call sites need no guards. ``enabled`` is ``False`` for the
    rare caller that genuinely must know (e.g. to skip building an expensive
    artifact it would only pass to a sink that discards it).

    ``__getattr__`` covers methods added to DebugSession later: an unknown
    attribute resolves to a callable that returns ``None``. That is deliberate
    — forgetting to stub a new method here must not crash a production run with
    Debug Mode off.
    """

    enabled = False
    dir: Optional[Path] = None
    session_id = ""

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, _name: str) -> Callable[..., None]:
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None
        return _noop

    @contextmanager
    def stage(self, *_a: Any, **_kw: Any):
        yield self

    # The methods below return something other than None on a live session.
    # They are stubbed explicitly — not left to __getattr__ — so a caller that
    # writes `gen = debug.log_generation(...)` without guarding on `enabled`
    # gets a value of the declared type rather than a None that only explodes
    # three frames later, in production, with Debug Mode off.

    def next_generation_index(self) -> int:
        return 0

    def log_generation(self, *_a: Any, **_kw: Any) -> int:
        return 0

    def log_compile(self, *_a: Any, **_kw: Any) -> int:
        return 0

    def log_patch(self, *_a: Any, **_kw: Any) -> None:
        return None

    def log_symbols(self, *_a: Any, **_kw: Any) -> None:
        return None

    def log_static_analysis(self, *_a: Any, **_kw: Any) -> None:
        return None

    def metrics(self, *_a: Any, **_kw: Any) -> Dict:
        return {}

    def snapshot_failure(self, *_a: Any, **_kw: Any) -> None:
        return None

    def finalize(self, *_a: Any, **_kw: Any) -> None:
        return None


# ── The real thing ──────────────────────────────────────────────────────────

class DebugSession:
    """An append-only, per-translation debug session rooted at one directory.

    Construct via :meth:`create`, which returns a :class:`_NullSession` when
    Debug Mode is off — so the caller writes one line and never branches:

    .. code-block:: python

        dbg = DebugSession.create(kernel_name, enabled=args.debug)
        dbg.log_input(source, classifier_results=..., patterns=...)

    Thread-safety: a session is owned by the pipeline thread that created it.
    The append-only writes are individually atomic (one ``open``/``write``/
    ``close`` per record), so a stray write from a progress thread cannot
    corrupt an earlier record — but ordering across threads is not guaranteed.
    """

    def __init__(self, kernel_name: str, root: str | Path = DEFAULT_ROOT,
                 session_id: Optional[str] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.enabled = True
        self.kernel_name = kernel_name or "unknown_kernel"
        self._clock = clock or time.time
        self._mono = time.perf_counter

        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(self._clock()))
        self.session_id = session_id or f"session_{stamp}_{_slug(self.kernel_name)}"

        base = Path(root) / self.session_id
        # Non-destructive: an existing session directory is never reused or
        # cleared. A collision (same second, same kernel) gets its own suffix.
        candidate, n = base, 1
        while candidate.exists():
            candidate = base.parent / f"{base.name}__{n}"
            n += 1
        self.dir = candidate
        self.dir.mkdir(parents=True, exist_ok=True)
        for stage in _ALL_STAGES:
            (self.dir / stage).mkdir(exist_ok=True)

        self._seq = 0                    # monotonic artifact counter
        self._generation = 0             # monotonic generation counter
        self._patch = 0                  # monotonic patch counter
        self._compile = 0                # monotonic compile counter
        self._t0 = self._mono()
        self._started_at = self._clock()

        self._stages: Dict[str, StageTiming] = {}
        self._tokens: Dict[str, TokenUsage] = {}
        self._counters: Dict[str, int] = {}
        self._state = "INIT"
        self._transitions: List[Dict] = []
        self._events: List[Dict] = []
        self._finalized = False

        self._append("manifest.jsonl", {
            "event": "session_start",
            "session_id": self.session_id,
            "kernel": self.kernel_name,
            "started_at": self._iso(self._started_at),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "argv": list(sys.argv),
        })

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def create(cls, kernel_name: str, enabled: Optional[bool] = None,
               root: Optional[str | Path] = None, **kwargs: Any):
        """Return a live session when Debug Mode is on, else a null object."""
        if not debug_enabled(enabled):
            return cls.disabled()
        root = root or os.environ.get(ENV_DEBUG_DIR) or DEFAULT_ROOT
        try:
            return cls(kernel_name, root=root, **kwargs)
        except OSError:
            # An unwritable debug root must not take down the pipeline. Degrade
            # to off, loudly — silence here would look like Debug Mode working.
            print(f"║ ⚠ Debug Mode disabled: cannot create session under {root!r}")
            return cls.disabled()

    @staticmethod
    def disabled() -> "_NullSession":
        """The shared no-op session used whenever Debug Mode is off."""
        return _NullSession()

    # ── Primitives ──────────────────────────────────────────────────────────

    def _iso(self, epoch: Optional[float] = None) -> str:
        e = self._clock() if epoch is None else epoch
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e))

    def elapsed_ms(self) -> float:
        """Milliseconds since the session opened."""
        return (self._mono() - self._t0) * 1000.0

    def _record_error(self, where: str, exc: BaseException) -> None:
        """A failure of the debug sink itself. Recorded, never raised."""
        try:
            path = self.dir / "errors.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "at": self._iso(),
                    "where": where,
                    "error": f"{type(exc).__name__}: {exc}",
                }, sort_keys=True) + "\n")
        except Exception:
            pass  # the sink for sink failures cannot itself fail loudly

    def _append(self, relpath: str, record: Dict) -> None:
        """Append one JSON record to a JSONL file. Never overwrites."""
        try:
            record = dict(record)
            record.setdefault("at", self._iso())
            record.setdefault("elapsed_ms", round(self.elapsed_ms(), 2))
            path = self.dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_jsonable(record), sort_keys=True,
                                    ensure_ascii=False) + "\n")
        except Exception as exc:
            self._record_error(f"_append:{relpath}", exc)

    def _unique(self, stage: str, name: str) -> Path:
        """A never-before-used path inside *stage*, prefixed by a sequence number.

        The prefix is what makes the directory append-only in practice: a second
        artifact with the same logical name lands beside the first, not on top
        of it. Callers therefore never need to ask "has this been written?".
        """
        self._seq += 1
        path = self.dir / stage / f"{self._seq:04d}_{_slug(name, 64)}"
        # Defensive: a caller-supplied name could still collide after slugging.
        if path.exists():
            path = path.with_name(f"{path.name}__{self._seq}")
        return path

    def write_text(self, stage: str, name: str, text: str) -> Optional[Path]:
        """Persist *text* as a new artifact. Returns its path."""
        try:
            path = self._unique(stage, name)
            path.write_text(text if text is not None else "", encoding="utf-8",
                            errors="replace")
            self._append("manifest.jsonl", {
                "event": "artifact",
                "stage": stage,
                "name": name,
                "path": str(path.relative_to(self.dir)),
                "bytes": len(text or ""),
                "kind": "text",
            })
            return path
        except Exception as exc:
            self._record_error(f"write_text:{stage}/{name}", exc)
            return None

    def write_json(self, stage: str, name: str, payload: Any) -> Optional[Path]:
        """Persist *payload* as a new ``.json`` artifact. Returns its path."""
        try:
            if not str(name).endswith(".json"):
                name = f"{name}.json"
            path = self._unique(stage, name)
            body = json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                              ensure_ascii=False)
            path.write_text(body, encoding="utf-8")
            self._append("manifest.jsonl", {
                "event": "artifact",
                "stage": stage,
                "name": name,
                "path": str(path.relative_to(self.dir)),
                "bytes": len(body),
                "kind": "json",
            })
            return path
        except Exception as exc:
            self._record_error(f"write_json:{stage}/{name}", exc)
            return None

    def count(self, key: str, n: int = 1) -> None:
        """Bump a named counter (retries, rejects, replans …)."""
        self._counters[key] = self._counters.get(key, 0) + n

    # ── Timing ──────────────────────────────────────────────────────────────

    @contextmanager
    def stage(self, name: str):
        """Time a stage. Records duration whether or not the body raises.

        An exception inside the body increments the stage's ``failures`` and is
        re-raised untouched — Debug Mode observes, it does not swallow.
        """
        t0 = self._mono()
        timing = self._stages.setdefault(name, StageTiming(name))
        timing.calls += 1
        try:
            yield self
        except BaseException:
            timing.failures += 1
            raise
        finally:
            dt = (self._mono() - t0) * 1000.0
            timing.total_ms += dt
            self._append("timeline.jsonl", {
                "event": "stage",
                "stage": name,
                "duration_ms": round(dt, 2),
            })

    def record_llm_call(self, model: str, tokens: int = 0, cost: float = 0.0,
                        latency_ms: float = 0.0, success: bool = True,
                        stage: str = "", raw_response: Optional[str] = None,
                        prompt: Optional[str] = None,
                        system_prompt: Optional[str] = None,
                        endpoint: str = "", error: str = "",
                        finish_reason: str = "") -> None:
        """Record one model call, complete and untruncated.

        This is the single hook that satisfies "no intermediate response should
        ever be lost": every provider call routes through it, so the raw text
        that produced a rejected port is on disk before anything parses it.

        *stage* names the destination directory (planning, translation,
        evaluation). An unknown stage lands in translation, never nowhere.
        """
        usage = self._tokens.setdefault(model, TokenUsage(model))
        usage.calls += 1
        usage.tokens += int(tokens or 0)
        usage.cost += float(cost or 0.0)
        usage.latency_ms += float(latency_ms or 0.0)

        target = stage if stage in _ALL_STAGES else STAGE_TRANSLATION
        label = f"{model}_{'ok' if success else 'fail'}"

        if prompt is not None:
            self.write_text(target, f"{label}_prompt.txt", prompt)
        if system_prompt:
            self.write_text(target, f"{label}_system_prompt.txt", system_prompt)
        if raw_response is not None:
            # Untruncated, on purpose. The whole point of this module.
            self.write_text(target, f"{label}_raw_response.txt", raw_response)

        self._append("timeline.jsonl", {
            "event": "llm_call",
            "model": model,
            "stage": target,
            "success": success,
            "tokens": int(tokens or 0),
            "cost": round(float(cost or 0.0), 6),
            "latency_ms": round(float(latency_ms or 0.0), 2),
            "endpoint": endpoint,
            "finish_reason": finish_reason,
            "error": error,
            "response_chars": len(raw_response) if raw_response is not None else 0,
        })

    # ── State machine ───────────────────────────────────────────────────────

    def transition(self, to_state: str, reason: str = "",
                   validation_result: Any = None, **extra: Any) -> None:
        """Record a state-machine transition: previous → next, why, how long.

        The elapsed time attached is the time spent *in the previous state*,
        which is the number a reader actually wants — "PLAN_GENERATED took 38s",
        not "we are 38s into the run".
        """
        now = self._mono()
        last = self._transitions[-1]["_mono"] if self._transitions else self._t0
        record = {
            "index": len(self._transitions),
            "previous_state": self._state,
            "next_state": to_state,
            "reason": reason,
            "validation_result": _jsonable(validation_result),
            "elapsed_in_previous_ms": round((now - last) * 1000.0, 2),
            "since_start_ms": round((now - self._t0) * 1000.0, 2),
            "at": self._iso(),
        }
        record.update({k: _jsonable(v) for k, v in extra.items()})
        self._transitions.append({**record, "_mono": now})
        self._append("state_trace.jsonl", record)
        self._state = to_state

    def event(self, kind: str, reason: str = "", iteration: Optional[int] = None,
              **extra: Any) -> None:
        """Record a retry-history event (generation, reject, compile, patch …)."""
        record = {
            "event": kind,
            "reason": reason,
            "iteration": iteration,
            "state": self._state,
        }
        record.update({k: _jsonable(v) for k, v in extra.items()})
        self._events.append({**record, "at": self._iso(),
                             "elapsed_ms": round(self.elapsed_ms(), 2)})
        self._append("timeline.jsonl", record)

    # ── Stage 1: input ──────────────────────────────────────────────────────

    def log_input(self, cuda_source: str, classifier_results: Any = None,
                  patterns: Any = None, preprocessed_source: str = "",
                  preprocessing_changelog: Any = None, **meta: Any) -> None:
        self.write_text(STAGE_INPUT, "original.cu", cuda_source)
        if preprocessed_source:
            self.write_text(STAGE_INPUT, "preprocessed_hipified.hip.cpp",
                            preprocessed_source)
        self.write_json(STAGE_INPUT, "input_report", {
            "kernel_name": self.kernel_name,
            "source_bytes": len(cuda_source or ""),
            "source_lines": len((cuda_source or "").splitlines()),
            "classifier_results": classifier_results,
            "detected_patterns": patterns,
            "preprocessing": {
                "applied": bool(preprocessed_source),
                "changelog": preprocessing_changelog or [],
                "transforms": len(preprocessing_changelog or []),
            },
            **meta,
        })

    # ── Stage 2: planning ───────────────────────────────────────────────────

    def log_planning(self, raw_response: str = "", extracted_plan: str = "",
                     parsed_plan: Any = None, validation: Any = None,
                     discarded: str = "", tokens: int = 0,
                     latency_ms: float = 0.0, attempt: int = 1,
                     model: str = "", success: bool = True, **meta: Any) -> None:
        tag = f"plan_attempt{attempt}"
        if raw_response:
            self.write_text(STAGE_PLANNING, f"{tag}_raw.txt", raw_response)
        if extracted_plan:
            self.write_text(STAGE_PLANNING, f"{tag}_extracted.txt", extracted_plan)
        if discarded:
            self.write_text(STAGE_PLANNING, f"{tag}_discarded.txt", discarded)
        self.write_json(STAGE_PLANNING, f"{tag}_report", {
            "attempt": attempt,
            "model": model,
            "success": success,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "raw_chars": len(raw_response or ""),
            "extracted_chars": len(extracted_plan or ""),
            "discarded_chars": len(discarded or ""),
            "parsed_plan": parsed_plan,
            "validation": validation,
            **meta,
        })
        self.event("plan_generated", reason=model or "planner",
                   attempt=attempt, success=success, tokens=tokens,
                   latency_ms=round(latency_ms, 2))

    # ── Stage 3: translation ────────────────────────────────────────────────

    def next_generation_index(self) -> int:
        """Reserve and return the next generation number (1-based, monotonic)."""
        self._generation += 1
        return self._generation

    def log_generation(self, raw_response: str, extracted_code: str = "",
                       discarded: str = "", iteration: int = 0,
                       generation: Optional[int] = None, model: str = "",
                       tokens: int = 0, latency_ms: float = 0.0,
                       chunks: Optional[List[str]] = None,
                       success: bool = True, **meta: Any) -> int:
        """Persist one complete generation. Never overwrites a previous one.

        Returns the generation index, which the caller should thread into the
        extraction / lexical / structural reports so a reader can tie a reject
        back to the exact text that caused it.
        """
        gen = generation if generation is not None else self.next_generation_index()
        tag = f"gen{gen:03d}_iter{iteration}"
        self.write_text(STAGE_TRANSLATION, f"{tag}_raw_response.txt", raw_response or "")
        if extracted_code:
            self.write_text(STAGE_TRANSLATION, f"{tag}_extracted.hip.cpp", extracted_code)
        if discarded:
            self.write_text(STAGE_TRANSLATION, f"{tag}_discarded_reasoning.txt", discarded)
        if chunks:
            self.write_json(STAGE_TRANSLATION, f"{tag}_stream_chunks", {
                "chunk_count": len(chunks),
                "chunks": list(chunks),
            })
        self.write_json(STAGE_TRANSLATION, f"{tag}_report", {
            "generation": gen,
            "iteration": iteration,
            "model": model,
            "success": success,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "raw_chars": len(raw_response or ""),
            "extracted_chars": len(extracted_code or ""),
            "discarded_chars": len(discarded or ""),
            "streamed": bool(chunks),
            **meta,
        })
        self.event("generation", reason=model or "coder", iteration=iteration,
                   generation=gen, success=success, tokens=tokens)
        return gen

    # ── Stage 4: extraction ─────────────────────────────────────────────────

    def log_extraction(self, extraction: Any, generation: Optional[int] = None,
                       iteration: int = 0, **meta: Any) -> None:
        """Persist a machine-readable extraction report.

        Accepts an ``ExtractionResult`` (dataclass) or any object exposing the
        same fields; unknown shapes degrade to their ``repr`` rather than being
        dropped.
        """
        get = lambda k, d=None: getattr(extraction, k, d) if extraction is not None else d
        code = get("code") or ""
        response_len = get("response_length", 0) or 0
        code_len = get("code_length", len(code)) or len(code)
        discarded_len = get("discarded_length", max(response_len - code_len, 0))
        diagnostics = list(get("diagnostics", []) or [])
        strategy = get("strategy", "unknown")

        # Confidence is derived, not invented: the fraction of the response that
        # survived extraction, damped by whether a strategy identified a block
        # at all. Stated as a formula in the artifact so nobody mistakes it for
        # a model-reported score.
        ratio = (code_len / response_len) if response_len else 0.0
        confidence = 0.0 if not code.strip() else round(min(1.0, 0.5 + 0.5 * ratio), 4)

        # A fenced block was removed exactly when the fence strategy fired, or
        # when a diagnostic names one. Both are facts the extractor reports; do
        # not re-derive them by re-scanning text the extractor already consumed.
        diag_text = " ".join(str(d).lower() for d in diagnostics)
        markdown_removed = "fence" in str(strategy).lower() or "fence" in diag_text

        self.write_json(STAGE_EXTRACTION, f"extraction_gen{(generation or 0):03d}", {
            "generation": generation,
            "iteration": iteration,
            "strategy_used": strategy,
            "code_block_detected": bool(code.strip()),
            "candidates_considered": get("candidates_considered", 0),
            "response_length": response_len,
            "code_length": code_len,
            "discarded_length": discarded_len,
            "markdown_removed": markdown_removed,
            "reasoning_removed": bool(discarded_len),
            "malformed_blocks": [str(d) for d in diagnostics
                                 if "malformed" in str(d).lower()
                                 or "unterminated" in str(d).lower()],
            "parser_confidence": confidence,
            "confidence_formula": "0 if no code else min(1, 0.5 + 0.5*code_len/response_len)",
            "parser_decisions": [str(d) for d in diagnostics],
            "parser_warnings": [str(d) for d in diagnostics if "warn" in str(d).lower()],
            "extraction_failed": not bool(code.strip()),
            **meta,
        })

    # ── Stage 5: lexical ────────────────────────────────────────────────────

    def log_lexical(self, lexical: Any, generation: Optional[int] = None,
                    iteration: int = 0, code: str = "", **meta: Any) -> None:
        if lexical is None:
            return
        get = lambda k, d=None: getattr(lexical, k, d)
        errors = list(get("errors", []) or [])
        warnings = list(get("warnings", []) or [])
        samples = list(get("prose_line_samples", []) or [])
        joined = " ".join(errors + warnings).lower()

        self.write_json(STAGE_LEXICAL, f"lexical_gen{(generation or 0):03d}", {
            "generation": generation,
            "iteration": iteration,
            "pass": bool(get("ok", False)),
            "decision": "PASS" if get("ok", False) else "REJECT",
            "reason": get("reason", lambda: "")() if callable(get("reason")) else get("reason", ""),
            "errors": errors,
            "warnings": warnings,
            "rejected_phrases": samples,
            "rejected_tokens": sorted({w.strip(".,:;") for s in samples
                                       for w in str(s).split()[:4] if w.strip(".,:;")}),
            "detected_reasoning": any(k in joined for k in
                                      ("reason", "prose", "explanation", "narrat")),
            "markdown_detected": "markdown" in joined or "fence" in joined,
            "ellipsis_detected": "..." in (code or "") or "ellipsis" in joined,
            "placeholder_detected": any(k in joined for k in
                                        ("placeholder", "truncat", "rest of")),
            "stats": get("stats", {}) or {},
            **meta,
        })

    # ── Stage 6: structural ─────────────────────────────────────────────────

    def log_structural(self, structural: Any, generation: Optional[int] = None,
                       iteration: int = 0, cuda_source: str = "",
                       hip_source: str = "", **meta: Any) -> None:
        if structural is None:
            return
        get = lambda k, d=None: getattr(structural, k, d)
        errors = list(get("errors", []) or [])
        joined = " ".join(errors).lower()

        payload = {
            "generation": generation,
            "iteration": iteration,
            "pass": bool(get("ok", False)),
            "decision": "PASS" if get("ok", False) else "REJECT",
            "structural_score": get("score", 0.0),
            "errors": errors,
            "warnings": list(get("warnings", []) or []),
            "missing_symbols": list(get("missing_symbols", []) or []),
            "brace_validation": "unbalanced braces" not in joined,
            "paren_validation": "unbalanced parentheses" not in joined,
            "truncation_marker": "truncation marker" in joined,
            "duplicate_definitions": "duplicate definitions" in joined,
            **meta,
        }
        if cuda_source and hip_source:
            try:
                from verification.symbols import extract_symbols
                src, prt = extract_symbols(cuda_source), extract_symbols(hip_source)
                payload["symbol_counts"] = {
                    "original": src.to_dict()["counts"],
                    "generated": prt.to_dict()["counts"],
                }
                payload["kernel_preservation"] = sorted(
                    set(src.kernels) - set(prt.kernels)) == []
                payload["helper_preservation"] = sorted(
                    set(src.helpers) - set(prt.helpers)) == []
                payload["function_preservation"] = sorted(
                    set(src.functions) - set(prt.functions)) == []
                payload["include_validation"] = {
                    "hip_runtime_present": any("hip_runtime" in i for i in prt.includes),
                    "cuda_headers_remaining": [i for i in prt.includes
                                               if "cuda" in i.lower() or i.endswith('.cuh"')],
                }
                payload["namespace_validation"] = _namespace_report(cuda_source, hip_source)
            except Exception as exc:
                self._record_error("log_structural:symbols", exc)
        self.write_json(STAGE_STRUCTURAL, f"structural_gen{(generation or 0):03d}", payload)

    # ── Stage 7: symbols ────────────────────────────────────────────────────

    def log_symbols(self, cuda_source: str, hip_source: str,
                    generation: Optional[int] = None, iteration: int = 0) -> Optional[Dict]:
        """Persist the original/generated symbol tables and their diff."""
        try:
            from verification.symbols import diff_symbols
            report = diff_symbols(cuda_source or "", hip_source or "")
        except Exception as exc:
            self._record_error("log_symbols", exc)
            return None
        report["generation"] = generation
        report["iteration"] = iteration
        self.write_json(STAGE_SYMBOLS, f"symbol_diff_gen{(generation or 0):03d}", report)
        return report

    # ── Stage 8: static analysis ────────────────────────────────────────────

    def log_static_analysis(self, hip_source: str, generation: Optional[int] = None,
                            iteration: int = 0) -> Optional[Dict]:
        """Persist pre-compile findings. Advisory: never gates anything."""
        try:
            from verification.static_analysis import analyze
            report = analyze(hip_source or "").to_dict()
        except Exception as exc:
            self._record_error("log_static_analysis", exc)
            return None
        report["generation"] = generation
        report["iteration"] = iteration
        self.write_json(STAGE_STATIC, f"static_analysis_gen{(generation or 0):03d}", report)
        return report

    # ── Stage 9: compiler ───────────────────────────────────────────────────

    def log_compile(self, command: List[str], stdout: str = "", stderr: str = "",
                    returncode: Optional[int] = None, cwd: str = "",
                    env: Optional[Dict[str, str]] = None,
                    compiler_version: str = "", source_path: str = "",
                    source_text: str = "", diagnostics: Any = None,
                    artifacts: Optional[List[str]] = None,
                    iteration: int = 0, **meta: Any) -> int:
        """Persist one compiler invocation, complete and untruncated.

        ``stdout``/``stderr`` are written verbatim to their own files. The JSON
        report carries the exact argv, the resolved environment subset that
        affects a HIP build, the compiler version, and the diagnostics the
        caller parsed out — but never a truncated copy of the output, because a
        truncated compiler log is the thing this module exists to abolish.
        """
        self._compile += 1
        n = self._compile
        tag = f"compile{n:03d}_iter{iteration}"

        self.write_text(STAGE_COMPILER, f"{tag}_stdout.txt", stdout or "")
        self.write_text(STAGE_COMPILER, f"{tag}_stderr.txt", stderr or "")
        if source_text:
            self.write_text(STAGE_COMPILER, f"{tag}_compiled_source.hip.cpp", source_text)

        # Only the variables that change a HIP build's behavior. A full environ
        # dump would leak API keys into an artifact people paste into issues.
        interesting = ("PATH", "HIP_PATH", "ROCM_PATH", "HIP_PLATFORM",
                       "HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND",
                       "AMD_OFFLOAD_ARCH", "HSA_OVERRIDE_GFX_VERSION",
                       "LD_LIBRARY_PATH", "CXX", "CC", "VERIFIER_BUILD_DIR")
        source_env = env if env is not None else os.environ
        captured = {k: source_env[k] for k in interesting if k in source_env}

        self.write_json(STAGE_COMPILER, f"{tag}_report", {
            "compile_index": n,
            "iteration": iteration,
            "command": list(command or []),
            "command_line": " ".join(str(c) for c in (command or [])),
            "returncode": returncode,
            "success": returncode == 0,
            "cwd": cwd or os.getcwd(),
            "compiler_version": compiler_version,
            "environment": captured,
            "environment_note": "filtered to HIP-relevant variables; secrets excluded",
            "source_path": source_path,
            "stdout_bytes": len(stdout or ""),
            "stderr_bytes": len(stderr or ""),
            "diagnostics": diagnostics,
            "object_files": artifacts or [],
            **meta,
        })
        self.event("compile", reason="hipcc", iteration=iteration,
                   returncode=returncode, success=returncode == 0)
        return n

    # ── Stage 10: evaluation ────────────────────────────────────────────────

    def log_evaluation(self, raw_response: str = "", parsed: Any = None,
                       model: str = "", iteration: int = 0, mode: str = "evaluate",
                       root_cause: Any = None, recommended_fixes: Any = None,
                       confidence: Any = None, parse_strategy: str = "",
                       **meta: Any) -> None:
        tag = f"eval_iter{iteration}_{_slug(mode, 24)}"
        if raw_response:
            self.write_text(STAGE_EVALUATION, f"{tag}_raw.txt", raw_response)
        self.write_json(STAGE_EVALUATION, f"{tag}_report", {
            "iteration": iteration,
            "mode": mode,
            "model": model,
            "parsed": parsed,
            "parse_succeeded": parsed is not None,
            "parse_strategy": parse_strategy,
            "root_cause_analysis": root_cause,
            "recommended_fixes": recommended_fixes,
            "confidence_score": confidence,
            "raw_chars": len(raw_response or ""),
            **meta,
        })

    # ── Stage 11: patches ───────────────────────────────────────────────────

    def log_patch(self, before: str, after: str, iteration: int = 0,
                  rationale: str = "", confidence: Any = None,
                  source_label: str = "refine", **meta: Any) -> Optional[Dict]:
        """Persist one repair iteration. Never overwrites a previous patch.

        The unified diff is computed here rather than taken from the caller, so
        "lines modified" is a fact about the two texts rather than a model's
        claim about them.
        """
        self._patch += 1
        n = self._patch
        tag = f"patch{n:03d}_iter{iteration}"
        before, after = before or "", after or ""

        diff_lines = list(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"before_iter{iteration}", tofile=f"after_iter{iteration}",
            lineterm="", n=3))
        diff_text = "\n".join(diff_lines)

        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        self.write_text(STAGE_PATCHES, f"{tag}_before.hip.cpp", before)
        self.write_text(STAGE_PATCHES, f"{tag}_after.hip.cpp", after)
        self.write_text(STAGE_PATCHES, f"{tag}.diff", diff_text)

        symbols_modified: List[str] = []
        try:
            from verification.symbols import extract_symbols
            b, a = extract_symbols(before), extract_symbols(after)
            symbols_modified = sorted(b.all_names() ^ a.all_names())
        except Exception as exc:
            self._record_error("log_patch:symbols", exc)

        report = {
            "patch_index": n,
            "iteration": iteration,
            "source": source_label,
            "lines_added": added,
            "lines_removed": removed,
            "lines_modified": added + removed,
            "unchanged": diff_text == "",
            "symbols_modified": symbols_modified,
            "rationale": rationale,
            "patch_confidence": confidence,
            "before_bytes": len(before),
            "after_bytes": len(after),
            **meta,
        }
        self.write_json(STAGE_PATCHES, f"{tag}_report", report)
        self.event("patch", reason=source_label, iteration=iteration,
                   lines_modified=added + removed)
        return report

    # ── Stage 12: failure snapshot ──────────────────────────────────────────

    def snapshot_failure(self, exc: Optional[BaseException] = None,
                         reason: str = "", context: Any = None) -> Optional[Path]:
        """Write a self-contained failure package.

        Everything needed to reproduce the failure offline is already on disk in
        the stage directories; this adds the one thing that is not — *why we
        stopped* — plus an index that points at the rest, so a reader opens one
        file rather than guessing which of forty to start with.
        """
        try:
            payload = {
                "session_id": self.session_id,
                "kernel": self.kernel_name,
                "reason": reason or (f"{type(exc).__name__}: {exc}" if exc else "unknown"),
                "exception_type": type(exc).__name__ if exc else None,
                "traceback": "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)) if exc else "",
                "final_state": self._state,
                "state_trace": [
                    {k: v for k, v in t.items() if k != "_mono"}
                    for t in self._transitions],
                "retry_history": list(self._events),
                "counters": dict(sorted(self._counters.items())),
                "stage_timings": [t.to_dict() for _, t in sorted(self._stages.items())],
                "token_usage": [u.to_dict() for _, u in sorted(self._tokens.items())],
                "context": context,
                "artifact_index": self._artifact_index(),
                "reproduce": (
                    "Every artifact referenced here is already inside this session "
                    "directory. No LLM call is required to re-read them: the raw "
                    "responses are in 02_planning/, 03_translation/ and 10_evaluation/, "
                    "the exact hipcc argv and full compiler output in 09_compiler/."
                ),
            }
            self.write_json(STAGE_FAILURE, "failure_snapshot", payload)
            self.event("failure_snapshot", reason=payload["reason"])
            path = self.dir / STAGE_FAILURE
            return path
        except Exception as sink_exc:
            self._record_error("snapshot_failure", sink_exc)
            return None

    def _artifact_index(self) -> List[Dict]:
        """Read back manifest.jsonl — the append-only record of every artifact."""
        entries: List[Dict] = []
        path = self.dir / "manifest.jsonl"
        if not path.exists():
            return entries
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "artifact":
                    entries.append({"stage": rec.get("stage"), "path": rec.get("path"),
                                    "bytes": rec.get("bytes")})
        except OSError as exc:
            self._record_error("_artifact_index", exc)
        return entries

    # ── Metrics + summary ───────────────────────────────────────────────────

    def metrics(self, result: Optional[Dict] = None) -> Dict:
        """The performance/accounting view of this session."""
        total_tokens = sum(u.tokens for u in self._tokens.values())
        total_cost = sum(u.cost for u in self._tokens.values())
        llm_ms = sum(u.latency_ms for u in self._tokens.values())
        return {
            "session_id": self.session_id,
            "kernel": self.kernel_name,
            "started_at": self._iso(self._started_at),
            "total_runtime_ms": round(self.elapsed_ms(), 2),
            "llm_latency_ms": round(llm_ms, 2),
            "per_stage": [t.to_dict() for _, t in sorted(self._stages.items())],
            "token_usage": [u.to_dict() for _, u in sorted(self._tokens.items())],
            "totals": {
                "llm_calls": sum(u.calls for u in self._tokens.values()),
                "tokens": total_tokens,
                "cost": round(total_cost, 6),
            },
            "counters": dict(sorted(self._counters.items())),
            "transitions": len(self._transitions),
            "events": len(self._events),
            "generations": self._generation,
            "patches": self._patch,
            "compiles": self._compile,
            "result": _jsonable(result) if result is not None else None,
        }

    def finalize(self, result: Optional[Dict] = None,
                 probable_root_cause: str = "",
                 recommended_action: str = "") -> Optional[Path]:
        """Write ``metrics.json`` and the human-readable ``summary.md``.

        Idempotent by refusal, not by overwrite: a second call appends a new
        summary rather than replacing the first, preserving the append-only
        guarantee even against a caller that finalizes twice.
        """
        try:
            metrics = self.metrics(result)
            if self._finalized:
                self.write_json(STAGE_FAILURE, "metrics_refinalized", metrics)
            else:
                (self.dir / "metrics.json").write_text(
                    json.dumps(_jsonable(metrics), indent=2, sort_keys=True,
                               ensure_ascii=False), encoding="utf-8")

            summary = self._render_summary(metrics, result or {},
                                           probable_root_cause, recommended_action)
            name = "summary.md" if not self._finalized else f"summary_{self._seq}.md"
            path = self.dir / name
            path.write_text(summary, encoding="utf-8")
            self._append("manifest.jsonl", {"event": "session_end",
                                            "summary": name,
                                            "final_state": self._state})
            self._finalized = True
            return path
        except Exception as exc:
            self._record_error("finalize", exc)
            return None

    # ── Summary rendering ───────────────────────────────────────────────────

    def _infer_root_cause(self, result: Dict) -> str:
        """A best-effort root cause, stated as an inference, never as a fact.

        Ordered most-specific first. Each branch names the artifact that proves
        it, so a reader can disagree with the inference by opening one file.

        The recorded *counters* are consulted before the abort reason, because an
        abort reason describes how the loop gave up, not why. "max_iterations_
        exhausted" is true of a run whose coder returned prose every single time,
        and it is the least useful sentence that could be written about it.
        """
        c = self._counters
        lexical, structural = c.get("lexical_rejects", 0), c.get("structural_rejects", 0)

        # The coder never returned source at all. Nothing downstream ever ran.
        if lexical and lexical >= self._generation > 0 and self._compile == 0:
            return (f"The coder returned reasoning/prose instead of source code on all "
                    f"{self._generation} generation(s); the lexical gate rejected each one "
                    f"and hipcc was never invoked. This is a prompting or model-selection "
                    f"failure, not a porting failure. The exact text is in "
                    f"`{STAGE_TRANSLATION}/` and the verdicts in `{STAGE_LEXICAL}/`.")

        # Structure broke every time, but the text was source code.
        if structural and structural >= self._generation > 0 and self._compile == 0:
            return (f"Every generation failed structural validation before hipcc ran "
                    f"(unbalanced braces, a truncation marker, or a dropped symbol). "
                    f"See `{STAGE_STRUCTURAL}/` — the compiler was never the problem.")

        if self._compile == 0 and self._generation > 0:
            return (f"No compile was ever attempted across {self._generation} generation(s): "
                    f"a pre-compile gate stopped each one. See `{STAGE_LEXICAL}/` and "
                    f"`{STAGE_STRUCTURAL}/`.")

        abort = (result or {}).get("abort_reason", "")
        mapping = {
            "harness_origin": (
                "Every compiler diagnostic pointed at harness/driver lines rather than "
                "the ported kernel — see `09_compiler/` diagnostics and the "
                "`error_origins` field. The port may be fine; the spec/harness is not."),
            "pipeline_timeout": (
                "The wall-clock budget expired before convergence. See `metrics.json` "
                "→ `per_stage` for where the time went."),
            "layer2_rejected": (
                "A refinement broke a build that previously compiled; the frozen base "
                "was restored. Compare the last two entries in `11_patches/`."),
            "hard_stagnation": (
                "The error set stopped shrinking across iterations even after re-planning "
                "— see `timeline.jsonl` for the per-iteration error counts."),
            "kimi_plateau": (
                "The coder returned an identical normalized error set on consecutive "
                "iterations; shim injection did not clear it."),
            "runtime_stagnation": (
                "The kernel compiled but crashed at runtime three times consecutively — "
                "typically wavefront64 semantics (shared-memory sizing, shuffle width). "
                "See `08_static_analysis/` for warp-32 warnings predicted before compile."),
            "max_iterations_exhausted": (
                "The iteration ceiling was reached without a passing compile."),
        }
        if abort in mapping:
            return mapping[abort]

        # No abort reason: infer from the last transition.
        last = self._transitions[-1]["next_state"] if self._transitions else self._state
        if "LEXICAL" in last:
            return ("The last decision was a lexical rejection — the model returned prose "
                    "or markdown where source was required. See `05_lexical/`.")
        if "STRUCTURAL" in last:
            return ("The last decision was a structural rejection — unbalanced braces, a "
                    "truncation marker, or a duplicated definition. See `06_structural/`.")
        if result.get("compile_passed"):
            return "No failure detected: the port compiled."
        if result.get("compile_errors"):
            return (f"The port did not compile; {len(result['compile_errors'])} "
                    f"diagnostic(s) remain. Full output in `09_compiler/`.")
        return "No single root cause could be inferred from the recorded artifacts."

    def _recommend(self, result: Dict) -> str:
        abort = (result or {}).get("abort_reason", "")
        if abort == "harness_origin":
            return (f"Add or correct `src/verification/specs/{self.kernel_name}.json` so "
                    f"the generated harness matches the kernel's real signature, then re-run.")
        if abort == "pipeline_timeout":
            return "Raise the wall-clock budget, or narrow the kernel, then re-run."
        if abort in ("runtime_stagnation", "layer2_rejected"):
            return ("Inspect `08_static_analysis/` for warp-32 findings on the crashing "
                    "generation; they usually name the defect the compiler cannot see.")
        if result.get("compile_passed"):
            return "None — verify the binary's output against the CUDA reference."
        return ("Open `12_failure/failure_snapshot.json`, then the raw response of the "
                "last generation in `03_translation/`. The defect is visible in the text "
                "the model actually returned, without re-running the pipeline.")

    def _render_summary(self, metrics: Dict, result: Dict,
                        root_cause: str, action: str) -> str:
        lines: List[str] = []
        add = lines.append

        status = ("PASSED" if result.get("compile_passed") and not result.get("abort_reason")
                  else result.get("abort_reason") or "INCOMPLETE")

        add(f"# Debug session — {self.kernel_name}")
        add("")
        add(f"- **Session**: `{self.session_id}`")
        add(f"- **Started**: {metrics['started_at']}")
        add(f"- **Final state**: `{self._state}`")
        add(f"- **Status**: **{status}**")
        add(f"- **Total runtime**: {metrics['total_runtime_ms'] / 1000:.1f}s "
            f"({metrics['llm_latency_ms'] / 1000:.1f}s in LLM calls)")
        add(f"- **Cost**: ${metrics['totals']['cost']:.4f} across "
            f"{metrics['totals']['llm_calls']} call(s), "
            f"{metrics['totals']['tokens']} tokens")
        add("")

        add("## Execution timeline")
        add("")
        if self._transitions:
            add("| # | From | To | Reason | Elapsed |")
            add("|---|------|----|--------|---------|")
            for t in self._transitions:
                add(f"| {t['index']} | `{t['previous_state']}` | `{t['next_state']}` "
                    f"| {str(t['reason'])[:70]} | {t['elapsed_in_previous_ms'] / 1000:.1f}s |")
        else:
            add("_No state transitions were recorded._")
        add("")

        add("## Retry history")
        add("")
        if self._events:
            for e in self._events:
                iter_txt = f"iter {e['iteration']}" if e.get("iteration") else "—"
                add(f"- `{e['at']}` **{e['event']}** ({iter_txt}) — "
                    f"{str(e.get('reason', ''))[:80]}")
        else:
            add("_No retry events were recorded._")
        add("")

        add("## Validation outcomes")
        add("")
        add(f"- Generations produced: **{metrics['generations']}** "
            f"(each stored separately under `{STAGE_TRANSLATION}/`)")
        add(f"- Patches applied: **{metrics['patches']}** (`{STAGE_PATCHES}/`)")
        add(f"- Compiler invocations: **{metrics['compiles']}** (`{STAGE_COMPILER}/`)")
        for key, val in metrics["counters"].items():
            add(f"- `{key}`: {val}")
        add("")

        add("## Compiler diagnostics")
        add("")
        errs = [str(e) for e in (result.get("compile_errors") or [])]
        # The pipeline mirrors structural-gate errors onto `compile_errors` so its
        # own downstream readers keep working. They are NOT compiler output, and a
        # summary that files them under "compiler diagnostics" — while pointing at
        # an empty 09_compiler/ — is precisely the kind of lie Debug Mode exists
        # to abolish. Separate them.
        synthetic = [e for e in errs if e.startswith("[structural]") or e.startswith("[lexical]")]
        real = [e for e in errs if e not in synthetic]

        # "hipcc never ran" is claimed only when nothing contradicts it. A real
        # diagnostic is proof the compiler ran, even if no compile artifact was
        # recorded (a caller may not have wired log_compile) — so the presence of
        # one always wins over the counter.
        if metrics["compiles"] == 0 and not real:
            add(f"**hipcc never ran.** {metrics['generations']} generation(s) were rejected "
                f"by a pre-compile gate, so there is no compiler output to show and "
                f"`{STAGE_COMPILER}/` is empty by design.")
        elif real:
            where = (f"The full, untruncated stdout/stderr of every one of the "
                     f"{metrics['compiles']} compiler invocation(s) is in "
                     f"`{STAGE_COMPILER}/`." if metrics["compiles"] else
                     "No compiler invocation was recorded for this run, so only the "
                     "parsed diagnostics survive.")
            add(f"{len(real)} diagnostic(s) on the final attempt. {where}")
            add("")
            add("```")
            for e in real[:10]:
                add(e[:200])
            add("```")
        else:
            add(f"_No compiler diagnostics on the final attempt "
                f"({metrics['compiles']} invocation(s) recorded in `{STAGE_COMPILER}/`)._")

        if synthetic:
            add("")
            add(f"{len(synthetic)} gate rejection(s) were recorded on the final attempt. "
                f"These are produced by the lexical/structural validators, not by hipcc:")
            add("")
            add("```")
            for e in synthetic[:10]:
                add(e[:200])
            add("```")
        add("")

        add("## Timing breakdown")
        add("")
        if metrics["per_stage"]:
            add("| Stage | Calls | Total | Mean | Failures |")
            add("|-------|-------|-------|------|----------|")
            for s in sorted(metrics["per_stage"], key=lambda x: -x["total_ms"]):
                add(f"| {s['stage']} | {s['calls']} | {_dur(s['total_ms'])} "
                    f"| {_dur(s['mean_ms'])} | {s['failures']} |")
        else:
            add("_No stage timings were recorded._")
        add("")

        if metrics["token_usage"]:
            add("## Token usage")
            add("")
            add("| Model | Calls | Tokens | Cost | Mean latency |")
            add("|-------|-------|--------|------|--------------|")
            for u in metrics["token_usage"]:
                add(f"| {u['model']} | {u['calls']} | {u['tokens']} "
                    f"| ${u['cost']:.4f} | {u['mean_latency_ms'] / 1000:.1f}s |")
            add("")

        add("## Probable root cause")
        add("")
        add(root_cause or self._infer_root_cause(result))
        add("")
        add("## Recommended next action")
        add("")
        add(action or self._recommend(result))
        add("")
        add("---")
        add("")
        add(f"_Every artifact referenced above lives in `{self.dir}`. "
            f"This session is replayable offline: no further LLM execution is "
            f"required to diagnose the run._")
        add("")
        return "\n".join(lines)


# ── Compiler introspection helper ───────────────────────────────────────────

_version_cache: Dict[str, str] = {}


def compiler_version(hipcc_path: str) -> str:
    """``hipcc --version`` output, cached per path.

    Cached because the compiler cannot change mid-run, and a debug session that
    shells out once per compile would distort the very timings it records.
    """
    if not hipcc_path:
        return ""
    if hipcc_path in _version_cache:
        return _version_cache[hipcc_path]
    try:
        proc = subprocess.run([hipcc_path, "--version"], capture_output=True,
                              text=True, timeout=10)
        out = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError):
        out = ""
    _version_cache[hipcc_path] = out
    return out


# ── Replay: consume a recorded session without a new LLM call ──────────────
#
# Debug Mode records; this replays. Given a session directory and a generation
# number, re-run the SAME text-level gates (extraction, lexical, structural)
# against the RECORDED raw response — no network call, no model, no cost. Two
# uses: verify a gate fix against a real captured failure for free and
# repeatably, and act as a forcing function on the artifact contract — if a
# stage cannot be replayed from its own recorded input alone, that stage was
# secretly depending on context nothing declared.
#
# Deliberately narrow: only the deterministic, pure-text stages replay (no
# LLM calls to replay them against, by construction). The compiler stage is
# NOT replayed here — hipcc's own determinism is the compiler's problem, and
# 09_compiler/ already keeps the exact command, environment and version for a
# human to reproduce that step manually if needed.

def _find_artifact(session_dir: "str | Path", stage: str, contains: str) -> Optional[Path]:
    """The single, sequence-numbered file in *stage* whose name contains
    *contains*, or None. Raises nothing — a missing artifact is a normal
    outcome for a generation that never reached that stage (e.g. a lexical
    reject has no structural report)."""
    stage_dir = Path(session_dir) / stage
    if not stage_dir.is_dir():
        return None
    matches = sorted(p for p in stage_dir.glob(f"*{contains}*") if p.is_file())
    return matches[0] if matches else None


def replay_generation(session_dir: "str | Path", generation: int) -> Dict:
    """Re-validate one recorded generation with zero new LLM calls.

    Reads ``03_translation/*gen{generation:03d}*_raw_response.txt`` back from
    disk and re-runs extraction → lexical → structural against it, exactly as
    ``router._postprocess_port`` did the first time. Returns a dict with the
    replayed verdicts and, where the original artifacts are present, a
    ``matches_recorded`` comparison per stage — ``False`` there means either
    the code under test has changed since the session was recorded, or the
    recorded artifact and the replay disagree for a reason worth investigating.

    Raises ``FileNotFoundError`` if the raw response for *generation* was never
    recorded (wrong generation number, or a session from before this stage
    existed) — a caller asking to replay something that was never captured
    should see that plainly, not a silently empty result.
    """
    from verification.extraction import extract_code
    from verification.lexical import validate_lexical
    from verification.structural import validate_structure
    # The SAME extraction fallback chain _postprocess_port uses — not just the
    # v2 extractor. Replaying only the v2 extractor's verdict, when the
    # original run fell through to the legacy regex fallback (as it does for
    # anything the v2 extractor rejects outright), would silently validate a
    # DIFFERENT string than the one hipcc/the structural gate actually saw —
    # a replay that quietly checks a different algorithm than the one it
    # claims to reproduce is worse than no replay at all.
    from router import ModelRouter as _Router

    session_dir = Path(session_dir)
    gen_tag = f"gen{generation:03d}"

    translation_dir = session_dir / STAGE_TRANSLATION
    # log_generation() names this file "..._{gen_tag}_iter{N}_raw_response.txt";
    # a direct glob for that exact suffix — not _find_artifact's looser
    # substring match — is what keeps this from also matching the extracted
    # code / report.json siblings the same generation writes alongside it.
    candidates = sorted(translation_dir.glob(f"*{gen_tag}*_raw_response.txt")) \
        if translation_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no raw response recorded for generation {generation} under {translation_dir}")
    raw_path = candidates[0]

    raw_response = raw_path.read_text(encoding="utf-8")

    original_cu_paths = sorted((session_dir / STAGE_INPUT).glob("*original.cu")) \
        if (session_dir / STAGE_INPUT).is_dir() else []
    original_source = (original_cu_paths[0].read_text(encoding="utf-8")
                       if original_cu_paths else "")

    extraction = extract_code(raw_response)
    code = extraction.code if extraction.ok else _Router._extract_code(raw_response)
    # Both gates run unconditionally on whatever `code` is — including empty —
    # exactly as _postprocess_port does; validate_lexical/validate_structure
    # both handle an empty string as a real (rejecting) verdict, not a case to
    # special-case around. The try/except mirrors _postprocess_port's own
    # defensive handling: a validator bug must not take down a replay either.
    try:
        lexical = validate_lexical(code)
    except Exception:
        lexical = None
    structural = (validate_structure(original_source, code)
                 if original_source else None)

    result: Dict[str, Any] = {
        "session_dir": str(session_dir),
        "generation": generation,
        "raw_response_path": str(raw_path),
        "replayed": {
            "extraction_ok": extraction.ok,
            "extraction_strategy": extraction.strategy,
            "lexical_ok": (lexical.ok if lexical is not None else None),
            "lexical_reason": (lexical.reason() if lexical is not None else "no code to validate"),
            "structural_ok": (structural.ok if structural is not None else None),
            "structural_reason": (structural.reason() if structural is not None else "no original source recorded"),
        },
        "recorded": {},
        "matches_recorded": {},
        "llm_calls_made": 0,  # the whole point: always zero
    }

    lex_artifact = _find_artifact(session_dir, STAGE_LEXICAL, gen_tag)
    if lex_artifact is not None:
        try:
            recorded_lex = json.loads(lex_artifact.read_text(encoding="utf-8"))
            result["recorded"]["lexical_pass"] = recorded_lex.get("pass")
            result["matches_recorded"]["lexical"] = (
                recorded_lex.get("pass") == result["replayed"]["lexical_ok"])
        except (json.JSONDecodeError, OSError):
            pass

    struct_artifact = _find_artifact(session_dir, STAGE_STRUCTURAL, gen_tag)
    if struct_artifact is not None:
        try:
            recorded_struct = json.loads(struct_artifact.read_text(encoding="utf-8"))
            result["recorded"]["structural_pass"] = recorded_struct.get("pass")
            result["matches_recorded"]["structural"] = (
                recorded_struct.get("pass") == result["replayed"]["structural_ok"])
        except (json.JSONDecodeError, OSError):
            pass

    return result
