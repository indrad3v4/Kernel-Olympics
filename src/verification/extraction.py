"""Extraction of C/C++/HIP source code from an LLM response.

The response may be:
  * a JSON object with a "code" or "ported_code" field
  * a markdown reply with one or more ``` fenced blocks
  * a plain-text reply with a leading explanation and a raw code section
  * a truncated stream that cut off mid-token

Regardless of shape, we return the largest, most code-shaped candidate we
can identify, or ``None`` when no candidate is confidently source code.
The lexical validator then decides whether the candidate is actually clean
enough to compile.

Nothing in this module is provider-specific.  Every rule fires on the
response text alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Public result type ──────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Outcome of ``extract_code``.

    ``code`` is the string that should be written to disk after further
    validation.  ``None`` means the response contained no confidently
    code-shaped section — the orchestrator must NOT write anything.

    Every field except ``code`` is diagnostic.  They are logged and,
    when a lexical/structural gate fires downstream, fed into the refine
    prompt so the model is told which extraction strategy fired and what
    was discarded.
    """
    code: Optional[str]
    strategy: str
    response_length: int = 0
    code_length: int = 0
    discarded_length: int = 0
    candidates_considered: int = 0
    diagnostics: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.code is not None and bool(self.code.strip())


# ── Constants ───────────────────────────────────────────────────────────────

# Anchors used to slide a code window through plain-text responses.
_CODE_ANCHOR = re.compile(
    r"""(?:
          \#include
        | \#define
        | \#pragma
        | \#ifdef
        | \#ifndef
        | __global__
        | __device__
        | __host__
        | __shared__
        | __constant__
        | extern\s+"C"
        | template\s*<
        | namespace\s+\w
    )""",
    re.VERBOSE,
)

# A markdown fence.  Language tag is captured but never trusted — a model
# may fence C++ inside ```text``` or ```plaintext``` when it's confused.
_FENCE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+#\-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)

# Language tags that plausibly hold C/C++/HIP.  Untagged fences are
# considered too — we score them by content, not by label.
_CODE_TAGS = {"", "cpp", "c++", "cxx", "c", "hip", "cuda", "cu",
              "hipcc", "opencl", "cl"}

# JSON field names, in priority order, that could hold the ported source.
_JSON_CODE_FIELDS = ("ported_code", "code", "hip_code", "source", "output", "port")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _score_code_shape(text: str) -> float:
    """Return a 0.0–1.0 estimate of how source-code-like *text* is.

    Cheap, monotonic, provider-agnostic.  Used to pick between multiple
    candidate blocks (multiple fences, JSON field vs raw block, ...).
    """
    if not text or not text.strip():
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    code_hits = 0
    prose_hits = 0
    for ln in lines:
        s = ln.strip()
        if (s.startswith("#include") or s.startswith("#define") or s.startswith("#pragma")
                or s.startswith("//") or s.startswith("/*") or s.startswith("*")
                or "__global__" in s or "__device__" in s or "__host__" in s
                or s.endswith(";") or s.endswith("{") or s.endswith("}")
                or s in {"{", "}", "};"}):
            code_hits += 1
            continue
        if any(c in s for c in ("{", "}", ";", "->", "::", "==", "!=", "<=", ">=")):
            code_hits += 1
            continue
        # Prose heuristics — strong markers only, so this stays cheap.
        if re.match(r"^(?:Let'?s|I think|Wait|Actually|Here|So we|We need|Note that)",
                    s, re.IGNORECASE):
            prose_hits += 1
            continue
        if re.match(r"^\s*[-*]\s+[A-Za-z]", ln):
            prose_hits += 1
            continue
        if re.match(r"^[A-Z][a-z].{20,}\.\s*$", s):
            prose_hits += 1

    total = code_hits + prose_hits
    if total == 0:
        return 0.0
    base = code_hits / total
    # A block with a real HIP construct scores above one without, even if the
    # ratios are equal — otherwise a 3-line ``int a; int b; int c;`` would
    # beat a 300-line kernel.
    if "__global__" in text or "__device__" in text or "#include" in text:
        base = min(1.0, base + 0.15)
    return base


def _strip_trailing_fence(text: str) -> str:
    """Drop everything from the first line-anchored ``` onwards.

    A raw-code extraction (window slide) captures from a code anchor to the
    end of the response.  If the response was a fenced block followed by
    prose, the closing ``` and the prose ride along.  We cut at the fence.
    """
    m = re.search(r"^[ \t]*```", text, re.MULTILINE)
    if not m:
        return text
    return text[:m.start()].rstrip() + "\n"


def _unescape_json_string(raw: str) -> str:
    """Decode a JSON-encoded string body.

    The JSON parser is preferred (handles unicode escapes, surrogate pairs,
    trailing whitespace), but the ``ported_code`` field can span a
    truncated JSON response where a strict decode fails.  In that case we
    fall back to a hand-decode of the common escapes so a partial response
    still yields as much valid text as possible.
    """
    try:
        return json.loads(f'"{raw}"')
    except (json.JSONDecodeError, ValueError):
        pass
    # Manual decode for truncated JSON — this is the ONLY place we tolerate
    # a lossy string decode, and only because dropping it means dropping
    # every truncated-stream response.
    out = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "\\" and i + 1 < n:
            nx = raw[i + 1]
            if nx == "n":
                out.append("\n"); i += 2; continue
            if nx == "t":
                out.append("\t"); i += 2; continue
            if nx == "r":
                out.append("\r"); i += 2; continue
            if nx == '"':
                out.append('"'); i += 2; continue
            if nx == "\\":
                out.append("\\"); i += 2; continue
            if nx == "/":
                out.append("/"); i += 2; continue
            if nx == "u" and i + 5 < n:
                try:
                    out.append(chr(int(raw[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(nx)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_json_string_field(response: str, field_names) -> Optional[str]:
    """Find ``"<field>": "..."`` and return the decoded string body, or None.

    Two-stage: try a strict ``json.loads`` on the whole response first (fast
    path for well-formed replies), then a targeted extract of the string
    body if the parse fails — the second stage handles truncated JSON where
    a real parser would raise before reaching the ``code`` field.
    """
    # Stage 1: whole-response strict parse.
    try:
        obj = json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for fname in field_names:
            v = obj.get(fname)
            if isinstance(v, str) and v.strip():
                return v

    # Stage 2: strict parse of a ```json ... ``` block.
    for m in re.finditer(r"```[ \t]*json[ \t]*\r?\n(.*?)```", response, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            for fname in field_names:
                v = obj.get(fname)
                if isinstance(v, str) and v.strip():
                    return v

    # Stage 3: hand-extract "field": "..." — tolerant of truncation.
    for fname in field_names:
        m = re.search(rf'"{re.escape(fname)}"\s*:\s*"', response)
        if not m:
            continue
        start = m.end()
        buf = []
        i = start
        n = len(response)
        while i < n:
            c = response[i]
            if c == "\\" and i + 1 < n:
                buf.append(c)
                buf.append(response[i + 1])
                i += 2
                continue
            if c == '"':
                break
            buf.append(c)
            i += 1
        body = "".join(buf)
        if body.strip():
            return _unescape_json_string(body)
    return None


def _extract_fenced_blocks(response: str):
    """Yield (language_tag, body) for every ``` block in *response*."""
    for m in _FENCE.finditer(response):
        yield (m.group(1) or "").lower(), m.group(2)


def _slide_code_window(response: str) -> Optional[str]:
    """Find the earliest code anchor and slide a window to end-of-code.

    "End of code" is heuristic — the last line whose prefix looks like code.
    A run of non-code lines ends the window.  Also cuts at any markdown
    fence encountered along the way (a fence line at file scope means the
    code ended there and prose began).
    """
    m = _CODE_ANCHOR.search(response)
    if not m:
        return None
    start = m.start()
    # Walk backwards to the start of that line so we do not slice mid-token.
    line_start = response.rfind("\n", 0, start) + 1
    tail = response[line_start:]
    tail = _strip_trailing_fence(tail)

    lines = tail.splitlines()

    # ── Trim leading prose ──────────────────────────────────────────────
    # The code anchor (e.g. __global__, #include) may have been found
    # inside a prose sentence: "Potential optimization: kernel uses
    # __global__ void scan(...)".  Walk forward past lines that have NO
    # strong C++ markers — the anchor itself uniquely identifies the first
    # code-carrying line, but anything before the first genuinely code-like
    # line is prose that slipped in.
    #
    # For leading trim we use a STRICTER check than for trailing trim:
    # the code marker must be at/near the start of the line, or the line
    # must have a sentence-ending C++ delimiter.  This prevents prose
    # sentences that merely *mention* "__global__" from being mistaken for
    # code.
    def _is_leading_code(s: str) -> bool:
        """Strict — is the line genuinely C++ code, not prose mentioning code?"""
        if not s:
            return True  # blank lines are neutral (preserve blank > #include)
        # Strong starts: C++ directives, comments, standard prefixes
        if (s.startswith("#") or s.startswith("//") or s.startswith("/*")
                or s.startswith("*")):
            return True
        # __global__ / __device__ at or near start of line (first 20 chars)
        stripped_left = s.lstrip()
        head = stripped_left[:20]
        if ("__global__" in head or "__device__" in head
                or "__host__" in head or "__shared__" in head
                or "__constant__" in head):
            return True
        # template / namespace at line start
        if stripped_left.startswith("template") or stripped_left.startswith("namespace"):
            return True
        # Line-ending delimiters that signal a statement/block
        if s.endswith(";") or s.endswith("{") or s.endswith("}"):
            return True
        return False

    first_code = -1
    for i, ln in enumerate(lines):
        if _is_leading_code(ln.strip()):
            first_code = i
            break
    if first_code < 0:
        return None
    if first_code > 0:
        lines = lines[first_code:]

    # ── Trim trailing prose ────────────────────────────────────────────
    # Walk forward identifying the last plausibly code-shaped line; a run
    # of 3+ consecutive non-code lines ends the window.  Trailing trim
    # uses the original lenient heuristic so we don't accidentally cut the
    # middle of a real function body.
    def _is_code_like(s: str) -> bool:
        """Lenient — does the line look plausibly like C/C++/HIP code?"""
        if not s:
            return True
        if (s.startswith("#") or s.startswith("//") or s.startswith("/*")
                or s.startswith("*")):
            return True
        if ("__global__" in s or "__device__" in s or "__host__" in s
                or "__shared__" in s or "__constant__" in s
                or "template" in s or "namespace " in s):
            return True
        if s.endswith(";") or s.endswith("{") or s.endswith("}"):
            return True
        if s in ("{", "}", "};"):
            return True
        if any(c in s for c in ("{", "}", ";", "->", "::")):
            return True
        return False

    last_code = -1
    consecutive_prose = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if _is_code_like(s):
            last_code = i
            consecutive_prose = 0
            continue
        consecutive_prose += 1
        if consecutive_prose >= 3 and last_code >= 0:
            break
    if last_code < 0:
        return None
    return "\n".join(lines[: last_code + 1]).strip() + "\n"


# ── Entry point ─────────────────────────────────────────────────────────────

def extract_code(response: str) -> ExtractionResult:
    """Extract the C/C++/HIP source from an arbitrary LLM *response*.

    Strategy order (each falls through on empty result):

      1. JSON object with a ``code``/``ported_code`` field.
      2. Markdown fenced block.  If multiple, pick the highest-scoring one.
      3. Sliding code-window anchored on ``#include``, ``__global__``, etc.

    Returns an ``ExtractionResult`` with the extracted string and full
    diagnostics.  Never raises.
    """
    if response is None:
        return ExtractionResult(code=None, strategy="empty", response_length=0,
                                diagnostics=["response is None"])

    diagnostics: List[str] = []
    resp_len = len(response)
    candidates_seen = 0

    # ── Strategy 1: JSON field ────────────────────────────────────────────
    js = _extract_json_string_field(response, _JSON_CODE_FIELDS)
    if js:
        candidates_seen += 1
        code = js.strip()
        return ExtractionResult(
            code=code,
            strategy="json-field",
            response_length=resp_len,
            code_length=len(code),
            discarded_length=max(0, resp_len - len(code)),
            candidates_considered=candidates_seen,
            diagnostics=diagnostics,
        )

    # ── Strategy 2: markdown fenced blocks ────────────────────────────────
    scored = []
    for tag, body in _extract_fenced_blocks(response):
        candidates_seen += 1
        if tag and tag not in _CODE_TAGS:
            # An explicitly non-code tag ("```json", "```text") is skipped
            # unless it happens to score high on code-shapedness later.
            score = _score_code_shape(body) * 0.5
        else:
            score = _score_code_shape(body)
        scored.append((score, tag, body))
    if scored:
        scored.sort(key=lambda t: (t[0], len(t[2])), reverse=True)
        best_score, best_tag, best_body = scored[0]
        if best_score >= 0.5:
            code = best_body.strip() + "\n"
            diagnostics.append(f"fence[{best_tag or 'untagged'}] score={best_score:.2f}")
            if len(scored) > 1:
                diagnostics.append(f"{len(scored) - 1} other fenced block(s) discarded")
            return ExtractionResult(
                code=code,
                strategy="markdown-fence",
                response_length=resp_len,
                code_length=len(code),
                discarded_length=max(0, resp_len - len(code)),
                candidates_considered=candidates_seen,
                diagnostics=diagnostics,
            )
        diagnostics.append(f"best fenced block scored only {best_score:.2f} — rejected")

    # ── Strategy 3: sliding code window over raw text ─────────────────────
    slid = _slide_code_window(response)
    if slid:
        candidates_seen += 1
        score = _score_code_shape(slid)
        diagnostics.append(f"raw-window score={score:.2f}")
        if score >= 0.5:
            return ExtractionResult(
                code=slid,
                strategy="raw-window",
                response_length=resp_len,
                code_length=len(slid),
                discarded_length=max(0, resp_len - len(slid)),
                candidates_considered=candidates_seen,
                diagnostics=diagnostics,
            )

    # ── No confident extraction ───────────────────────────────────────────
    diagnostics.append("no candidate crossed the code-shape threshold")
    return ExtractionResult(
        code=None,
        strategy="rejected",
        response_length=resp_len,
        code_length=0,
        discarded_length=resp_len,
        candidates_considered=candidates_seen,
        diagnostics=diagnostics,
    )


__all__ = ["ExtractionResult", "extract_code"]
