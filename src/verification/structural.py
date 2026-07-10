"""Structural validation of an LLM-generated HIP port, BEFORE hipcc.

Why this exists
---------------
A hipcc compile costs ~20s of a 180s pipeline budget. Spending it on a
generation that is obviously broken — unbalanced braces, a truncated final
block, a dropped ``main()`` — buys nothing: the compiler will report errors the
coder never introduced, every refinement iteration reproduces them, and the
error count stays flat while the budget drains (the Δ+0, new:0 signature).

This module answers one question in <1ms:

    "Is this generation structurally coherent enough to be worth a compile?"

It is deliberately CONSERVATIVE. A false reject stalls the loop, which is worse
than a wasted compile. Every check here is one that cannot plausibly fire on
valid C++. Anything fuzzier (line-count ratios, symbol preservation) is reported
as a WARNING and fed to the repair prompt rather than used to reject.

Design notes
------------
- Brace/paren counting is string- and comment-aware. A naive count trips on
  ``'{'``, ``"}"``, and ``// }`` and would reject valid code.
- Angle brackets are NOT balance-checked: ``a < b`` and ``a >> b`` make that
  unsound without a real parser. Template imbalance is left to hipcc.
- Symbol preservation is scoped to top-level function/kernel definitions. That
  covers the failure class actually observed (dropped main(), dropped helper),
  without the false positives a general C++ symbol extractor produces on
  templates and macros.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# A generated port shorter than this fraction of its source is suspicious.
# Advisory only — a port can legitimately shrink (removed CUDA-only guards).
LINE_COUNT_WARN_RATIO = 0.5

# Below this fraction, the generation is almost certainly truncated.
LINE_COUNT_REJECT_RATIO = 0.25

# Markers a model emits when it gives up mid-file. These are text, not C++.
_TRUNCATION_MARKERS = re.compile(
    r'^\s*(?:'
    r'//\s*\.\.\.\s*(?:rest|remaining|truncated|omitted)'
    r'|/\*\s*\.\.\.\s*\*/'
    r'|//\s*TRUNCATED'
    r'|<!--\s*TRUNCATED'
    r'|#\s*\.\.\.'
    r'|\.\.\.\s*$'
    r')',
    re.MULTILINE | re.IGNORECASE,
)

# Top-level function/kernel definition: optional qualifiers, a return type,
# a name, an argument list, then an opening brace (not a semicolon — that is a
# prototype, which defines no symbol).
_FUNC_DEF = re.compile(
    r'^[ \t]*'
    r'(?:(?:__global__|__device__|__host__|static|inline|extern|template\s*<[^>]*>)\s+)*'
    r'(?:[A-Za-z_]\w*(?:\s*[*&])?\s+)+'      # return type (possibly qualified)
    r'([A-Za-z_]\w*)'                        # <-- captured: the name
    r'\s*\([^;{]*\)\s*'
    r'(?:const\s*)?'
    r'\{',
    re.MULTILINE,
)


@dataclass
class ValidationResult:
    """Outcome of a structural check.

    ``ok`` gates the compile. ``warnings`` never gate anything — they are
    context for the repair prompt, so the model is told *what* looks wrong
    rather than being handed a raw compiler error it did not cause.
    """
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_symbols: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """0.0–1.0 structural integrity, for per-iteration logging."""
        if not self.ok:
            return 0.0
        return max(0.0, 1.0 - 0.1 * len(self.warnings))

    def reason(self) -> str:
        """One-line human summary, safe to put in a log or a prompt."""
        if self.ok and not self.warnings:
            return "structurally valid"
        parts = []
        if self.errors:
            parts.append("REJECTED: " + "; ".join(self.errors))
        if self.warnings:
            parts.append("warnings: " + "; ".join(self.warnings))
        return " | ".join(parts)


def _strip_strings_and_comments(code: str) -> str:
    """Blank out string/char literals and comments, preserving length & newlines.

    Balance counting on raw source is wrong: ``char c = '{';`` and ``// }`` both
    contribute phantom braces. Replacing their contents with spaces (rather than
    deleting) keeps offsets stable if a caller wants to map back to a line.
    """
    out = list(code)
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        # line comment
        if ch == '/' and i + 1 < n and code[i + 1] == '/':
            while i < n and code[i] != '\n':
                out[i] = ' '
                i += 1
            continue
        # block comment
        if ch == '/' and i + 1 < n and code[i + 1] == '*':
            out[i] = out[i + 1] = ' '
            i += 2
            while i + 1 < n and not (code[i] == '*' and code[i + 1] == '/'):
                if code[i] != '\n':
                    out[i] = ' '
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = ' '
                i += 2
            continue
        # string / char literal
        if ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n and code[i] != quote:
                if code[i] == '\\':      # escape: skip the escaped char too
                    out[i] = ' '
                    i += 1
                    if i < n:
                        out[i] = ' '
                        i += 1
                    continue
                if code[i] != '\n':
                    out[i] = ' '
                i += 1
            if i < n:
                i += 1
            continue
        i += 1
    return ''.join(out)


def strip_comments(code: str) -> str:
    """Blank out comments only, preserving string literals, length and newlines.

    :func:`_strip_strings_and_comments` is the right tool for balance counting and
    symbol extraction — a brace or an identifier inside ``"..."`` is not code. It
    is the WRONG tool for anything that reads a preprocessor directive, because
    ``#include "foo.cuh"`` carries its payload *inside* a string literal, and
    blanking it turns the line into ``#include "        "``.

    Offsets and line numbers are preserved, so a match found here maps directly
    back onto the original source.
    """
    out = list(code)
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        if ch == '/' and i + 1 < n and code[i + 1] == '/':
            while i < n and code[i] != '\n':
                out[i] = ' '
                i += 1
            continue
        if ch == '/' and i + 1 < n and code[i + 1] == '*':
            out[i] = out[i + 1] = ' '
            i += 2
            while i + 1 < n and not (code[i] == '*' and code[i + 1] == '/'):
                if code[i] != '\n':
                    out[i] = ' '
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = ' '
                i += 2
            continue
        # A string or char literal is skipped over intact — that is the whole
        # point — but we must still consume it so a '//' inside it is not
        # mistaken for a comment.
        if ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n and code[i] != quote:
                if code[i] == '\\':
                    i += 1
                i += 1
            i += 1
            continue
        i += 1
    return ''.join(out)


def _balance(code: str, opener: str, closer: str) -> Tuple[bool, int]:
    """Return (balanced, depth). Negative depth = a closer with no opener."""
    depth = 0
    for ch in code:
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth < 0:
                return False, depth
    return depth == 0, depth


def extract_top_level_functions(code: str) -> List[str]:
    """Names of top-level function/kernel *definitions* (not prototypes).

    Scoped narrowly on purpose: this catches "the coder dropped main()" and
    "the coder dropped a __device__ helper", which is the observed failure
    class. It does not attempt structs, macros, or templates — a regex cannot
    do those without false positives, and false positives stall the loop.
    """
    cleaned = _strip_strings_and_comments(code)
    names = []
    for m in _FUNC_DEF.finditer(cleaned):
        name = m.group(1)
        # Filter control-flow keywords that look like calls: `if (x) {`
        if name in {'if', 'for', 'while', 'switch', 'catch', 'return', 'else'}:
            continue
        names.append(name)
    return names


def validate_structure(source: str, ported: str) -> ValidationResult:
    """Gate a generated port before it reaches hipcc.

    ``source`` is the original CUDA; ``ported`` the model's HIP output.
    Rejects only on defects that cannot occur in valid C++.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not ported or not ported.strip():
        return ValidationResult(ok=False, errors=["empty generation"])

    cleaned = _strip_strings_and_comments(ported)

    # ---- hard rejects: cannot happen in valid C++ ----
    ok_brace, depth = _balance(cleaned, '{', '}')
    if not ok_brace:
        errors.append(
            f"unbalanced braces (depth {depth:+d}) — "
            + ("truncated before the final '}'" if depth > 0 else "stray '}'"))

    ok_paren, pdepth = _balance(cleaned, '(', ')')
    if not ok_paren:
        errors.append(f"unbalanced parentheses (depth {pdepth:+d})")

    if _TRUNCATION_MARKERS.search(ported):
        errors.append("contains a truncation marker ('// ... rest of code')")

    src_lines = len([l for l in source.splitlines() if l.strip()])
    prt_lines = len([l for l in ported.splitlines() if l.strip()])
    if src_lines and prt_lines < src_lines * LINE_COUNT_REJECT_RATIO:
        errors.append(
            f"generation is {prt_lines} lines vs {src_lines} in source "
            f"(<{int(LINE_COUNT_REJECT_RATIO*100)}%) — almost certainly truncated")

    # ---- symbol preservation: WARN, do not reject ----
    # A missing symbol is strong evidence of a bad port, but a regex extractor
    # is not trustworthy enough to gate a compile on. Feed it to the repair
    # prompt instead — that is where it actually helps.
    src_funcs = set(extract_top_level_functions(source))
    prt_funcs = set(extract_top_level_functions(ported))
    missing = sorted(src_funcs - prt_funcs)
    if missing:
        warnings.append("symbols dropped: " + ", ".join(missing))

    if src_lines and prt_lines < src_lines * LINE_COUNT_WARN_RATIO:
        warnings.append(f"port is {prt_lines}/{src_lines} lines — unusually short")

    # Duplicated body: the same function defined twice usually means the model
    # re-emitted the file after an apology. hipcc calls it a redefinition.
    dupes = [n for n in prt_funcs
             if len([m for m in extract_top_level_functions(ported) if m == n]) > 1]
    if dupes:
        errors.append("duplicate definitions: " + ", ".join(sorted(set(dupes))))

    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        missing_symbols=missing,
    )
