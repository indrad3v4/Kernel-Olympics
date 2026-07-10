"""Lexical validation of an LLM-generated port, BEFORE structural checks.

Why this exists
---------------
The structural validator answers "is this balanced C++?"  A response that is
pure reasoning ("Let's search memory more concretely. I think...") can still
have balanced braces (zero of each) yet is not a source file. hipcc will
report ``error: unknown type name 'local'`` — a symptom of the orchestrator
having written prose to disk, not of a HIP portability issue.

This module answers a strictly text-level question:

    "Does the top-level content of this file look like C/C++ source,
     or does it look like an English explanation of what the source
     might have been?"

Design invariants
-----------------
* Runs BEFORE the structural validator.  A response that fails here never
  reaches the compiler, never triggers a repair prompt, and is never written
  to disk.
* Only rejects on markers that cannot appear at top-level in valid C++.
  Comments, doc comments, copyright headers, string literals containing
  English are tolerated: they contribute zero non-comment prose lines.
* Provider-agnostic.  Every rule fires purely on the text the writer is
  about to save, regardless of which model produced it.

Failure classes we catch
------------------------
1. Reasoning cues ("Let's", "I think", "Wait maybe", "Actually,", ...)
2. Section headings ("Explanation:", "Summary:", "Analysis:", ...)
3. Bullet / numbered lists at top level
4. Markdown headings (#, ##) or unescaped code-fence lines
5. A response that has ZERO code-shaped lines but non-trivial content
6. A response where the ratio of prose lines to code lines exceeds a bound
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# Words / phrases that a reasoning-mode model emits.  Matching on WHOLE WORDS
# or line-anchored constructs keeps this from firing on identifiers like
# ``needs_flush`` or comment tokens like ``// note: ...``.
_REASONING_LEADS = re.compile(
    r"""^\s*(?:
          Let(?:\s|')s\b            # "Let's ...", "Let us ..."
        | I\s+think\b
        | I\s+believe\b
        | I\s+will\b
        | I'll\b
        | I'm\s+going\s+to\b
        | Wait[,.\s]                # "Wait,", "Wait ", "Wait."
        | Actually[,.\s]
        | Hmm[,.\s]
        | Okay[,.\s]                # "Okay, so ..."
        | So\s+we\b
        | So\s+the\b
        | We\s+need\s+to\b
        | We\s+can\b
        | We\s+should\b
        | We\s+must\b
        | Need\s+to\b
        | Should\s+we\b
        | Maybe\b
        | Perhaps\b
        | Note\s+that\b
        | Notice\s+that\b
        | Here'?s\s+
        | Here\s+is\s+
        | This\s+means\b
        | The\s+idea\s+is\b
        | The\s+plan\s+is\b
        | Assistant[:\s]
        | User[:\s]
        | Reasoning[:\s]
        | Thought[:\s]
        | Thinking[:\s]
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Section headings a model writes when it drops out of code-emission mode.
# Must be at the *start* of a line and followed by a colon or line break.
_SECTION_HEADINGS = re.compile(
    r"""^\s*(?:
          Explanation
        | Summary
        | Analysis
        | Reasoning
        | Thought
        | Thinking
        | Plan
        | Approach
        | Solution
        | Answer
        | Question
        | Task
        | Context
        | Goal
        | Objective
        | Changes
        | Rationale
        | Notes?
        | Background
        | Overview
        | Discussion
        | Conclusion
        | Steps?
        | Output
        | Input
        | Response
        | Assistant
        | User
    )\s*:\s*(?:$|[A-Z])""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Markdown constructs that appear at start-of-line and never appear in C++.
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+[A-Za-z]", re.MULTILINE)
_MARKDOWN_FENCE = re.compile(r"^\s*```", re.MULTILINE)
_BULLET_LIST = re.compile(r"^\s*[-*]\s+[A-Za-z]", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^\s*\d+\.\s+[A-Z][a-z]", re.MULTILINE)  # "1. Introduce ..."
# Markdown bold at line start with a matching close on the same line — the
# classic ``**Bold header**`` shape.  This is deliberately restrictive:
# ``StopWatchInterface **t`` is legal C++ (pointer-to-pointer), so we cannot
# reject on ``**`` alone.  A leading ``**`` at line start followed by word
# chars and a matching ``**`` before end-of-line is the actual failure mode.
_MARKDOWN_BOLD = re.compile(
    r"^\s*\*\*[A-Za-z][^\n*]*\*\*\s*$",
    re.MULTILINE,
)

# Conversational role tags a chat model sometimes leaks.
_ROLE_TAG = re.compile(r"^\s*(assistant|user|system)\s*[:>|]", re.IGNORECASE | re.MULTILINE)

# Trailing-ellipsis line — a model that trailed off mid-thought.
_TRAILING_ELLIPSIS = re.compile(r"^[^/#\n]*\.\.\.\s*$", re.MULTILINE)

# A code-shaped line: an obviously-C construct.  The list is deliberately
# broad — anything that appears at file scope in normal HIP source qualifies.
_CODE_ANCHORS = (
    "#include",
    "#define",
    "#pragma",
    "#ifdef",
    "#ifndef",
    "#if ",
    "#endif",
    "#else",
    "#elif",
    "#undef",
    "__global__",
    "__device__",
    "__host__",
    "__shared__",
    "__constant__",
    "extern ",
    "static ",
    "inline ",
    "template",
    "namespace",
    "using ",
    "typedef ",
    "struct ",
    "class ",
    "enum ",
    "void ",
    "int ",
    "float ",
    "double ",
    "bool ",
    "char ",
    "unsigned ",
    "signed ",
    "return",
    "if ",
    "if(",
    "for ",
    "for(",
    "while ",
    "while(",
    "switch ",
    "switch(",
    "hip",
    "std::",
    "//",
    "/*",
    "*/",
    " * ",
)

# Characters whose presence strongly implies code, not prose.
_CODE_ONLY_CHARS = ("{", "}", ";", "->", "::", "==", "!=", "<=", ">=", "&&", "||",
                    "<<", ">>", "++", "--", "+=", "-=", "*=", "/=")

# Absolute cap on the fraction of non-empty top-level lines that may look
# like prose.  Comments count as code (they start with "//" or "/*"), so
# this bound is against genuine free-text lines only.
_MAX_PROSE_RATIO = 0.15


@dataclass
class LexicalResult:
    """Outcome of a lexical scan.

    ``ok`` gates every downstream stage — extraction retry, structural
    validation, and the file writer.  ``reason`` is a one-line human summary
    suitable for logs and refine-prompt feedback.
    """
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    prose_line_samples: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def reason(self) -> str:
        if self.ok and not self.warnings:
            return "lexically clean"
        parts = []
        if self.errors:
            parts.append("REJECTED: " + "; ".join(self.errors))
        if self.warnings:
            parts.append("warnings: " + "; ".join(self.warnings))
        return " | ".join(parts) or "lexical validation failed"


def _blank_out_comments_and_strings(code: str) -> str:
    """Return *code* with comments and string/char literals replaced by spaces.

    The lexical checks below fire on top-level PROSE.  A copyright block, a
    URL in a comment, or an English string literal must not trip them, so we
    blank those regions first while preserving line offsets.
    """
    out = list(code)
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            while i < n and code[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                if code[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
            continue
        if ch == '"':
            quote = ch
            i += 1
            while i < n and code[i] != quote:
                if code[i] == "\\" and i + 1 < n:
                    out[i] = " "
                    i += 1
                    if i < n and code[i] != "\n":
                        out[i] = " "
                        i += 1
                    continue
                if code[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                i += 1
            continue
        if ch == "'":
            # A real char literal is at most 4 chars long: '\n', '\0', 'a'.
            # An English apostrophe ("Let's", "don't") would otherwise eat
            # the entire rest of the line and hide reasoning from the
            # classifier.  Look ahead for a closing quote within 4 chars;
            # only then treat this as a literal.
            close = -1
            j = i + 1
            limit = min(n, i + 6)
            while j < limit:
                if code[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if code[j] == "'":
                    close = j
                    break
                if code[j] == "\n":
                    break
                j += 1
            if close < 0:
                i += 1
                continue
            for k in range(i + 1, close):
                if code[k] != "\n":
                    out[k] = " "
            i = close + 1
            continue
        i += 1
    return "".join(out)


def _is_code_line(line: str) -> bool:
    """A line that plausibly belongs in a C/C++ source file.

    Comment lines are code (they were preserved in the raw ``code``, not the
    blanked copy — but the caller always feeds a raw line).  A line with any
    C-only punctuation, a preprocessor directive, or a top-level keyword is
    code.  Everything else is prose.
    """
    stripped = line.strip()
    if not stripped:
        return True  # blank lines are neutral
    if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
        return True
    if any(stripped.startswith(a) for a in _CODE_ANCHORS):
        return True
    if any(c in stripped for c in _CODE_ONLY_CHARS):
        return True
    # A closing "}" alone, an opening "{" alone, a label like "err:" or "out:":
    if stripped in {"{", "}", "};", "})", "});", "()"}:
        return True
    if re.match(r"^[A-Za-z_]\w*\s*:$", stripped):  # label
        return True
    return False


def _classify_prose_line(line: str) -> Tuple[bool, str]:
    """Return (looks_like_reasoning, category).

    Category is a short tag that goes into the rejection reason so the
    refine prompt can tell the model exactly why its output was dropped.
    """
    stripped = line.strip()
    if not stripped:
        return False, ""
    if _REASONING_LEADS.match(stripped):
        return True, "reasoning"
    if _SECTION_HEADINGS.match(stripped):
        return True, "section-heading"
    if _MARKDOWN_HEADING.match(line):
        return True, "markdown-heading"
    if _BULLET_LIST.match(line):
        return True, "bullet-list"
    if _NUMBERED_LIST.match(line):
        return True, "numbered-list"
    if _ROLE_TAG.match(line):
        return True, "role-tag"
    # An English sentence: starts with a capital letter, contains a space,
    # ends with a period, and has no code punctuation.
    if (re.match(r"^[A-Z][a-z]", stripped)
            and " " in stripped
            and stripped.endswith(".")
            and not any(c in stripped for c in "{};=()")):
        return True, "prose-sentence"
    return False, ""


def validate_lexical(code: str) -> LexicalResult:
    """Reject *code* if its top-level content is prose, not source.

    Runs on the string the writer is about to save.  A pass here does not
    mean the code compiles — that is the structural gate and hipcc.  It
    only means the string is not an English explanation of code.
    """
    if code is None or not code.strip():
        return LexicalResult(ok=False, errors=["empty extraction"])

    stripped = code.strip()

    # Fast rejects — these never appear in valid top-level C++.
    if _MARKDOWN_FENCE.search(code):
        return LexicalResult(
            ok=False,
            errors=["markdown code fence at top level — extractor grabbed the wrapper too"],
        )
    if _MARKDOWN_HEADING.search(code):
        return LexicalResult(
            ok=False,
            errors=["markdown heading (# ...) at top level"],
        )
    if _ROLE_TAG.search(code):
        return LexicalResult(
            ok=False,
            errors=["conversational role tag (assistant:/user:) at top level"],
        )
    if _TRAILING_ELLIPSIS.search(code):
        return LexicalResult(
            ok=False,
            errors=["trailing ellipsis — response ended mid-thought"],
        )
    if _MARKDOWN_BOLD.search(code):
        return LexicalResult(
            ok=False,
            errors=["markdown bold (**...**) at top level"],
        )

    # The heavy check runs on comment/string-blanked source so English inside
    # copyright headers and string literals is invisible.  Line offsets are
    # preserved, so samples we emit still line up with the original file.
    blanked = _blank_out_comments_and_strings(code)
    raw_lines = code.splitlines()
    blanked_lines = blanked.splitlines()

    prose_lines: List[Tuple[int, str, str]] = []
    code_lines = 0
    non_empty_lines = 0
    reasoning_hits = 0
    heading_hits = 0

    for i, (raw, blank) in enumerate(zip(raw_lines, blanked_lines)):
        if not raw.strip():
            continue
        non_empty_lines += 1
        # Determine whether the *raw* line is code (comment lines count as
        # code); the *blanked* line drives prose detection so prose inside
        # comments and strings is ignored.
        if _is_code_line(raw):
            code_lines += 1
            continue
        prose, category = _classify_prose_line(blank if blank.strip() else raw)
        if prose:
            prose_lines.append((i + 1, category, raw.strip()[:120]))
            if category == "reasoning":
                reasoning_hits += 1
            elif category == "section-heading":
                heading_hits += 1

    stats = {
        "non_empty_lines": non_empty_lines,
        "code_lines": code_lines,
        "prose_lines": len(prose_lines),
        "reasoning_hits": reasoning_hits,
        "heading_hits": heading_hits,
    }

    # Hard rejects driven by counts.
    if reasoning_hits >= 1:
        samples = [f"L{n} [{cat}]: {txt}" for n, cat, txt in prose_lines[:3]]
        return LexicalResult(
            ok=False,
            errors=[f"reasoning at top level ({reasoning_hits} line(s))"],
            prose_line_samples=samples,
            stats=stats,
        )
    if heading_hits >= 1:
        samples = [f"L{n} [{cat}]: {txt}" for n, cat, txt in prose_lines[:3]]
        return LexicalResult(
            ok=False,
            errors=[f"section heading at top level ({heading_hits} line(s))"],
            prose_line_samples=samples,
            stats=stats,
        )
    # Any bullet/numbered list at top level, even one line, is a bad extract.
    for _, cat, _ in prose_lines:
        if cat in {"bullet-list", "numbered-list"}:
            samples = [f"L{n} [{c}]: {t}" for n, c, t in prose_lines[:3]]
            return LexicalResult(
                ok=False,
                errors=["markdown list at top level"],
                prose_line_samples=samples,
                stats=stats,
            )

    if non_empty_lines and code_lines == 0:
        return LexicalResult(
            ok=False,
            errors=["no code-shaped lines detected"],
            stats=stats,
        )

    if non_empty_lines:
        prose_ratio = len(prose_lines) / non_empty_lines
        if prose_ratio > _MAX_PROSE_RATIO:
            samples = [f"L{n} [{c}]: {t}" for n, c, t in prose_lines[:3]]
            return LexicalResult(
                ok=False,
                errors=[f"prose ratio {prose_ratio:.0%} exceeds cap "
                        f"{_MAX_PROSE_RATIO:.0%}"],
                prose_line_samples=samples,
                stats=stats,
            )

    # A file with no #include or top-level qualifier is unusual but not
    # invalid — a snippet reduced to a single kernel body may legitimately
    # skip both.  Report as a warning so the writer can log it.
    warnings: List[str] = []
    if "#include" not in code and not re.search(r"__(?:global|device|host)__", code):
        warnings.append("no #include or CUDA/HIP qualifier — snippet or partial file")

    return LexicalResult(ok=True, warnings=warnings, stats=stats)


__all__ = ["LexicalResult", "validate_lexical"]
