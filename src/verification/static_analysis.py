"""Pre-compilation static analysis of a generated HIP port.

Why this exists
---------------
Between the structural gate (<1ms, "is this balanced C++?") and hipcc (~20s,
"is this correct C++?") there is a band of defects that are cheap to find and
expensive to discover from a compiler diagnostic:

  * a ``__global__`` function that returns something other than ``void``
  * a residual CUDA symbol no HIP header defines (``cudaMalloc``, ``__syncwarp``)
  * a 32-bit shuffle mask left on a wavefront64 target
  * two definitions of the same function
  * code after an unconditional ``return`` in the same block

None of these gate the pipeline. They are *findings*, persisted before the
compile so a post-mortem can answer "did we already know?" without re-running
hipcc. When a compile fails, the finding that predicted it is already on disk
with a line number.

Design invariants
-----------------
* **Deterministic.** Same input text → byte-identical findings, in a stable
  order (line, then severity, then rule). No dict iteration order, no clocks.
* **Advisory only.** Nothing here returns a verdict the pipeline may gate on.
  ``severity`` is metadata, not control flow.
* **Honest about being a regex.** These rules run on text, not an AST. Each is
  scoped to a pattern that cannot plausibly fire on valid code; where that
  guarantee is weakest the rule downgrades itself rather than reject.
* **Provider-agnostic.** Nothing here knows which model wrote the code.

The analyzer never raises on malformed input: a caller mid-failure is exactly
who needs it most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List

from .structural import (
    _strip_strings_and_comments, strip_comments, extract_top_level_functions,
)


# Severity ranks, used only for sorting and display.
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    """One static-analysis observation, anchored to a line.

    ``rule`` is a stable identifier — reports group on it, so its spelling must
    not change once written. ``line`` is 1-indexed; 0 means "the whole file".
    """
    rule: str
    severity: str
    line: int
    message: str
    evidence: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StaticAnalysisReport:
    """All findings for one generation, plus the counts a summary wants."""
    findings: List[Finding] = field(default_factory=list)
    lines_analyzed: int = 0

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def infos(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "info"]

    def counts(self) -> Dict[str, int]:
        return {
            "error": len(self.errors),
            "warning": len(self.warnings),
            "info": len(self.infos),
            "total": len(self.findings),
        }

    def to_dict(self) -> Dict:
        return {
            "lines_analyzed": self.lines_analyzed,
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
        }


# ── Rule tables ─────────────────────────────────────────────────────────────

# CUDA symbols with no HIP spelling that a hipify pass would have rewritten.
# Their survival into a "ported" file means the port is incomplete, and hipcc
# will say so 20 seconds from now with an unhelpful "undeclared identifier".
_RESIDUAL_CUDA = (
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemset",
    "cudaDeviceSynchronize", "cudaStreamCreate", "cudaStreamSynchronize",
    "cudaGetLastError", "cudaGetErrorString", "cudaEventCreate",
    "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime",
    "cudaSetDevice", "cudaGetDeviceCount", "cudaGetDeviceProperties",
    "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost", "cudaSuccess",
    "cudaError_t", "cudaStream_t", "cudaEvent_t", "cudaDeviceProp",
)

# Constructs that compile on CUDA and either do not exist on HIP or mean
# something different on a 64-lane wavefront.
_INVALID_HIP = {
    "__syncwarp": "no HIP equivalent — use __syncthreads()",
    "__activemask": "no HIP equivalent — use __ballot_sync(~0ull, 1)",
    "__match_all_sync": "no HIP equivalent — needs an algorithm redesign",
    "__match_any_sync": "no HIP equivalent — needs an algorithm redesign",
}

# A CUDA header that survived the port. hipcc has no such file.
_CUDA_HEADER = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*[<"](?:cuda[\w.]*\.h|cuda_runtime(?:_api)?\.h'
    r'|device_launch_parameters\.h|cooperative_groups\.h|[^">]*\.cuh)[>"]',
    re.MULTILINE,
)

# A 32-bit full mask handed to a warp intrinsic. On wavefront64 this addresses
# half the lanes; the kernel compiles and silently computes the wrong answer.
_MASK32 = re.compile(r'__(?:shfl\w*|ballot|all|any)_sync\s*\(\s*0x[fF]{8}\b')

# ``__global__`` must return void. Anything else is a hard compile error.
_GLOBAL_NON_VOID = re.compile(
    r'^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?__global__[ \t]+(?!void\b)([A-Za-z_]\w*)[ \t]+\w+[ \t]*\(',
    re.MULTILINE,
)

# A line that *starts* like a device-side declaration. Used only as the anchor
# for the malformed-declaration rule below.
_DECL_START = re.compile(
    r'^[ \t]*(?:__global__|__device__|__host__|__shared__|__constant__)[ \t]+\S')

# A literal 32 asserted to be the warp width — the most common wavefront64 defect.
# Covers the three spellings that actually occur: a macro definition, a constant
# initializer, and a comparison against the builtin.
_WARP_LITERAL_32 = re.compile(
    r'^[ \t]*#[ \t]*define[ \t]+(?:WARP_SIZE|WARPSIZE)[ \t]+32\b'
    r'|\b(?:WARP_SIZE|WARPSIZE|warp_size|warpSize)\s*=\s*32\b'
    r'|\bwarpSize\s*==\s*32\b',
    re.MULTILINE)


def _line_of(text: str, index: int) -> int:
    """1-indexed line number of *index* within *text*."""
    return text.count("\n", 0, index) + 1


def _snippet(source_lines: List[str], line_no: int, width: int = 120) -> str:
    """The source line at *line_no* (1-indexed), trimmed. Empty when out of range."""
    if 1 <= line_no <= len(source_lines):
        return source_lines[line_no - 1].strip()[:width]
    return ""


def _find_unreachable_code(cleaned: str, lines: List[str]) -> List[Finding]:
    """Statements following an unconditional ``return``/``break`` in the same block.

    Scoped to the narrowest defensible case: the very next non-blank line, where
    the jump is not guarded by a conditional on the same line (``if (x) return;``)
    and the next line neither closes the block nor is a label. Anything subtler
    needs an AST, and a false positive here sends a repair prompt chasing a
    phantom.
    """
    findings: List[Finding] = []
    cleaned_lines = cleaned.splitlines()
    for i, raw in enumerate(cleaned_lines):
        stripped = raw.strip()
        if not re.fullmatch(r'(?:return\b[^;]*|break|continue)\s*;', stripped):
            continue
        # `if (...) return;` guards a branch — what follows is reachable.
        if re.search(r'\b(?:if|else|for|while|case|default)\b', raw):
            continue
        for j in range(i + 1, len(cleaned_lines)):
            nxt = cleaned_lines[j].strip()
            if not nxt:
                continue
            if nxt.startswith("}"):        # block ends; nothing was unreachable
                break
            if re.match(r'^(?:case\b|default\s*:|\w+\s*:(?!:))', nxt):
                break                      # a label can be jumped to
            findings.append(Finding(
                rule="unreachable-code",
                severity="warning",
                line=j + 1,
                message=f"statement follows an unconditional "
                        f"'{stripped.split()[0].rstrip(';')}' on line {i + 1}",
                evidence=_snippet(lines, j + 1),
            ))
            break
    return findings


def _find_malformed_declarations(lines: List[str]) -> List[Finding]:
    """A device-side declaration line terminating in nothing parseable.

    Fires on e.g. ``__global__ void k(float* a)`` with no ``{`` and no ``;`` and
    no continuation — the shape a truncated generation leaves behind.
    """
    findings: List[Finding] = []
    for i, raw in enumerate(lines):
        if not _DECL_START.match(raw):
            continue
        stripped = raw.strip()
        if stripped.endswith((";", "{", "}", ",", "\\", "(", "&&", "||", "+", "-", "=")):
            continue
        # An unclosed parameter list continues on the next line — that is fine.
        if stripped.count("(") > stripped.count(")"):
            continue
        # A complete signature with the brace on the following line is idiomatic.
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt.startswith("{"):
            continue
        findings.append(Finding(
            rule="malformed-declaration",
            severity="warning",
            line=i + 1,
            message="declaration ends without ';', '{', or a continuation — "
                    "possible truncation",
            evidence=stripped[:120],
        ))
    return findings


def _find_duplicate_definitions(ported: str, lines: List[str]) -> List[Finding]:
    """The same top-level function defined more than once.

    hipcc calls this a redefinition. It is what a model emits when it re-sends
    the whole file after an apology, and it is invisible in a truncated diff.
    """
    names = extract_top_level_functions(ported)
    seen: Dict[str, int] = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
    findings: List[Finding] = []
    for name in sorted(n for n, c in seen.items() if c > 1):
        # Anchor to the SECOND definition — the one hipcc will point at.
        hits = [m.start() for m in re.finditer(
            r'^[^\n]*\b' + re.escape(name) + r'\s*\([^;{]*\)\s*(?:const\s*)?\{',
            ported, re.MULTILINE)]
        line_no = _line_of(ported, hits[1]) if len(hits) > 1 else 0
        findings.append(Finding(
            rule="duplicate-definition",
            severity="error",
            line=line_no,
            message=f"'{name}' is defined {seen[name]} times — hipcc will report "
                    f"a redefinition",
            evidence=_snippet(lines, line_no),
        ))
    return findings


def _find_invalid_hip_constructs(cleaned: str, directives: str,
                                 lines: List[str]) -> List[Finding]:
    """CUDA-only intrinsics, headers and symbols that hipcc cannot resolve.

    *cleaned* has comments AND string literals blanked — correct for identifiers.
    *directives* has only comments blanked — required for ``#include``, whose
    payload lives inside a string literal. Both preserve offsets, so a match in
    either maps back onto the same line number.
    """
    findings: List[Finding] = []

    for sym in sorted(_INVALID_HIP):
        for m in re.finditer(r'\b' + re.escape(sym) + r'\b', cleaned):
            line_no = _line_of(cleaned, m.start())
            findings.append(Finding(
                rule="invalid-hip-construct",
                severity="error",
                line=line_no,
                message=f"'{sym}': {_INVALID_HIP[sym]}",
                evidence=_snippet(lines, line_no),
            ))

    for m in _CUDA_HEADER.finditer(directives):
        line_no = _line_of(directives, m.start())
        findings.append(Finding(
            rule="cuda-header",
            severity="error",
            line=line_no,
            message="CUDA header survived the port — hipcc has no such file; "
                    "use <hip/hip_runtime.h>",
            evidence=_snippet(lines, line_no),
        ))

    for sym in _RESIDUAL_CUDA:
        for m in re.finditer(r'\b' + re.escape(sym) + r'\b', cleaned):
            line_no = _line_of(cleaned, m.start())
            findings.append(Finding(
                rule="residual-cuda-symbol",
                severity="error",
                line=line_no,
                message=f"'{sym}' has no HIP definition — the port is incomplete",
                evidence=_snippet(lines, line_no),
            ))

    for m in _GLOBAL_NON_VOID.finditer(cleaned):
        line_no = _line_of(cleaned, m.start())
        findings.append(Finding(
            rule="global-non-void",
            severity="error",
            line=line_no,
            message=f"__global__ function returns '{m.group(1)}' — it must return void",
            evidence=_snippet(lines, line_no),
        ))

    return findings


def _find_wavefront_hazards(cleaned: str, lines: List[str]) -> List[Finding]:
    """32-lane assumptions that compile cleanly and then compute the wrong answer.

    These are ``warning``, not ``error``: the code is valid C++ and hipcc will
    never complain. That is precisely why they belong in a persisted report — a
    SIGSEGV three stages later is otherwise unattributable.
    """
    findings: List[Finding] = []
    for m in _MASK32.finditer(cleaned):
        line_no = _line_of(cleaned, m.start())
        findings.append(Finding(
            rule="warp-mask-32",
            severity="warning",
            line=line_no,
            message="32-bit lane mask on a warp intrinsic — addresses half of a "
                    "64-lane wavefront",
            evidence=_snippet(lines, line_no),
        ))
    for m in _WARP_LITERAL_32.finditer(cleaned):
        line_no = _line_of(cleaned, m.start())
        findings.append(Finding(
            rule="warp-size-32",
            severity="warning",
            line=line_no,
            message="warp size hardcoded to 32 — an AMD wavefront is 64 lanes",
            evidence=_snippet(lines, line_no),
        ))
    return findings


def analyze(ported: str) -> StaticAnalysisReport:
    """Run every rule over *ported* and return a deterministic report.

    Findings are sorted by (line, severity, rule, message) so two runs over the
    same text produce byte-identical JSON — which is what makes one debug
    session diffable against another.
    """
    if not ported or not ported.strip():
        return StaticAnalysisReport(findings=[], lines_analyzed=0)

    lines = ported.splitlines()
    # Comments and string literals are blanked so no rule fires on
    # ``// remember to replace cudaMalloc`` or ``printf("cudaMalloc failed")``.
    cleaned = _strip_strings_and_comments(ported)
    # Comments only. `#include "foo.cuh"` keeps its payload, which lives inside
    # a string literal and would otherwise be blanked out of existence.
    directives = strip_comments(ported)

    findings: List[Finding] = []
    for rule_fn, args in (
        (_find_invalid_hip_constructs, (cleaned, directives, lines)),
        (_find_wavefront_hazards, (cleaned, lines)),
        (_find_unreachable_code, (cleaned, lines)),
        (_find_malformed_declarations, (lines,)),
        (_find_duplicate_definitions, (ported, lines)),
    ):
        try:
            findings.extend(rule_fn(*args))
        except Exception as exc:  # a rule bug must never break a debug dump
            findings.append(Finding(
                rule="analyzer-error",
                severity="info",
                line=0,
                message=f"rule {rule_fn.__name__} raised {exc!r} — skipped",
            ))

    findings.sort(key=lambda f: (f.line, SEVERITY_ORDER.get(f.severity, 9),
                                 f.rule, f.message))
    return StaticAnalysisReport(findings=findings, lines_analyzed=len(lines))
