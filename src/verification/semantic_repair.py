"""Semantic Translation Repair Engine — compiler diagnostics → minimal source patches.

Why this exists
---------------
By the time a port reaches this module, infrastructure, orchestration, extraction
recovery and structural validation have already done their jobs: the file is
balanced C++, it is not prose, and it was produced by a real model call. What
remains are *semantic translation defects* — a ``#define`` the extractor dropped,
a ``__device__`` helper that never made it across, a struct whose definition was
left behind in the CUDA original. hipcc reports these as ``use of undeclared
identifier`` twenty seconds into a compile, and historically each one cost a full
LLM refine cycle to re-invent code that already existed in the source file.

This engine treats every remaining compiler error as evidence that semantic
information was *lost during translation*, and recovers it from the original CUDA
source rather than asking a model to invent new code. The original ``.cu`` file
is the source of truth; the generated HIP file is only a transform of it.

Design invariants
-----------------
* **Deterministic.** Same (cuda_source, hip_source, diagnostics) → byte-identical
  patches, in a stable order. No clocks in the patch content, no dict-iteration
  order, no model calls.
* **Minimal.** Never regenerate the translation unit. Every patch is the smallest
  legal edit — almost always an *additive* restoration of a definition that the
  original CUDA source already contains, inserted where it is legal and used.
* **Recovers, never guesses.** A symbol is only restored when it is found,
  verbatim, in the original CUDA source (or the compiler itself names the fix, as
  with a ``did you mean`` rename). When resolution fails the diagnostic is left
  *unresolved* for the LLM path — this engine never fabricates a definition.
* **No AST binding required.** "AST" here is the regex symbol table
  (:mod:`verification.symbols`) plus a lightweight scope/definition extractor.
  Everything it reports, it reports from text it can point at.
* **Provider-agnostic and side-effect-free.** The engine mutates nothing on disk;
  recompilation is delegated to an injected callable so the caller owns the
  toolchain. Debug artifacts go through a :class:`DebugSession` (or its null
  object), which is the only optional collaborator.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .symbols import extract_symbols, diff_symbols, _normalize as _normalize_symbol
from .structural import _strip_strings_and_comments, strip_comments


# ── Confidence tiers (Phase 7) ──────────────────────────────────────────────

CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW = "low"

_CONF_SCORE = {CONF_HIGH: 0.9, CONF_MEDIUM: 0.6, CONF_LOW: 0.3}

# Strategy → confidence tier. High-confidence repairs restore a definition the
# original CUDA source already contains, or apply a fix the compiler itself
# named — both are verifiable without judgement. Medium repairs relocate or
# substitute. Low repairs would change algorithm shape and are never applied
# automatically; they are reported for a stronger (LLM) pass.
_STRATEGY_CONFIDENCE = {
    "insert-include": CONF_HIGH,
    "restore-macro": CONF_HIGH,
    "restore-type": CONF_HIGH,
    "restore-helper": CONF_HIGH,
    "restore-variable": CONF_HIGH,
    "restore-declaration": CONF_HIGH,
    "apply-compiler-rename": CONF_HIGH,
    "qualify-namespace": CONF_HIGH,
    "repair-signature": CONF_MEDIUM,
    "relocate-scope": CONF_MEDIUM,
    "replace-hip-intrinsic": CONF_MEDIUM,
    "algorithm-rewrite": CONF_LOW,
    "kernel-restructure": CONF_LOW,
}


# ── Diagnostic parsing (Phase 1 input) ──────────────────────────────────────

# hipcc/clang diagnostic line: "<file>:<line>:<col>: error: <message>".
# The build-dir path prefix is already stripped upstream (verifier._shorten_
# error_line), but we tolerate its presence anyway.
_DIAG_LINE = re.compile(
    r'(?:^|\s)(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*'
    r'(?:fatal\s+)?(?P<severity>error|warning):\s*(?P<message>.*)$'
)

# Message-body patterns. Each names the single symbol the diagnostic is about,
# and (optionally) a compiler-suggested replacement.
_MSG_UNDECLARED = re.compile(
    r"use of undeclared identifier ['\"]?([A-Za-z_]\w*)['\"]?"
    r"(?:.*?;?\s*did you mean ['\"]?([A-Za-z_]\w*)['\"]?)?")
_MSG_UNKNOWN_TYPE = re.compile(
    r"unknown type name ['\"]?([A-Za-z_]\w*)['\"]?"
    r"(?:.*?did you mean ['\"]?([A-Za-z_:<>]+)['\"]?)?")
_MSG_NOT_A_TYPE = re.compile(
    r"['\"]?([A-Za-z_]\w*)['\"]? (?:does not name a type|was not declared in this scope)")
_MSG_NO_TEMPLATE = re.compile(r"no template named ['\"]?([A-Za-z_]\w*)['\"]?")
_MSG_NO_MEMBER = re.compile(
    r"no member named ['\"]?([A-Za-z_]\w*)['\"]? in ['\"]?([A-Za-z_][\w:<>]*)['\"]?")
_MSG_FILE_NOT_FOUND = re.compile(r"['\"]([^'\"]+\.[Hh](?:pp|xx|h)?|[^'\"]+\.cuh)['\"] file not found")
_MSG_NO_MATCHING_FN = re.compile(r"no matching function for call to ['\"]?([A-Za-z_]\w*)['\"]?")
_MSG_ARG_COUNT = re.compile(
    r"(too few|too many) arguments to (?:function call|function ['\"]?([A-Za-z_]\w*)['\"]?)")
_MSG_UNDEF_TEMPLATE = re.compile(
    r"implicit instantiation of undefined template ['\"]([A-Za-z_][\w:]*)")


@dataclass(frozen=True)
class Diagnostic:
    """One parsed compiler diagnostic, normalized to what a repair needs.

    ``kind`` is the semantic category the message maps to; ``symbol`` is the
    identifier/type/header the repair must resolve; ``suggestion`` is the
    compiler's own ``did you mean`` when present. Everything else is location
    context. Two diagnostics that differ only by line number normalize to the
    same ``key`` — that is what makes the learning cache stable.
    """
    raw: str
    kind: str            # undeclared | unknown-type | no-member | missing-include
                         # | no-matching-function | arg-count | undefined-template | unknown
    symbol: str = ""
    suggestion: str = ""
    owner: str = ""      # for no-member: the type the member was looked up in
    file: str = ""
    line: int = 0
    col: int = 0
    severity: str = "error"

    @property
    def key(self) -> str:
        """Location-independent identity, for the repair-learning cache."""
        return f"{self.kind}:{self.symbol}" + (f":{self.owner}" if self.owner else "")


def parse_diagnostics(diagnostics: List[str]) -> List[Diagnostic]:
    """Parse raw hipcc/clang ``error:`` strings into :class:`Diagnostic` records.

    Only ``error`` lines yield a repairable diagnostic; caret/context lines
    (``|``, ``^~~~``) and notes are ignored. Never raises — a line it cannot
    parse becomes a ``kind="unknown"`` record so the caller can still count it.
    """
    out: List[Diagnostic] = []
    seen_keys: set = set()
    for raw in diagnostics or []:
        if not raw or not raw.strip():
            continue
        text = raw.strip()
        m = _DIAG_LINE.search(text)
        if m:
            message = m.group("message").strip()
            severity = m.group("severity")
            file = m.group("file")
            line = int(m.group("line"))
            col = int(m.group("col"))
        else:
            # No `error:`/`warning:` prefix matched. A `note:`/`remark:` line
            # (context for another diagnostic) or a caret line is not itself an
            # error and must not be counted as one; only a genuine bare error
            # line (e.g. a linker "ld.lld: error: ...") reaches the parser.
            if re.search(r':\d+:\d+:\s*(?:note|remark):', text) or "error" not in text:
                continue
            message, severity, file, line, col = text, "error", "", 0, 0
        if severity != "error":
            continue

        diag = _classify_message(raw=text, message=message, file=file, line=line, col=col)
        # De-dupe by (kind, symbol, owner, line): the same undeclared identifier
        # reported at the same location twice is one repair, not two.
        dedup = (diag.kind, diag.symbol, diag.owner, diag.line)
        if dedup in seen_keys:
            continue
        seen_keys.add(dedup)
        out.append(diag)
    return out


def _classify_message(raw: str, message: str, file: str, line: int, col: int) -> Diagnostic:
    """Map a diagnostic message body to a semantic :class:`Diagnostic`."""
    def _mk(kind, symbol="", suggestion="", owner=""):
        return Diagnostic(raw=raw, kind=kind, symbol=symbol, suggestion=suggestion,
                          owner=owner, file=file, line=line, col=col, severity="error")

    m = _MSG_FILE_NOT_FOUND.search(message)
    if m:
        return _mk("missing-include", symbol=m.group(1))

    m = _MSG_NO_MEMBER.search(message)
    if m:
        return _mk("no-member", symbol=m.group(1), owner=m.group(2))

    m = _MSG_UNDECLARED.search(message)
    if m:
        return _mk("undeclared", symbol=m.group(1), suggestion=m.group(2) or "")

    m = _MSG_UNKNOWN_TYPE.search(message)
    if m:
        return _mk("unknown-type", symbol=m.group(1), suggestion=m.group(2) or "")

    m = _MSG_NO_TEMPLATE.search(message)
    if m:
        return _mk("unknown-type", symbol=m.group(1))

    m = _MSG_UNDEF_TEMPLATE.search(message)
    if m:
        # "Foo<int>" → resolve the template name "Foo".
        return _mk("undefined-template", symbol=m.group(1).split("<")[0].split("::")[-1])

    m = _MSG_NOT_A_TYPE.search(message)
    if m:
        return _mk("unknown-type", symbol=m.group(1))

    m = _MSG_NO_MATCHING_FN.search(message)
    if m:
        return _mk("no-matching-function", symbol=m.group(1))

    m = _MSG_ARG_COUNT.search(message)
    if m:
        return _mk("arg-count", symbol=(m.group(2) or ""))

    return _mk("unknown")


# ── CUDA source index (Phase 1 resolution target) ───────────────────────────

@dataclass(frozen=True)
class Definition:
    """The verbatim source text of one named definition in the CUDA original.

    ``kind`` classifies it (macro/type/helper/kernel/variable/include); ``text``
    is the exact slice of the original source — restoring it into the HIP unit
    is a copy, never a paraphrase.
    """
    name: str
    kind: str            # macro | type | helper | kernel | variable | include
    text: str
    start: int = 0
    end: int = 0


def _iter_matches_with_body(code: str, header_re: "re.Pattern") -> List[Tuple[str, int, int]]:
    """For each header match, brace-match to the closing ``}`` (and trailing ``;``).

    Returns ``(name, start, end)`` spans over the ORIGINAL code (offsets valid
    against the un-stripped text). Brace matching runs on a comment/string-blanked
    copy so a ``}`` inside a literal or comment does not close the body early.
    """
    blanked = _strip_strings_and_comments(code)
    spans: List[Tuple[str, int, int]] = []
    for m in header_re.finditer(blanked):
        name = m.group(1)
        brace = blanked.find("{", m.start())
        if brace < 0:
            continue
        depth, i, n = 0, brace, len(blanked)
        end = -1
        while i < n:
            c = blanked[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end < 0:
            continue
        # Swallow a trailing ';' (struct/class/enum definitions have one).
        j = end
        while j < n and blanked[j] in " \t":
            j += 1
        if j < n and blanked[j] == ";":
            end = j + 1
        spans.append((name, m.start(), end))
    return spans


class CudaSourceIndex:
    """Name → verbatim definition, built once from the original CUDA source.

    This is the "recover from the original, don't invent" half of the engine.
    Every lookup returns text that literally exists in the ``.cu`` file, so a
    restoration can never introduce a construct the author did not write.
    """

    # A macro definition, including line-continued bodies.
    _MACRO = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)', re.MULTILINE)
    # struct / class / union / enum definitions (with a body).
    _RECORD = re.compile(
        r'^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?'
        r'(?:typedef[ \t]+)?(?:struct|class|union|enum(?:[ \t]+class)?)[ \t]+'
        r'([A-Za-z_]\w*)\b[^;{]*\{',
        re.MULTILINE)
    # __device__ / __host__ / __global__ / plain helper function DEFINITIONS.
    _FUNC = re.compile(
        r'^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?'
        r'(?:(?:static|inline|extern|__forceinline__|__device__|__host__|__global__|constexpr)[ \t]+)*'
        r'[A-Za-z_][\w:<>,*&\t ]*?\b([A-Za-z_]\w*)[ \t]*\([^;{]*\)[ \t]*(?:const[ \t]*)?\{',
        re.MULTILINE)
    # typedef … NAME;  and  using NAME = …;
    _TYPEDEF = re.compile(r'^[ \t]*typedef[ \t]+[^;{]*?\b([A-Za-z_]\w*)[ \t]*;', re.MULTILINE)
    _USING = re.compile(r'^[ \t]*using[ \t]+([A-Za-z_]\w*)[ \t]*=[ \t]*[^;]+;', re.MULTILINE)
    # A file-scope (constant/global) variable definition.
    _CONSTANT = re.compile(
        r'^[ \t]*(?:(?:static|extern|const|constexpr)[ \t]+)*'
        r'(?:__constant__|__device__)[ \t]+[\w:<>,*&\t ]+?\b([A-Za-z_]\w*)[ \t]*'
        r'(?:\[[^\]]*\])?[ \t]*(?:=[^;]*)?;', re.MULTILINE)
    _INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"][^>"]+[>"])', re.MULTILINE)

    def __init__(self, cuda_source: str):
        self.source = cuda_source or ""
        self._blanked = _strip_strings_and_comments(self.source)
        self._directives = strip_comments(self.source)
        self._defs: Dict[str, Definition] = {}
        self._includes: List[str] = []
        self._build()

    def _build(self) -> None:
        # Records (struct/class/enum/union) and functions carry a brace body.
        for name, start, end in _iter_matches_with_body(self.source, self._RECORD):
            self._defs.setdefault(name, Definition(name, "type", self.source[start:end], start, end))
        for name, start, end in _iter_matches_with_body(self.source, self._FUNC):
            if name in _CONTROL_KEYWORDS:
                continue
            # A __global__ definition is a kernel; everything else a helper.
            header = self.source[start:self.source.find("{", start)]
            kind = "kernel" if "__global__" in header else "helper"
            self._defs.setdefault(name, Definition(name, kind, self.source[start:end], start, end))

        # Macros (line-continuation aware).
        for m in self._MACRO.finditer(self._directives):
            name = m.group(1)
            if name in self._defs:
                continue
            text = self._read_macro(m.start())
            self._defs[name] = Definition(name, "macro", text, m.start(), m.start() + len(text))

        # typedef / using aliases.
        for rx, kind in ((self._TYPEDEF, "type"), (self._USING, "type")):
            for m in rx.finditer(self._blanked):
                name = m.group(1)
                if name not in self._defs:
                    self._defs[name] = Definition(name, kind, self.source[m.start():m.end()],
                                                  m.start(), m.end())

        # File-scope constant/device variables.
        for m in self._CONSTANT.finditer(self._blanked):
            name = m.group(1)
            if name not in self._defs:
                self._defs[name] = Definition(name, "variable", self.source[m.start():m.end()],
                                              m.start(), m.end())

        # Includes (from comment-stripped text so the payload survives).
        self._includes = [m.group(1) for m in self._INCLUDE.finditer(self._directives)]

    def _read_macro(self, start: int) -> str:
        """Read a ``#define`` from *start*, following ``\\``-newline continuations."""
        lines = self.source[start:].splitlines(keepends=True)
        collected: List[str] = []
        for ln in lines:
            collected.append(ln)
            if not ln.rstrip("\n").rstrip("\r").endswith("\\"):
                break
        return "".join(collected).rstrip("\n").rstrip("\r")

    # ── Public lookup ────────────────────────────────────────────────────
    def lookup(self, name: str) -> Optional[Definition]:
        return self._defs.get(name)

    def includes(self) -> List[str]:
        return list(self._includes)

    def names(self) -> List[str]:
        return sorted(self._defs)

    def find_similar(self, name: str, threshold: float = 0.82) -> Optional[str]:
        """The closest defined name under CUDA↔HIP-normalized comparison.

        Used to detect a rename: an identifier absent from CUDA under its exact
        spelling but present under a normalized one (``warpReduce`` vs
        ``wavefront_reduce``). Deterministic — ties break on sort order.
        """
        target = _normalize_symbol(name)
        best, best_score = None, 0.0
        for cand in sorted(self._defs):
            score = difflib.SequenceMatcher(None, target, _normalize_symbol(cand)).ratio()
            if score > best_score:
                best, best_score = cand, score
        return best if best is not None and best_score >= threshold else None


_CONTROL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "else", "do", "sizeof",
    "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "dim3",
    "case", "default", "operator",
})


# ── Precise definition detection (shared by HipUnit + the engine) ───────────
#
# These distinguish a *definition* of a name from a mere *use* of it. The
# distinction matters: ``__global__ void k(..., Params p)`` USES the type
# ``Params`` as a parameter — a loose "is there a record keyword before the
# name?" scan false-matches it and then refuses to restore the genuinely
# missing struct. Each checker below anchors on the shape only a definition has.

def _defines_macro(code: str, name: str) -> bool:
    return bool(re.search(r'^[ \t]*#[ \t]*define[ \t]+' + re.escape(name) + r'\b',
                          strip_comments(code), re.MULTILINE))


def _defines_type(code: str, name: str) -> bool:
    blanked = _strip_strings_and_comments(code)
    esc = re.escape(name)
    return bool(
        re.search(r'\b(?:struct|class|union|enum(?:[ \t]+class)?)[ \t]+' + esc + r'\b[ \t]*[:{]', blanked)
        or re.search(r'\btypedef\b[^;{]*\b' + esc + r'[ \t]*;', blanked)
        or re.search(r'\busing[ \t]+' + esc + r'[ \t]*=', blanked))


def _defines_function(code: str, name: str) -> bool:
    # A function DEFINITION has a body: ``name(...) {`` (optionally ``const``).
    # A call site ``name(args);`` ends in ``;`` and never reaches the ``{``.
    blanked = _strip_strings_and_comments(code)
    return bool(re.search(r'\b' + re.escape(name) + r'[ \t]*\([^;{]*\)[ \t]*(?:const[ \t]*)?\{',
                          blanked))


def _defines_variable(code: str, name: str) -> bool:
    blanked = _strip_strings_and_comments(code)
    return bool(re.search(
        r'(?:__constant__|__device__)[ \t]+[\w:<>,*&\t ]+?\b' + re.escape(name)
        + r'\b[ \t]*(?:\[[^\]]*\])?[ \t]*(?:=|;)', blanked))


# ── HIP translation-unit analysis (Phase 1 "current/parent scope") ──────────

# Device-only builtins that are meaningless outside a __global__/__device__ body.
_DEVICE_BUILTINS = ("threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize",
                    "__syncthreads", "__shared__", "__shfl", "__ballot")


@dataclass(frozen=True)
class Region:
    """A function body in the HIP unit, with its device/host nature and span."""
    name: str
    is_device: bool      # __global__ or __device__ — device execution space
    start: int
    end: int


class HipUnit:
    """A read-only view of the generated HIP source: what it defines and where.

    The engine consults this first (Phase 1 order: current scope → parent scope)
    before falling back to the CUDA original: a symbol that IS defined in the HIP
    unit is not "missing", so the diagnostic is a scope/spelling problem, not a
    dropped definition.
    """

    _REGION = CudaSourceIndex._FUNC  # same function-definition shape

    def __init__(self, hip_source: str):
        self.source = hip_source or ""
        self._blanked = _strip_strings_and_comments(self.source)
        self._table = extract_symbols(self.source)
        self._defined = self._table.all_names() | set(self._table.macros)
        self._regions = self._build_regions()

    def _build_regions(self) -> List[Region]:
        regions: List[Region] = []
        for name, start, end in _iter_matches_with_body(self.source, self._REGION):
            if name in _CONTROL_KEYWORDS:
                continue
            header = self.source[start:self.source.find("{", start)]
            is_device = "__global__" in header or "__device__" in header
            regions.append(Region(name, is_device, start, end))
        return regions

    def defines(self, name: str) -> bool:
        """True if the HIP unit already defines *name* (function/macro/type/var).

        Uses precise definition detection so a name that only *appears* as a
        parameter type or a call argument is NOT mistaken for a definition —
        that mistake would suppress a genuinely-needed restoration.
        """
        if name in self._defined:
            return True
        return (_defines_type(self.source, name)
                or _defines_function(self.source, name)
                or _defines_variable(self.source, name))

    def includes(self) -> List[str]:
        return list(self._table.includes)

    def region_at(self, offset: int) -> Optional[Region]:
        for r in self._regions:
            if r.start <= offset < r.end:
                return r
        return None

    def uses_device_builtin_at_host_scope(self) -> List[Tuple[str, int]]:
        """Device builtins that appear outside any device region (Phase 4).

        Returns ``(builtin, line)`` for each occurrence at file/host scope — a
        ``threadIdx`` in ``main`` is a scope defect the compiler reports as an
        undeclared identifier, but its true cause is that device code leaked
        into a host function.
        """
        out: List[Tuple[str, int]] = []
        for builtin in _DEVICE_BUILTINS:
            for m in re.finditer(r'\b' + re.escape(builtin) + r'\b', self._blanked):
                r = self.region_at(m.start())
                if r is None or not r.is_device:
                    line = self.source.count("\n", 0, m.start()) + 1
                    out.append((builtin, line))
        return sorted(set(out))


# ── Repair records ──────────────────────────────────────────────────────────

@dataclass
class RepairPatch:
    """One minimal, localized edit and the full provenance behind it (Phase 8)."""
    strategy: str
    symbol: str
    root_cause: str
    confidence: str
    confidence_score: float
    diagnostic: str
    cuda_snippet: str = ""
    hip_snippet_before: str = ""
    inserted_text: str = ""
    diff: str = ""
    line: int = 0
    applied: bool = False
    compile_errors_before: int = 0
    compile_errors_after: int = 0
    accepted: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "diagnostic": self.diagnostic,
            "cuda_snippet": self.cuda_snippet,
            "hip_snippet_before": self.hip_snippet_before,
            "inserted_text": self.inserted_text,
            "diff": self.diff,
            "line": self.line,
            "applied": self.applied,
            "accepted": self.accepted,
            "compile_errors_before": self.compile_errors_before,
            "compile_errors_after": self.compile_errors_after,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class RepairResult:
    """Outcome of a repair session over one HIP unit."""
    patched_code: str
    patches: List[RepairPatch] = field(default_factory=list)
    unresolved: List[Dict] = field(default_factory=list)
    errors_before: int = 0
    errors_after: int = 0
    iterations: int = 0
    changed: bool = False

    @property
    def accepted_patches(self) -> List[RepairPatch]:
        return [p for p in self.patches if p.accepted]

    def to_dict(self) -> Dict:
        return {
            "changed": self.changed,
            "errors_before": self.errors_before,
            "errors_after": self.errors_after,
            "iterations": self.iterations,
            "patches_accepted": len(self.accepted_patches),
            "patches_attempted": len(self.patches),
            "patches": [p.to_dict() for p in self.patches],
            "unresolved": self.unresolved,
        }


# ── Resolution (Phase 1) + cause classification ─────────────────────────────

@dataclass(frozen=True)
class Resolution:
    """How a missing symbol was resolved, and why it went missing."""
    strategy: str            # one of _STRATEGY_CONFIDENCE keys
    root_cause: str          # omitted | renamed | hip-replacement-mistake
                             # | removed-during-extraction | wrong-scope | ...
    definition: Optional[Definition] = None
    inserted_text: str = ""
    note: str = ""


# A missing standard header we can supply for a well-known symbol used bare.
_STDLIB_SYMBOL_HEADERS = {
    "printf": "<cstdio>", "fprintf": "<cstdio>", "sprintf": "<cstdio>",
    "malloc": "<cstdlib>", "free": "<cstdlib>", "exit": "<cstdlib>",
    "memcpy": "<cstring>", "memset": "<cstring>", "strlen": "<cstring>",
    "sqrt": "<cmath>", "fabs": "<cmath>", "pow": "<cmath>", "expf": "<cmath>",
    "std::vector": "<vector>", "std::string": "<string>",
    "std::cout": "<iostream>", "std::endl": "<iostream>",
}

# A CUDA header whose HIP equivalent is a single known include.
_CUDA_HEADER_TO_HIP = {
    "cuda_runtime.h": "<hip/hip_runtime.h>",
    "cuda_runtime_api.h": "<hip/hip_runtime.h>",
    "cuda.h": "<hip/hip_runtime.h>",
    "device_launch_parameters.h": "<hip/hip_runtime.h>",
    "curand_kernel.h": "<hiprand/hiprand_kernel.h>",
}



def _unified_diff(before: str, after: str) -> str:
    """A deterministic unified diff. Stable filenames, no clocks."""
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="hip_before", tofile="hip_after", lineterm="", n=2))


# ── The engine ──────────────────────────────────────────────────────────────

class SemanticRepairEngine:
    """Convert compiler diagnostics into minimal, semantics-preserving patches.

    Lifecycle::

        eng = SemanticRepairEngine(cuda_source, hip_source, debug_session=dbg)
        result = eng.repair(diagnostics, recompile=verifier_callable)
        if result.changed:
            use(result.patched_code)

    ``recompile`` is ``(hip_source) -> (ok: bool, errors: List[str])``. When
    provided, each candidate patch is compiled and kept only if it does **not**
    increase the error count — so the engine can never make a build worse. When
    omitted, all high-confidence *additive* patches are applied in one pass
    (safe: they only restore definitions the unit was missing).

    The ``cache`` maps a location-independent diagnostic key to the strategy
    that last fixed it, and is shared across kernels in a session (Phase 9) so a
    repair learned on one file is tried first on the next.
    """

    def __init__(self, cuda_source: str, hip_source: str,
                 debug_session=None, cache: Optional[Dict[str, str]] = None):
        self.cuda_source = cuda_source or ""
        self.hip_source = hip_source or ""
        self.index = CudaSourceIndex(self.cuda_source)
        self.debug = debug_session  # DebugSession or None; may also be a null obj
        self.cache: Dict[str, str] = cache if cache is not None else {}
        self._symbol_diff = diff_symbols(self.cuda_source, self.hip_source)

    # ── Phase 1: resolve a single diagnostic ─────────────────────────────
    def _resolve(self, diag: Diagnostic, unit: HipUnit) -> Optional[Resolution]:
        """Determine how to restore the semantic information the diagnostic names.

        Search order mirrors the mission's Phase 1: the HIP unit's own scopes
        first (already-defined => not missing), then the original CUDA source,
        then rename candidates, then well-known headers. Returns ``None`` when no
        deterministic recovery exists — the diagnostic is then left to the LLM.
        """
        # Missing include: map a CUDA header to its HIP form, or supply a
        # stdlib header for a bare well-known symbol.
        if diag.kind == "missing-include":
            header_base = diag.symbol.split("/")[-1]
            hip_inc = _CUDA_HEADER_TO_HIP.get(header_base)
            if hip_inc:
                return Resolution("insert-include", "hip-replacement-mistake",
                                  inserted_text=f"#include {hip_inc}",
                                  note=f"CUDA header {diag.symbol} -> {hip_inc}")
            # A project ".cuh" that the port dropped: the definitions it held
            # cannot be conjured, so this is left unresolved on purpose.
            return None

        symbol = diag.symbol
        if not symbol:
            return None

        # If the HIP unit already defines it, the problem is scope/spelling, not
        # a dropped definition — handled by the scope pass, not here.
        if unit.defines(symbol):
            return None

        # A compiler-verified rename: the diagnostic itself says "did you mean
        # Y", and Y is defined in the unit. Applying the compiler's own
        # suggestion is high-confidence and needs no CUDA lookup.
        if diag.suggestion and unit.defines(diag.suggestion):
            return Resolution("apply-compiler-rename", "renamed",
                              inserted_text=diag.suggestion,
                              note=f"{symbol} -> {diag.suggestion} (compiler-suggested)")

        # Recover the definition from the original CUDA source — the heart of
        # the engine. The symbol existed; translation/extraction dropped it.
        cuda_def = self.index.lookup(symbol)
        if cuda_def is not None:
            strategy = {
                "macro": "restore-macro",
                "type": "restore-type",
                "helper": "restore-helper",
                "kernel": "restore-helper",
                "variable": "restore-variable",
            }.get(cuda_def.kind, "restore-declaration")
            # "removed-during-extraction" when the symbol is in the CUDA symbol
            # table's removed set; "omitted" otherwise. Both restore identically.
            cause = ("removed-during-extraction"
                     if symbol in self._symbol_diff.get("removed_symbols", [])
                     else "omitted")
            return Resolution(strategy, cause, definition=cuda_def,
                              inserted_text=cuda_def.text)

        # A stdlib symbol used without its header (translation dropped the
        # include, not the definition).
        std_header = _STDLIB_SYMBOL_HEADERS.get(symbol)
        if std_header:
            return Resolution("insert-include", "omitted",
                              inserted_text=f"#include {std_header}",
                              note=f"{symbol} needs {std_header}")

        # A near-name match in CUDA suggests the port renamed it wrongly, but we
        # cannot rewrite call sites deterministically without risking meaning —
        # report as a rename hypothesis for the stronger pass, do not apply.
        similar = self.index.find_similar(symbol)
        if similar:
            return Resolution("apply-compiler-rename", "renamed",
                              note=f"{symbol} ~ CUDA symbol {similar} (unapplied hypothesis)")

        return None

    # ── Phase 3: turn a resolution into a minimal patch on the HIP text ──
    def _build_patch(self, diag: Diagnostic, res: Resolution,
                     current: str) -> Optional[RepairPatch]:
        conf = _STRATEGY_CONFIDENCE.get(res.strategy, CONF_LOW)
        patch = RepairPatch(
            strategy=res.strategy, symbol=diag.symbol, root_cause=res.root_cause,
            confidence=conf, confidence_score=_CONF_SCORE[conf],
            diagnostic=diag.raw, line=diag.line,
            cuda_snippet=(res.definition.text if res.definition else res.note),
            inserted_text=res.inserted_text,
        )

        if res.strategy == "apply-compiler-rename" and res.inserted_text:
            new_code = self._apply_rename(current, diag.symbol, res.inserted_text)
            if new_code == current:
                return None
        elif res.strategy == "insert-include":
            if res.inserted_text.split()[-1] in current:
                return None  # already present
            new_code = self._insert_include(current, res.inserted_text)
        elif res.inserted_text:
            # Additive restoration of a macro/type/helper/variable/decl.
            if self._already_present(current, diag.symbol, res):
                return None
            new_code = self._insert_definition(current, res.inserted_text)
        else:
            return None  # unapplied hypothesis — reported, not patched

        if new_code == current:
            return None
        patch.hip_snippet_before = self._context_snippet(current, diag.symbol)
        patch.diff = _unified_diff(current, new_code)
        patch._new_code = new_code  # type: ignore[attr-defined]
        return patch

    # ── Insertion helpers (smallest legal edit) ──────────────────────────
    @staticmethod
    def _header_insert_offset(code: str) -> int:
        """Byte offset just after the leading ``#include``/``#define`` block.

        Restored macros, types, helpers and variables land here: after the
        includes (so they can use them) and after ``#define WAVEFRONT_SIZE``,
        before the first function that uses them. This keeps the edit minimal
        and legal without reordering anything the model wrote.
        """
        last = 0
        for m in re.finditer(r'^[ \t]*#[ \t]*(?:include|define|pragma)\b.*$',
                             code, re.MULTILINE):
            # Follow a macro's line continuations to its true end.
            end = m.end()
            while code[m.start():end].rstrip().endswith("\\"):
                nl = code.find("\n", end)
                if nl < 0:
                    break
                end = code.find("\n", nl + 1)
                if end < 0:
                    end = len(code)
                    break
            last = max(last, end)
        if last == 0:
            return 0
        nl = code.find("\n", last)
        return (nl + 1) if nl >= 0 else len(code)

    def _insert_definition(self, code: str, text: str) -> str:
        pos = self._header_insert_offset(code)
        block = text.rstrip() + "\n"
        marker = "// [semantic-repair] restored from original CUDA source\n"
        return code[:pos] + "\n" + marker + block + code[pos:]

    def _insert_include(self, code: str, include_line: str) -> str:
        includes = list(re.finditer(r'^[ \t]*#[ \t]*include\b.*$', code, re.MULTILINE))
        if includes:
            pos = includes[-1].end()
            nl = code.find("\n", pos)
            insert_at = (nl + 1) if nl >= 0 else len(code)
            return code[:insert_at] + include_line + "\n" + code[insert_at:]
        return include_line + "\n" + code

    @staticmethod
    def _apply_rename(code: str, old: str, new: str) -> str:
        """Replace whole-word uses of *old* with *new*.

        This is deliberately conservative — a rename is only reached when the
        compiler itself proposed it (``did you mean``) and the target is defined
        in the unit, so the substitution is compiler-verified.
        """
        return re.sub(r'\b' + re.escape(old) + r'\b', new, code)

    @staticmethod
    def _already_present(code: str, symbol: str, res: Resolution) -> bool:
        """Guard against restoring a definition the unit already has.

        Additive patches must never create a duplicate definition (hipcc reports
        that as a redefinition, a strictly worse state). Kind-specific and
        precise: a name that merely appears as a parameter type or call argument
        is NOT a definition and must not block its own restoration.
        """
        kind = res.definition.kind if res.definition else ""
        if kind == "macro":
            return _defines_macro(code, symbol)
        if kind == "type":
            return _defines_type(code, symbol)
        if kind in ("helper", "kernel"):
            return _defines_function(code, symbol)
        if kind == "variable":
            return _defines_variable(code, symbol)
        # Unknown kind: fall back to any precise definition shape.
        return (_defines_type(code, symbol) or _defines_function(code, symbol)
                or _defines_variable(code, symbol) or _defines_macro(code, symbol))

    @staticmethod
    def _context_snippet(code: str, symbol: str, width: int = 3) -> str:
        lines = code.splitlines()
        for i, ln in enumerate(lines):
            if re.search(r'\b' + re.escape(symbol) + r'\b', ln):
                lo, hi = max(0, i - width), min(len(lines), i + width + 1)
                return "\n".join(lines[lo:hi])
        return ""

    # ── Phase 4: scope repair (report device-on-host leakage) ────────────
    def _scope_findings(self, unit: HipUnit) -> List[Dict]:
        findings = []
        for builtin, line in unit.uses_device_builtin_at_host_scope():
            findings.append({
                "kind": "wrong-scope",
                "symbol": builtin,
                "line": line,
                "root_cause": "device builtin used outside a __global__/__device__ region",
                "confidence": CONF_LOW,
                "note": "requires relocating code into a device function — "
                        "reported for a stronger pass, not auto-applied",
            })
        return findings

    # ── Phase 6: compiler-guided repair loop ─────────────────────────────
    def repair(self, diagnostics: List[str],
               recompile: Optional[Callable[[str], Tuple[bool, List[str]]]] = None,
               max_passes: int = 6) -> RepairResult:
        """Repair *diagnostics* against the HIP unit and return a result.

        One patch per pass, most-confident first; recompile after each (when a
        compiler is available) and keep the patch only if it did not increase
        the error count. Loops until no diagnostic yields a new applicable patch
        or the pass budget is spent — never regenerating the whole file.
        """
        t_engine = time.perf_counter()
        current = self.hip_source
        errors_now = list(diagnostics or [])
        errors_before = len(parse_diagnostics(errors_now))
        result = RepairResult(patched_code=current, errors_before=errors_before,
                              errors_after=errors_before)

        applied_keys: set = set()
        passes = 0

        while passes < max_passes:
            passes += 1
            diags = parse_diagnostics(errors_now)
            unit = HipUnit(current)

            # Build every candidate patch this pass, then act on the highest
            # confidence one (deterministic tie-break on line, then symbol).
            candidates: List[Tuple[RepairPatch, Diagnostic]] = []
            for diag in diags:
                if diag.key in applied_keys:
                    continue
                res = self._resolve(diag, unit)
                if res is None:
                    continue
                patch = self._build_patch(diag, res, current)
                if patch is not None:
                    candidates.append((patch, diag))

            if not candidates:
                break

            candidates.sort(key=lambda pd: (-pd[0].confidence_score, pd[1].line,
                                            pd[1].symbol, pd[0].strategy))
            patch, diag = candidates[0]
            t0 = time.perf_counter()
            new_code = patch._new_code  # type: ignore[attr-defined]
            patch.applied = True
            patch.compile_errors_before = len(parse_diagnostics(errors_now))

            if recompile is not None:
                ok, new_errors = recompile(new_code)
                new_count = len(parse_diagnostics(new_errors))
                patch.compile_errors_after = new_count
                # Accept only a strict non-increase; a patch that adds errors is
                # discarded and its diagnostic left for the LLM path.
                if ok or new_count <= patch.compile_errors_before:
                    patch.accepted = True
                    current = new_code
                    errors_now = new_errors
                else:
                    patch.accepted = False
            else:
                # No compiler: additive high-confidence restorations only.
                if patch.confidence == CONF_HIGH:
                    patch.accepted = True
                    current = new_code
                patch.compile_errors_after = patch.compile_errors_before

            patch.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            applied_keys.add(diag.key)
            result.patches.append(patch)
            self._record_patch(patch, diag)
            if patch.accepted:
                self.cache[diag.key] = patch.strategy

            if recompile is not None and not errors_now:
                break  # clean compile — done

        unit = HipUnit(current)
        result.unresolved = self._collect_unresolved(errors_now, unit) + self._scope_findings(unit)
        result.patched_code = current
        result.errors_after = len(parse_diagnostics(errors_now))
        result.iterations = passes
        result.changed = current != self.hip_source and any(p.accepted for p in result.patches)
        self._record_session(result, (time.perf_counter() - t_engine) * 1000.0)
        return result

    def _collect_unresolved(self, errors_now: List[str], unit: HipUnit) -> List[Dict]:
        out = []
        for diag in parse_diagnostics(errors_now):
            if self._resolve(diag, unit) is None:
                out.append({"kind": diag.kind, "symbol": diag.symbol,
                            "diagnostic": diag.raw, "line": diag.line,
                            "reason": "no deterministic recovery from CUDA source"})
        return out

    # ── Phase 8: deterministic debug artifacts ───────────────────────────
    def _record_patch(self, patch: RepairPatch, diag: Diagnostic) -> None:
        dbg = self.debug
        if dbg is None or not getattr(dbg, "enabled", False):
            return
        try:
            dbg.log_semantic_repair(patch.to_dict(), diagnostic=diag.raw)
        except AttributeError:
            # Older DebugSession without the dedicated hook: fall back to the
            # generic patch log so nothing is lost.
            try:
                dbg.log_patch(before=patch.hip_snippet_before,
                              after=patch.hip_snippet_before + "\n" + patch.inserted_text,
                              rationale=f"[semantic-repair:{patch.strategy}] "
                                        f"{patch.root_cause} — {patch.symbol}",
                              confidence=patch.confidence_score,
                              source_label="semantic-repair")
            except Exception:
                pass
        except Exception:
            pass

    def _record_session(self, result: RepairResult, elapsed_ms: float) -> None:
        dbg = self.debug
        if dbg is None or not getattr(dbg, "enabled", False):
            return
        try:
            dbg.write_json("11_patches", "semantic_repair_session",
                           {**result.to_dict(), "engine_elapsed_ms": round(elapsed_ms, 2)})
        except Exception:
            pass


__all__ = [
    "SemanticRepairEngine", "RepairResult", "RepairPatch",
    "Diagnostic", "parse_diagnostics", "CudaSourceIndex", "HipUnit",
    "CONF_HIGH", "CONF_MEDIUM", "CONF_LOW",
]
