"""Symbol inventory and CUDA→HIP diff.

The structural validator already answers "did the port drop a function?" — but
it answers it as a boolean, scoped to top-level definitions, and it throws the
inventory away. A post-mortem needs the inventory itself: which kernels the
original had, which the port has, what was renamed, what appeared from nowhere.

This module extracts a *symbol table* from a translation unit and diffs two of
them. It is regex-based and knows it: a symbol that a macro assembles, or a
kernel inside a template specialization, will be missed. Everything it does
report, it reports deterministically — sorted, stable, no clocks.

Renaming is inferred, never asserted. Two symbols count as a rename candidate
only when one vanished, one appeared, and their names are close enough under a
normalized comparison (``warpReduce`` → ``wavefrontReduce``). The report calls
these ``rename_candidates``, not ``renames``, because a regex cannot tell a
rename from a coincidence.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set


# A __global__ kernel definition (not a prototype — a prototype ends in ';').
_KERNEL_DEF = re.compile(
    r'^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?'
    r'(?:static[ \t]+|extern[ \t]+"C"[ \t]+)*'
    r'__global__[ \t]+\w[\w:<>,* \t]*?'
    r'\b([A-Za-z_]\w*)[ \t]*\([^;{]*\)[ \t]*\{',
    re.MULTILINE,
)

# A __device__ / __host__ helper definition.
_HELPER_DEF = re.compile(
    r'^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?'
    r'(?:static[ \t]+|inline[ \t]+)*'
    r'(?:__device__|__host__)(?:[ \t]+(?:__forceinline__|inline|static))*[ \t]+'
    r'\w[\w:<>,* \t]*?'
    r'\b([A-Za-z_]\w*)[ \t]*\([^;{]*\)[ \t]*\{',
    re.MULTILINE,
)

# An ordinary host function definition, device qualifiers absent.
_HOST_FUNC_DEF = re.compile(
    r'^[ \t]*(?!.*(?:__global__|__device__))'
    r'(?:static[ \t]+|inline[ \t]+|extern[ \t]+)*'
    r'(?:[A-Za-z_]\w*(?:[ \t]*[*&])?[ \t]+)+'
    r'([A-Za-z_]\w*)[ \t]*\([^;{]*\)[ \t]*(?:const[ \t]*)?\{',
    re.MULTILINE,
)

_MACRO_DEF = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(.*)$', re.MULTILINE)

_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"][^>"]+[>"])', re.MULTILINE)

# Control-flow keywords the host-function regex would otherwise capture.
_NOT_FUNCTIONS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "else", "do",
    "sizeof", "static_cast", "dim3", "case", "default",
})

# Host-level symbols that a DEVICE_SUBSET port would never generate.
_HOST_ONLY_SYMBOLS = frozenset({
    "main",
})


def _is_host_only(name: str) -> bool:
    """Return True when *name* should not be expected in a DEVICE_SUBSET port.

    DEVICE_SUBSET targets only ``__global__`` kernels and ``__device__``
    helpers.  Host-level definitions such as ``main``, ``shuffle_*_test``, and
    similar driver/test functions are only needed when the full program is
    compiled as a standalone binary.
    """
    return (
        name in _HOST_ONLY_SYMBOLS
        or (name.startswith("shuffle_") and name.endswith("_test"))
    )


def _strip(code: str) -> str:
    """Blank comments and string literals, preserving offsets.

    Reuses the structural validator's implementation so both modules agree on
    what "code" means — a symbol mentioned only in a comment is not a symbol.
    """
    from .structural import _strip_strings_and_comments
    return _strip_strings_and_comments(code)


def _strip_comments(code: str) -> str:
    """Blank comments but keep string literals.

    Preprocessor directives carry their payload inside a string literal
    (``#include "foo.cuh"``, ``#define MSG "hi"``). Scanning them in text where
    literals have been blanked yields ``#include "        "`` — a whitespace
    artifact reported as an include. Directives are read from here instead.
    """
    from .structural import strip_comments
    return strip_comments(code)


@dataclass
class SymbolTable:
    """Every symbol one translation unit defines, by kind."""
    kernels: List[str] = field(default_factory=list)
    helpers: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    macros: Dict[str, str] = field(default_factory=dict)
    includes: List[str] = field(default_factory=list)

    def all_names(self) -> Set[str]:
        return set(self.kernels) | set(self.helpers) | set(self.functions)

    def to_dict(self) -> Dict:
        return {
            "kernels": sorted(self.kernels),
            "helpers": sorted(self.helpers),
            "functions": sorted(self.functions),
            "macros": dict(sorted(self.macros.items())),
            "includes": sorted(self.includes),
            "counts": {
                "kernels": len(self.kernels),
                "helpers": len(self.helpers),
                "functions": len(self.functions),
                "macros": len(self.macros),
                "includes": len(self.includes),
            },
        }


def extract_symbols(code: str) -> SymbolTable:
    """Build a :class:`SymbolTable` from a CUDA or HIP translation unit.

    A name is assigned to exactly one kind, most-specific first: a
    ``__global__`` definition is a kernel and never also a function.
    """
    if not code or not code.strip():
        return SymbolTable()

    cleaned = _strip(code)

    kernels = [m.group(1) for m in _KERNEL_DEF.finditer(cleaned)]
    helpers = [m.group(1) for m in _HELPER_DEF.finditer(cleaned)]

    claimed = set(kernels) | set(helpers)
    functions = [
        m.group(1) for m in _HOST_FUNC_DEF.finditer(cleaned)
        if m.group(1) not in _NOT_FUNCTIONS and m.group(1) not in claimed
    ]

    # Directives are read from comment-stripped (NOT literal-stripped) text.
    # `#define X 64  // comment` must yield a body of "64", and
    # `#include "foo.cuh"` must yield `"foo.cuh"` rather than `"        "`.
    directives = _strip_comments(code)
    macros: Dict[str, str] = {}
    for m in _MACRO_DEF.finditer(directives):
        macros[m.group(1)] = m.group(2).strip()

    includes = [m.group(1) for m in _INCLUDE.finditer(directives)]

    # Deduplicate while keeping a deterministic (sorted) order. A duplicate
    # definition is the static analyzer's finding to report, not ours.
    return SymbolTable(
        kernels=sorted(set(kernels)),
        helpers=sorted(set(helpers)),
        functions=sorted(set(functions)),
        macros=macros,
        includes=sorted(set(includes)),
    )


def _normalize(name: str) -> str:
    """Fold the spellings a CUDA→HIP port legitimately changes.

    ``warpReduce`` and ``wavefront_reduce`` should compare equal-ish, so a
    rename is proposed rather than reported as one removal plus one addition.
    """
    n = name.lower().replace("_", "")
    for cuda, hip in (("warp", "wave"), ("wavefront", "wave"), ("cuda", "hip")):
        n = n.replace(cuda, hip)
    return n


def _rename_candidates(removed: Set[str], added: Set[str],
                       threshold: float = 0.72) -> List[Dict]:
    """Pair a removed symbol with an added one when the names nearly match.

    Greedy and deterministic: candidates are considered in sorted order and each
    symbol is consumed at most once. The result is a *hypothesis* — the caller
    must not treat a pairing as evidence the port is correct.
    """
    pairs: List[Dict] = []
    unmatched_added = sorted(added)
    for old in sorted(removed):
        best, best_score = None, 0.0
        for new in unmatched_added:
            score = difflib.SequenceMatcher(
                None, _normalize(old), _normalize(new)).ratio()
            if score > best_score:
                best, best_score = new, score
        if best is not None and best_score >= threshold:
            pairs.append({"from": old, "to": best, "similarity": round(best_score, 4)})
            unmatched_added.remove(best)
    return pairs


def diff_symbols(cuda_source: str, hip_source: str,
                 port_mode: str = "FULL") -> Dict:
    """Diff the symbol tables of the original CUDA and the generated HIP.

    Parameters
    ----------
    cuda_source, hip_source:
        Source text of the original CUDA and generated HIP.
    port_mode:
        ``"FULL"`` (default) — report all differences as-is.
        ``"DEVICE_SUBSET"`` — exclude host-level symbols (``main``,
        ``shuffle_*_test``, etc.) from ``missing`` / ``removed`` lists.
        These symbols are intentionally dropped by a device-only port and
        should not be flagged.

    Returns a machine-readable report. ``missing_*`` lists are the ones that
    matter: a kernel present in the source and absent from the port is a dropped
    kernel, regardless of what else the port added.

    A symbol that appears in ``rename_candidates`` is ALSO listed in the
    corresponding ``removed``/``added`` set. The rename is a hypothesis layered
    on top of the raw diff, never a substitute for it — a reader who trusts only
    the raw sets is never misled.
    """
    original = extract_symbols(cuda_source)
    ported = extract_symbols(hip_source)

    def _bucket(src: List[str], dst: List[str]) -> Dict:
        s, d = set(src), set(dst)
        return {
            "original": sorted(s),
            "generated": sorted(d),
            "missing": sorted(s - d),
            "added": sorted(d - s),
            "preserved": sorted(s & d),
        }

    kernels = _bucket(original.kernels, ported.kernels)
    helpers = _bucket(original.helpers, ported.helpers)
    functions = _bucket(original.functions, ported.functions)

    # DEVICE_SUBSET: host-level symbols are intentionally dropped.
    # Filter them out so the diff does NOT flag main(), shuffle_*_test,
    # etc. as "missing".
    host_only: Set[str] = set()
    if port_mode == "DEVICE_SUBSET":
        host_only = {n for n in original.functions if _is_host_only(n)}
        if host_only:
            functions["missing"] = sorted(set(functions["missing"]) - host_only)
            functions["original"] = sorted(set(functions["original"]) - host_only)

    all_removed = original.all_names() - ported.all_names()
    all_added = ported.all_names() - original.all_names()

    if host_only:
        all_removed = all_removed - host_only

    macro_diff = {
        "removed": sorted(set(original.macros) - set(ported.macros)),
        "added": sorted(set(ported.macros) - set(original.macros)),
        "changed": [
            {"name": k, "original": original.macros[k], "generated": ported.macros[k]}
            for k in sorted(set(original.macros) & set(ported.macros))
            if original.macros[k] != ported.macros[k]
        ],
    }

    return {
        "original_symbols": original.to_dict(),
        "generated_symbols": ported.to_dict(),
        "kernels": kernels,
        "helpers": helpers,
        "functions": functions,
        "macros": macro_diff,
        "includes": {
            "removed": sorted(set(original.includes) - set(ported.includes)),
            "added": sorted(set(ported.includes) - set(original.includes)),
        },
        "removed_symbols": sorted(all_removed),
        "added_symbols": sorted(all_added),
        "rename_candidates": _rename_candidates(all_removed, all_added),
        "summary": {
            "kernels_dropped": len(kernels["missing"]),
            "helpers_dropped": len(helpers["missing"]),
            "functions_dropped": len(functions["missing"]),
            "symbols_added": len(all_added),
            "macros_changed": len(macro_diff["changed"]),
        },
    }
