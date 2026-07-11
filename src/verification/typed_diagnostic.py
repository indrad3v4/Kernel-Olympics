"""Typed compiler diagnostics for stagnation/cycle detection.

The router's plateau detector compared normalised error *strings* — two
different missing symbols at different sites often produced the same
normalised string after path/line stripping, so the loop flagged a false
cycle and aborted. Two identical defects at different source offsets
sometimes produced different strings (message body varied by column), so
a real cycle wasn't caught.

This module builds a *set* of ``(kind, symbol, owner)`` triples per
iteration. Set equality is the collision test: two iterations converged
to the same defect surface iff their triple sets are equal.

The parsing engine is shared with :mod:`verification.semantic_repair`,
which already normalises hipcc/clang diagnostic strings into
:class:`Diagnostic`. This module re-exports that type under a clearer
name and adds the set-summary helper.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Iterable, List, Tuple

from verification.semantic_repair import Diagnostic as TypedDiag
from verification.semantic_repair import parse_diagnostics


DiagKey = Tuple[str, str, str]  # (kind, symbol, owner)

# Strip "<path>:<line>:<col>:" prefixes and the "error:"/"warning:" tag from
# an unknown-kind diagnostic so two runs of the same untyped defect at
# different offsets still collide in the set. Trailing whitespace collapsed.
_LOC_PREFIX = re.compile(r'([^\s:]+:)?\d+:\d+:\s*(?:fatal\s+)?(?:error|warning):\s*',
                         re.IGNORECASE)


def _normalize_unknown(raw: str) -> str:
    text = _LOC_PREFIX.sub("", raw or "").strip()
    return re.sub(r"\s+", " ", text)[:120]


def diagnostic_set(errors: Iterable[str]) -> FrozenSet[DiagKey]:
    """Parse *errors* and return the frozen set of ``(kind, symbol, owner)``.

    Unparseable / unknown-kind diagnostics contribute
    ``("unknown", <location-stripped message>, "")`` so a genuinely novel
    error is not silently dropped, while identical defects at different
    line numbers still collide.
    """
    diags = parse_diagnostics(list(errors))
    triples: List[DiagKey] = []
    for d in diags:
        if d.kind == "unknown":
            triples.append(("unknown", _normalize_unknown(d.raw), ""))
        else:
            triples.append((d.kind, d.symbol, d.owner))
    return frozenset(triples)


__all__ = ["TypedDiag", "parse_diagnostics", "diagnostic_set", "DiagKey"]
