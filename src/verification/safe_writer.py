"""File-writer hardening for generated HIP source.

Every write of an LLM-produced ``.cpp`` / ``.hip.cpp`` / ``.cu`` file MUST go
through :func:`safe_write_source`.  That function runs a lexical → structural
validation chain and refuses to write when either stage fails.  A refused
write returns a ``WriteResult`` describing why, so the caller can retry the
model instead of leaving the previous good file behind, or aborting.

This is the last barrier between LLM text and disk.  A failure here means
the pipeline has already exhausted its extraction + validation options; the
correct response is always to re-run generation, never to relax the check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from verification.lexical import validate_lexical, LexicalResult
from verification.structural import (
    validate_structure,
    ValidationResult as StructuralResult,
)


@dataclass
class WriteResult:
    """Outcome of a hardened file write."""
    written: bool
    path: Optional[Path] = None
    reason: str = ""
    lexical: Optional[LexicalResult] = None
    structural: Optional[StructuralResult] = None
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe representation for run logs."""
        return {
            "written": self.written,
            "path": str(self.path) if self.path else None,
            "reason": self.reason,
            "lexical_ok": None if self.lexical is None else self.lexical.ok,
            "lexical_errors": [] if self.lexical is None else list(self.lexical.errors),
            "structural_ok": None if self.structural is None else self.structural.ok,
            "structural_errors": [] if self.structural is None else list(self.structural.errors),
            "diagnostics": list(self.diagnostics),
        }


def safe_write_source(
    path: Path,
    code: str,
    source_cuda: str = "",
    *,
    require_structural: bool = True,
) -> WriteResult:
    """Validate *code* and write it to *path* only if it passes every gate.

    ``source_cuda`` is the ORIGINAL CUDA source, used by the structural gate
    for symbol-preservation checks.  Leave empty when writing a pre-processed
    driver, template harness, or other file that has no CUDA counterpart —
    in that case only lexical validation runs.

    ``require_structural`` may be turned off ONLY for cases where the file
    is a fragment (e.g. an intermediate refine that will be wrapped later
    by a harness generator).  The lexical gate always fires.

    Returns a :class:`WriteResult`.  On failure the file on disk is left
    untouched — no partial writes, no zero-byte artifacts.
    """
    path = Path(path)
    diagnostics: List[str] = []

    # Guard 1: empty / whitespace-only input.  Even the writer refuses this.
    if not code or not code.strip():
        return WriteResult(
            written=False,
            path=path,
            reason="empty content",
            diagnostics=["safe_write_source refused to write an empty file"],
        )

    # Guard 2: lexical.  Reasoning, markdown, and role tags die here.
    lex = validate_lexical(code)
    if not lex.ok:
        logger.warning("safe_write_source refused %s: lexical %s",
                       path.name, lex.reason())
        return WriteResult(
            written=False,
            path=path,
            reason=f"lexical: {lex.reason()}",
            lexical=lex,
            diagnostics=[f"prose_samples={lex.prose_line_samples[:2]}"],
        )

    # Guard 3: structural.  Unbalanced braces, truncation markers, dropped
    # symbols get rejected before hipcc sees them.
    struct: Optional[StructuralResult] = None
    if require_structural:
        try:
            struct = validate_structure(source_cuda or code, code)
        except Exception as exc:  # never let a validator bug leak
            logger.exception("structural validator crashed for %s", path.name)
            diagnostics.append(f"structural validator crashed: {exc!r}")
            struct = StructuralResult(ok=True, warnings=["structural check errored"])
        if not struct.ok:
            logger.warning("safe_write_source refused %s: structural %s",
                           path.name, struct.reason())
            return WriteResult(
                written=False,
                path=path,
                reason=f"structural: {struct.reason()}",
                lexical=lex,
                structural=struct,
                diagnostics=diagnostics,
            )

    # All gates passed — write atomically via a sibling temp file so a
    # concurrent reader cannot see a truncated .cpp.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(code, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.exception("safe_write_source could not write %s", path)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return WriteResult(
            written=False,
            path=path,
            reason=f"OSError: {exc!r}",
            lexical=lex,
            structural=struct,
            diagnostics=diagnostics,
        )

    return WriteResult(
        written=True,
        path=path,
        reason="ok",
        lexical=lex,
        structural=struct,
        diagnostics=diagnostics,
    )


__all__ = ["WriteResult", "safe_write_source"]
