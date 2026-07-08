"""
Pattern signatures — the canonical fingerprint of a kernel's migration problems.

A *pattern signature* is the order-independent, de-duplicated set of
``(pattern_name, severity)`` pairs that the risk classifier detected in a CUDA
kernel. It is what the Pattern Memory cache is keyed on.

Two kernels share a signature only when they exhibit the **same set of
CUDA→ROCm migration problems** — regardless of source text, whitespace,
comments, identifier names or statement ordering. This is the property that
makes the cache correct: a hit always returns an *applicable* fix, because the
stored fix solved the same collection of problems.

This module is intentionally pure: it has no dependency on the classifier, the
database, or any I/O. It only knows how to build, normalise, serialise and hash
signatures, which keeps it trivially testable and reusable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# ── Canonical severity vocabulary ────────────────────────────────────────────
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"
SEVERITY_UNKNOWN = "UNKNOWN"

# Map the many ways a severity may be spelled onto the canonical vocabulary.
# Anything unrecognised (or missing) collapses to UNKNOWN so signatures stay
# deterministic even with dirty classifier output.
_SEVERITY_ALIASES = {
    "high": SEVERITY_HIGH,
    "critical": SEVERITY_HIGH,
    "medium": SEVERITY_MEDIUM,
    "med": SEVERITY_MEDIUM,
    "moderate": SEVERITY_MEDIUM,
    "low": SEVERITY_LOW,
    "info": SEVERITY_LOW,
    "informational": SEVERITY_LOW,
}

_UNKNOWN_PATTERN = "unknown_pattern"

# Serialisation separators. Pattern names are lowercase identifiers and never
# contain these characters, so the serialised form round-trips unambiguously.
_FIELD_SEP = ":"   # between a pattern name and its severity
_PAIR_SEP = "|"    # between (name, severity) pairs


def normalize_severity(severity) -> str:
    """Collapse any severity spelling onto the canonical vocabulary.

    Missing/unknown severities become ``UNKNOWN`` (never raises).
    """
    if severity is None:
        return SEVERITY_UNKNOWN
    return _SEVERITY_ALIASES.get(str(severity).strip().lower(), SEVERITY_UNKNOWN)


def normalize_pattern_name(name) -> str:
    """Canonicalise a pattern name (lowercased, stripped).

    Empty/None names become ``unknown_pattern`` so an unrecognised finding still
    contributes deterministically to the signature instead of being dropped.
    """
    if name is None:
        return _UNKNOWN_PATTERN
    cleaned = str(name).strip().lower()
    return cleaned or _UNKNOWN_PATTERN


@dataclass(frozen=True)
class PatternSignature:
    """Immutable, canonical fingerprint of a kernel's migration problems.

    ``pairs`` is always sorted and de-duplicated, so equality, serialisation and
    hashing are deterministic and independent of classifier output ordering.
    """

    pairs: Tuple[Tuple[str, str], ...]

    # ── Construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_pairs(cls, pairs: Iterable[Sequence]) -> "PatternSignature":
        """Build a signature from ``(name, severity)`` pairs (any order/dupes)."""
        canonical = {
            (normalize_pattern_name(p[0]), normalize_severity(p[1] if len(p) > 1 else None))
            for p in pairs
            if p is not None and len(p) >= 1
        }
        return cls(tuple(sorted(canonical)))

    @classmethod
    def from_findings(cls, findings) -> "PatternSignature":
        """Build a signature from classifier findings.

        Accepts the classifier's list of ``{"pattern", "severity", ...}`` dicts,
        and also tolerates ``(name, severity)`` tuples for flexibility.
        """
        extracted = []
        for finding in findings or []:
            if isinstance(finding, dict):
                extracted.append((finding.get("pattern"), finding.get("severity")))
            elif isinstance(finding, (tuple, list)) and finding:
                name = finding[0]
                severity = finding[1] if len(finding) > 1 else None
                extracted.append((name, severity))
            # Anything else is silently ignored — fail gracefully, never raise.
        return cls.from_pairs(extracted)

    @classmethod
    def deserialize(cls, key: str) -> "PatternSignature":
        """Reconstruct a signature from its serialised string form."""
        if not key:
            return cls(())
        pairs = []
        for token in key.split(_PAIR_SEP):
            if not token:
                continue
            name, _, severity = token.partition(_FIELD_SEP)
            pairs.append((name, severity))
        return cls.from_pairs(pairs)

    # ── Views ────────────────────────────────────────────────────────────────
    @property
    def is_empty(self) -> bool:
        """True when no migration problems were detected — nothing to cache."""
        return len(self.pairs) == 0

    def serialize(self) -> str:
        """Deterministic string key, e.g. ``ballot:HIGH|warp_shuffle:HIGH``."""
        return _PAIR_SEP.join(f"{name}{_FIELD_SEP}{severity}" for name, severity in self.pairs)

    def digest(self) -> str:
        """Stable, cross-process content hash of the serialised signature.

        Uses SHA-256 (not Python's per-process-salted ``hash()``) so the same
        signature always yields the same id on any machine or run.
        """
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()[:16]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.serialize() or "<empty-signature>"
