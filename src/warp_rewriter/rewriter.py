"""Deterministic CUDA→ROCm warp-size rewriter (P1.1).

The classifier already *detects* the warp(32)→wavefront(64) hazards; this pass
*fixes* the ones that have a single, provably-correct mechanical rewrite —
BEFORE the LLM loop ever runs. The model then only has to handle the genuinely
ambiguous residue, instead of re-deriving warp-size arithmetic every iteration
(and getting it wrong, at ~150s/lap).

Design rule: **every transform is anchored to an unambiguous warp token** — a
`__shfl*`/`__ballot*`/`__*_sync` intrinsic, `__syncwarp`, `__activemask`,
`threadIdx.x >> 5`, or a block-dimension divided by 32. There is deliberately
no blind `32`→`64` or bare-`0xffffffff` substitution: a lone `32` is usually an
array bound, not a warp size, and guessing is exactly the failure mode this
project exists to catch.

What is intentionally NOT rewritten (left to the LLM, because the correct fix
depends on intent, not syntax):
  - `__shfl_*_sync(..., width=32)` — the sub-group width may be deliberate.
  - shuffle-reduction step counts (offsets 16,8,4,2,1 need a 32 step added for
    a full 64-lane reduction) — that changes the algorithm, not a token.
  - host-side warp counts held in arbitrarily-named locals.
"""
import re
from typing import List, Tuple

# The warp-level intrinsics whose leading `0xffffffff` argument is a 32-lane
# participation mask that must widen to 64 lanes on a wavefront.
_SYNC_INTRINSICS = (
    r"__shfl_sync|__shfl_up_sync|__shfl_down_sync|__shfl_xor_sync|"
    r"__ballot_sync|__any_sync|__all_sync|__match_any_sync|__match_all_sync"
)

_MASK32 = "0xffffffff"
_MASK64 = "0xffffffffffffffff"


class WarpRewriter:
    """Apply provably-correct warp32→wavefront64 rewrites to kernel source."""

    def rewrite(self, source: str) -> Tuple[str, List[str]]:
        """Return (rewritten_source, changes).

        ``changes`` is a human-readable list of exactly what was rewritten and
        how many times — empty if the source had nothing mechanical to fix.
        """
        changes: List[str] = []
        text = source

        text = self._widen_sync_masks(text, changes)
        text = self._sub(text, changes,
                         rf"\b__syncwarp\s*\([^)]*\)", "__syncthreads()",
                         "__syncwarp() → __syncthreads() (HIP has no warp-scoped barrier)")
        text = self._sub(text, changes,
                         r"\b__activemask\s*\(\s*\)", "__ballot(1)",
                         "__activemask() → __ballot(1) (64-bit active-lane mask)")
        text = self._sub(text, changes,
                         r"(threadIdx\.[xyz]\s*>>\s*)5\b", r"\g<1>6",
                         "threadIdx >> 5 → >> 6 (warp index for 64-lane wavefront)")
        text = self._sub(text, changes,
                         r"(threadIdx\.[xyz]\s*&\s*)(?:0x1f|31)\b", r"\g<1>0x3f",
                         "threadIdx & 0x1f → & 0x3f (lane id for 64-lane wavefront)")
        text = self._sub(text, changes,
                         r"(\b(?:blockDim\.[xyz]|blockSize|BLOCK_SIZE|blockSz)\s*[/%]\s*)32\b", r"\g<1>64",
                         "blockDim/32 → /64 (warps per block on a 64-lane wavefront)")
        return text, changes

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sub(text: str, changes: List[str], pattern: str, repl: str, note: str) -> str:
        new_text, n = re.subn(pattern, repl, text)
        if n:
            changes.append(f"{note} [{n}×]")
        return new_text

    def _widen_sync_masks(self, text: str, changes: List[str]) -> str:
        """Widen the leading 32-bit mask of each warp *_sync intrinsic to 64-bit.

        Anchored to the intrinsic name + its opening paren, so an unrelated
        `0xffffffff` elsewhere in the source is never touched.
        """
        pattern = re.compile(rf"((?:{_SYNC_INTRINSICS})\s*\(\s*){re.escape(_MASK32)}\b")
        new_text, n = pattern.subn(rf"\g<1>{_MASK64}", text)
        if n:
            changes.append(f"warp mask {_MASK32} → {_MASK64} in *_sync intrinsics [{n}×]")
        return new_text


def rewrite_warp_size(source: str) -> Tuple[str, List[str]]:
    """Module-level convenience wrapper around :meth:`WarpRewriter.rewrite`."""
    return WarpRewriter().rewrite(source)
