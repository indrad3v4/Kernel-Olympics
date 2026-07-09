"""
Prompt Evolution — RL-style prompt versioning for the CUDA→HIP porting loop.

TRIZ Principles Applied:
  #15 Dynamics      — checklist items are parameterized, not static; each
                      iteration can add/remove items to match the problem.
  #23 Feedback      — reward signal = compile error delta
                      (errors_before - errors_after).  Items that correlate
                      with error reduction get higher scores.
  #22 Throwing Away — low-scoring checklist items are discarded after
                      sustained negative reward.
  #9  Preliminary Anti-Action — seed prompt starts with known-good items;
                      expansion pool adds targeted items only when needed
                      (prevents prompt bloat).
  #20 Continuation  — prompt keeps evolving until win condition
                      (compile_passed == True) or max iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── PromptVersion dataclass ────────────────────────────────────────────────────

@dataclass
class PromptVersion:
    """A single version of the porting prompt checklist.

    Tracks how well this version performed across compile iterations so the
    PromptOptimizer can score individual checklist items (TRIZ #23 Feedback).
    """

    version_id: str
    checklist: List[str]
    iterations_used: int = 0
    compile_errors_before: int = 0
    compile_errors_after: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def reward(self) -> float:
        """Average per-iteration reward = error reduction / iterations used.

        Returns 0.0 when no iterations have been recorded (avoids misleading
        nonzero reward from un-earned error deltas).
        """
        if self.iterations_used == 0:
            return 0.0
        return (self.compile_errors_before - self.compile_errors_after) / max(self.iterations_used, 1)

    @property
    def win_rate(self) -> float:
        """Fraction of iterations where errors decreased."""
        return self.wins / max(self.wins + self.losses, 1)


# ── PromptOptimizer ────────────────────────────────────────────────────────────

class PromptOptimizer:
    """Reinforcement-learning-style optimizer for the Kimi refine checklist.

    The optimizer scores each checklist item by how often it correlates with
    a reduction in compile errors.  Low-scoring items are pruned (TRIZ #22),
    and targeted expansion items are added only when the latest compile errors
    reference specific CUDA APIs (TRIZ #15 Dynamics / #9 Preliminary
    Anti-Action).
    """

    # TRIZ #9: Preliminary Anti-Action — seed with known-good items so the
    # first iteration already has a strong checklist.
    SEED_CHECKLIST: List[str] = [
        "__shfl_xor_sync mask 0x1f → 0x3f for wavefront64",
        "__shfl_down_sync masks → 0xffffffffffffffffULL (64-bit)",
        "warpSize 32 → WAVEFRONT_SIZE 64 or dynamic",
        "shared memory sized for warp 32 → WAVEFRONT_SIZE (64)",
        "__syncwarp() → __syncthreads()",
        "#define WAVEFRONT_SIZE 64 at top",
        "Replace #include <cuda_runtime.h> → #include <hip/hip_runtime.h>",
        "Remove #include <helper_cuda.h>, <helper_functions.h>",
        'Remove ALL #include "*.cuh" local headers',
    ]

    # TRIZ #15: Dynamics — expansion pool grows the checklist to match the
    # specific compile errors observed.  Items are only pulled in when the
    # corresponding CUDA keyword appears in the error stream.
    EXPANSION_POOL: List[str] = [
        "Replace cudaMalloc→hipMalloc, cudaFree→hipFree, cudaMemcpy→hipMemcpy",
        "Replace cudaEvent_t→hipEvent_t, cudaEventCreate→hipEventCreate",
        "Replace checkCudaErrors(x) → (void)(x) — no HIP equivalent macro",
        "Replace cudaMallocHost→hipHostMalloc, cudaFreeHost→hipHostFree",
        "Replace cudaDeviceSynchronize→hipDeviceSynchronize",
        "Replace cudaGetDevice→hipDeviceGet, cudaGetDeviceProperties→hipGetDeviceProperties",
        "Replace cudaMemcpyHostToDevice→hipMemcpyHostToDevice (all enum values)",
        "Replace cudaError_t→hipError_t, cudaSuccess→hipSuccess",
        "Replace cudaStream_t→hipStream_t, cudaStreamCreate→hipStreamCreate",
        "Replace cudaMemset→hipMemset, cudaDeviceProp→hipDeviceProp_t",
    ]

    # Keyword → expansion-pool index mapping.  When a compile error contains
    # the keyword, the corresponding expansion item is pulled into the
    # checklist (TRIZ #15).
    _EXPANSION_KEYWORDS: Dict[str, int] = {
        "cudaMalloc": 0,
        "cudaEvent": 1,
        "checkCudaErrors": 2,
        "cudaMallocHost": 3,
        "cudaDeviceSynchronize": 4,
        "cudaGetDevice": 5,
        "cudaMemcpyHostToDevice": 6,
        "cudaError": 7,
        "cudaStream": 8,
        "cudaMemset": 9,
    }

    def __init__(self) -> None:
        self._version_counter: int = 1
        self.current_version: PromptVersion = PromptVersion(
            "v1_seed", list(self.SEED_CHECKLIST)
        )
        self._item_scores: Dict[str, int] = {}
        self._error_history: List[int] = []
        self._versions: Dict[str, PromptVersion] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def record_iteration(
        self,
        errors_before: int,
        errors_after: int,
        checklist_used: List[str],
    ) -> Dict:
        """Record one compile iteration and update item scores (TRIZ #23).

        Returns a dict with the reward signal, error delta, and top-scoring
        items so the caller can log/diagnose the evolution step.
        """
        delta = errors_before - errors_after
        if delta > 0:
            reward = 1
        elif delta < 0:
            reward = -1
        else:
            reward = 0

        # Score each item in the checklist that was used this iteration.
        for item in checklist_used:
            self._item_scores[item] = self._item_scores.get(item, 0) + reward

        # Update current version stats.
        self.current_version.iterations_used += 1
        self.current_version.compile_errors_before += errors_before
        self.current_version.compile_errors_after += errors_after
        if reward > 0:
            self.current_version.wins += 1
        elif reward < 0:
            self.current_version.losses += 1

        self._error_history.append(errors_after)

        # Top 5 items by score.
        sorted_items = sorted(
            self._item_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        top_items = sorted_items[:5]

        return {
            "reward": reward,
            "error_delta": delta,
            "errors_before": errors_before,
            "errors_after": errors_after,
            "top_items": top_items,
        }

    def evolve_prompt(self, last_compile_errors: List[str]) -> PromptVersion:
        """Evolve the checklist based on the latest compile errors.

        TRIZ #15 (Dynamics)  — reorder by item score, add expansion items that
                               match observed compile errors.
        TRIZ #22 (Throwing)  — drop the bottom-2 items if they have negative
                               scores.
        TRIZ #20 (Continuation) — returns the new version for the next loop
                               iteration; keeps evolving until compile passes.
        """
        self._version_counter += 1
        new_version_id = f"v{self._version_counter}"

        # Start from current checklist, sorted by item score descending.
        current_items = list(self.current_version.checklist)
        current_items.sort(
            key=lambda item: self._item_scores.get(item, 0), reverse=True
        )

        # TRIZ #22: Throwing Away — drop bottom-2 items with negative scores.
        if len(current_items) >= 2:
            bottom_two = current_items[-2:]
            kept = [
                item
                for item in current_items
                if not (item in bottom_two and self._item_scores.get(item, 0) < 0)
            ]
            current_items = kept

        # TRIZ #15: Dynamics — scan compile errors for CUDA keywords and add
        # matching expansion items.
        error_blob = " ".join(last_compile_errors)
        for keyword, pool_idx in self._EXPANSION_KEYWORDS.items():
            if keyword in error_blob:
                expansion_item = self.EXPANSION_POOL[pool_idx]
                if expansion_item not in current_items:
                    current_items.append(expansion_item)

        # Store old version, set new current.
        self._versions[self.current_version.version_id] = self.current_version
        new_version = PromptVersion(new_version_id, current_items)
        self.current_version = new_version
        return new_version

    def get_checklist(self) -> List[str]:
        """Return the current version's checklist."""
        return self.current_version.checklist

    def get_stats(self) -> Dict:
        """Return summary statistics for the current prompt version."""
        sorted_items = sorted(
            self._item_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        top_5 = sorted_items[:5]
        return {
            "win_rate": self.current_version.win_rate,
            "reward": self.current_version.reward,
            "iterations_used": self.current_version.iterations_used,
            "version_id": self.current_version.version_id,
            "top_item_scores": top_5,
        }

    def reset(self) -> None:
        """Reset the optimizer to its initial seed state."""
        self._version_counter = 1
        self.current_version = PromptVersion("v1_seed", list(self.SEED_CHECKLIST))
        self._item_scores = {}
        self._error_history = []
        self._versions = {}


# ── Module-level singleton ─────────────────────────────────────────────────────
# Global optimizer instance — persists across loop iterations within a single
# route() call.  Callers may also instantiate PromptOptimizer locally for
# per-kernel isolation (as route() does).
prompt_opt = PromptOptimizer()
