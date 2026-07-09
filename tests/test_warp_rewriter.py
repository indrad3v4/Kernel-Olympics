"""Tests for the deterministic warp-size rewriter (P1.1).

Two things matter equally: it MUST fix the mechanical warp32→wavefront64
hazards, and it MUST NOT touch a `32` or `0xffffffff` that isn't a warp size.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_rewriter import WarpRewriter, rewrite_warp_size


def r(src):
    return WarpRewriter().rewrite(src)


# ── each transform ──────────────────────────────────────────────────

def test_syncwarp_becomes_syncthreads():
    out, changes = r("__syncwarp(0xffffffff);")
    assert "__syncthreads()" in out and "__syncwarp" not in out
    assert any("syncwarp" in c for c in changes)


def test_shfl_mask_widens_to_64bit():
    out, _ = r("int n = __shfl_down_sync(0xffffffff, v, 16);")
    assert "0xffffffffffffffff" in out
    assert "__shfl_down_sync(0xffffffffffffffff" in out


def test_ballot_and_vote_masks_widen():
    out, _ = r("__ballot_sync(0xffffffff, p); __all_sync(0xffffffff, q);")
    assert out.count("0xffffffffffffffff") == 2


def test_activemask_becomes_ballot():
    out, _ = r("unsigned m = __activemask();")
    assert "__ballot(1)" in out and "__activemask" not in out


def test_warp_index_shift():
    out, _ = r("int warp = threadIdx.x >> 5;")
    assert "threadIdx.x >> 6" in out


def test_lane_id_mask():
    out, _ = r("int lane = threadIdx.x & 0x1f; int l2 = threadIdx.y & 31;")
    assert "threadIdx.x & 0x3f" in out
    assert "threadIdx.y & 0x3f" in out


def test_warps_per_block():
    out, _ = r("int nWarps = blockSize / 32; int w = blockDim.x / 32;")
    assert "blockSize / 64" in out
    assert "blockDim.x / 64" in out


# ── safety: never touch a non-warp 32 / mask ────────────────────────

def test_unrelated_array_size_32_untouched():
    src = "__shared__ float tile[32]; int rows = height / 32; x = y * 32;"
    out, changes = r(src)
    assert out == src            # nothing anchored to a warp token
    assert changes == []


def test_unrelated_0xffffffff_untouched():
    # A plain 32-bit bitmask with no warp intrinsic around it must survive.
    src = "uint32_t color = pixel & 0xffffffff;"
    out, changes = r(src)
    assert out == src
    assert changes == []


def test_shfl_width_argument_is_left_for_the_llm():
    # The 4th (width) arg is semantically ambiguous — must NOT be auto-changed.
    src = "__shfl_down_sync(0xffffffff, v, d, 32);"
    out, _ = r(src)
    assert "d, 32)" in out       # width 32 preserved
    assert "0xffffffffffffffff" in out  # but the mask still widened


# ── properties ──────────────────────────────────────────────────────

def test_idempotent():
    src = "__syncwarp(); int w = threadIdx.x >> 5; __shfl_xor_sync(0xffffffff, v, 1);"
    once, _ = r(src)
    twice, changes2 = r(once)
    assert once == twice
    assert changes2 == []        # second pass finds nothing left to fix


def test_clean_source_is_noop():
    src = "int idx = blockIdx.x * blockDim.x + threadIdx.x;\noutput[idx] = input[idx];"
    out, changes = r(src)
    assert out == src and changes == []


def test_module_wrapper_matches_class():
    src = "__syncwarp();"
    assert rewrite_warp_size(src) == WarpRewriter().rewrite(src)


# ── golden: a realistic warp-reduce fragment ────────────────────────

def test_golden_warp_reduce_fragment():
    cuda = (
        "int lane = threadIdx.x & 0x1f;\n"
        "int wid  = threadIdx.x >> 5;\n"
        "val += __shfl_down_sync(0xffffffff, val, 16);\n"
        "__syncwarp();\n"
        "int nwarps = blockDim.x / 32;\n"
    )
    out, changes = r(cuda)
    assert "threadIdx.x & 0x3f" in out
    assert "threadIdx.x >> 6" in out
    assert "__shfl_down_sync(0xffffffffffffffff, val, 16)" in out
    assert "__syncthreads()" in out
    assert "blockDim.x / 64" in out
    assert len(changes) == 5      # one note per distinct transform fired
