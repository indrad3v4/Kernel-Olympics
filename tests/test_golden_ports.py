"""Golden-reference port tests (P1.2).

A hand-verified wavefront-64 HIP port is the regression oracle: it pins what
"correct" looks like, and it lets us assert that the deterministic rewriter is
*consistent* with it — fixing the mechanical hazards while correctly leaving
the algorithmic ones (reduction step count) to the porter.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_rewriter import rewrite_warp_size

ROOT = os.path.join(os.path.dirname(__file__), '..')
GOLDEN = os.path.join(ROOT, 'sample_kernels', 'golden', 'warp_reduce.hip.cpp')
CUDA = os.path.join(ROOT, 'sample_kernels', 'cuda', 'warp_reduce.cu')


def _read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def _code(text):
    """Strip // and /* */ comments so assertions test code, not prose (the
    golden's docstring literally mentions shared[32] and 0xffffffff)."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    return text


# ── the golden itself is a correct wavefront-64 port ────────────────

def test_golden_exists_and_names_the_kernel():
    g = _read(GOLDEN)
    assert "__global__ void warp_reduce_kernel(" in g


def test_golden_shared_mem_sized_for_wavefront():
    g = _code(_read(GOLDEN))
    assert "shared[64]" in g and "shared[32]" not in g


def test_golden_has_full_64_lane_reduction():
    g = _code(_read(GOLDEN))
    # 6 steps for 64 lanes, including the offset-32 step the NVIDIA source omits.
    assert g.count("__shfl_down(") == 6
    assert "__shfl_down(val, 32)" in g


def test_golden_has_no_32bit_warp_mask_or_syncwarp():
    g = _code(_read(GOLDEN))
    assert "0xffffffff" not in g      # no 32-lane participation mask survives
    assert "__syncwarp" not in g


# ── the rewriter is consistent with the golden ──────────────────────

def test_rewriter_widens_masks_like_the_golden():
    out, changes = rewrite_warp_size(_read(CUDA))
    # Every 32-bit shuffle mask is widened — none of the bare-mask shfls remain.
    assert "__shfl_down_sync(0xffffffff," not in out
    assert any("mask" in c for c in changes)


def test_rewriter_leaves_the_algorithmic_step_count_alone():
    """The rewriter must NOT invent the offset-32 reduction step — that's an
    algorithm change, not a token rewrite. The source has 5 shuffle calls; the
    rewriter output still has 5 (the golden's 6th is the porter's job)."""
    src_code = _code(_read(CUDA))
    out_code = _code(rewrite_warp_size(_read(CUDA))[0])
    assert src_code.count("__shfl_down_sync(") == 5
    assert out_code.count("__shfl_down_sync(") == 5   # count unchanged by rewriter
    assert "val, 32)" not in out_code                 # no offset-32 step invented
