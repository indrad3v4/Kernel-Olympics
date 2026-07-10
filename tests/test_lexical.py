"""Tests for verification.lexical — the reasoning/prose reject gate."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from verification.lexical import validate_lexical  # noqa: E402


# ── Positive: strings that MUST pass ────────────────────────────────────────

def test_pass_bare_include():
    code = "#include <hip/hip_runtime.h>\n"
    r = validate_lexical(code)
    assert r.ok, r.reason()


def test_pass_kernel_body():
    code = """\
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vec_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
"""
    r = validate_lexical(code)
    assert r.ok, r.reason()


def test_pass_copyright_header_is_ignored():
    """A copyright comment must never trigger the prose detector."""
    code = """\
/*
 * Copyright (c) 2024 NVIDIA Corporation. All rights reserved.
 *
 * NOTICE: This is a copyright header.  It contains English prose but must
 * not fail the lexical validator because it is a comment.  We would like
 * this to be treated as source-code annotation, not top-level reasoning.
 */
#include <hip/hip_runtime.h>

__global__ void k() {}
"""
    r = validate_lexical(code)
    assert r.ok, r.reason()


def test_pass_string_literal_with_english():
    code = """\
#include <cstdio>
__global__ void k() {
    printf("Hello from device!  This is a full English sentence.\\n");
}
"""
    r = validate_lexical(code)
    assert r.ok, r.reason()


# ── Negative: strings that MUST fail ────────────────────────────────────────

def test_reject_observed_failure_verbatim():
    """The exact text from the bug report must fail."""
    code = """\
#include <iostream>
#include <hip/hip_runtime.h>

local headers". So we need inline the cuh.

Let's search memory more concretely.

I think...

Wait maybe...

Let's design our own kernels...
"""
    r = validate_lexical(code)
    assert not r.ok
    # We want a reason a human can act on.  Any of these is meaningful for
    # this input — reasoning phrases, section headings, trailing ellipsis.
    reason = r.reason().lower()
    assert any(w in reason for w in ("reasoning", "prose", "heading",
                                    "ellipsis", "mid-thought"))


def test_reject_lets_reasoning():
    code = "#include <hip/hip_runtime.h>\nLet's think about this.\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_i_think():
    code = "#include <hip/hip_runtime.h>\nI think this needs a different approach.\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_wait_maybe():
    code = "#include <hip/hip_runtime.h>\nWait, maybe we should use shared memory.\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_markdown_heading():
    code = "# Ported Kernel\n#include <hip/hip_runtime.h>\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_markdown_fence():
    code = "```cpp\n#include <hip/hip_runtime.h>\n```\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_bullet_list():
    code = """\
#include <hip/hip_runtime.h>

- First, replace warpSize with 64
- Then, adjust masks
"""
    r = validate_lexical(code)
    assert not r.ok


def test_reject_numbered_list():
    code = """\
#include <hip/hip_runtime.h>

1. Replace warpSize with 64
2. Adjust masks
"""
    r = validate_lexical(code)
    assert not r.ok


def test_reject_section_heading():
    code = """\
#include <hip/hip_runtime.h>

Explanation: this kernel does vector addition.

__global__ void k() {}
"""
    r = validate_lexical(code)
    assert not r.ok


def test_reject_role_tag():
    code = "assistant: sure, here is the kernel.\n#include <hip/hip_runtime.h>\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_pure_prose_no_code():
    code = "This kernel adds two vectors element-wise.\nIt does not use shared memory.\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_empty():
    assert not validate_lexical("").ok
    assert not validate_lexical(None).ok
    assert not validate_lexical("   \n \n").ok


def test_reject_trailing_ellipsis():
    code = "#include <hip/hip_runtime.h>\n__global__ void k() { ...\n"
    r = validate_lexical(code)
    assert not r.ok


def test_reject_markdown_bold_header_line():
    code = "**Important**\n#include <hip/hip_runtime.h>\n"
    r = validate_lexical(code)
    assert not r.ok


def test_accept_cpp_pointer_to_pointer():
    """``**t`` in a C++ signature must not be mistaken for markdown bold."""
    code = """\
#include <hip/hip_runtime.h>
struct Timer {};
static inline void sdkCreateTimer(Timer **t) { *t = new Timer(); }
"""
    r = validate_lexical(code)
    assert r.ok, r.reason()


def test_reject_high_prose_ratio():
    code = """\
#include <hip/hip_runtime.h>
Above is the include.
Below is more text.
Nothing else really matters.
Just prose everywhere.
The compiler will complain.
"""
    r = validate_lexical(code)
    assert not r.ok


def test_stats_populated_on_failure():
    code = "#include <hip/hip_runtime.h>\nLet's do it.\n"
    r = validate_lexical(code)
    assert not r.ok
    assert r.stats.get("code_lines", 0) >= 1
    assert r.stats.get("prose_lines", 0) >= 1
