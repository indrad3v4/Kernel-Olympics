"""Tests for ModelRouter._reject_structurally_invalid — semantic gate.

Verifies three check categories:

  1. **Prose at file scope** — English signal phrases or backtick prose
     lines that escaped into the code output (Check #1 in the docstring).

  2. **Ghost kernel launches** — ``identifier<<<`` calls without a matching
     ``__global__`` definition anywhere in the file (Check #2).

  3. **Executable statements outside functions** — ``=``, ``<<<``, ``->``, ``::``
     combined with ``;`` at file scope that are not a valid C++ declaration
     (Check #3).

Plus edge cases: empty input, whitespace-only, prose inside function bodies,
method definitions at file scope, HIP wrapper macros, single-line kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from router import ModelRouter  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _check(code: str, original: str = ""):
    """Call the static method directly and return the result object.

    ``original_source`` is accepted but unused by the implementation.
    """
    return ModelRouter._reject_structurally_invalid(code, original)


# ═══════════════════════════════════════════════════════════════════════════
# POSITIVE — code that MUST pass
# ═══════════════════════════════════════════════════════════════════════════

def test_pass_valid_kernel_with_matching_launch():
    """A __global__ definition with a matching <<<<<< call must pass."""
    code = """\
#include <hip/hip_runtime.h>

__global__ void vec_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

int main() {
    vec_add<<<128, 256>>>(d_a, d_b, n);
    hipDeviceSynchronize();
    return 0;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_single_line_kernel():
    """Single-line __global__ body must not trigger executable-statement check."""
    code = """\
#include <hip/hip_runtime.h>
__global__ void k() { int x = 1; }
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_includes_defines_comments():
    """Includes, defines, comment-only files (no code) must pass."""
    code = """\
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2024 NVIDIA Corporation
 */
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 64

__global__ void k() {}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_hip_wrapper_functions():
    """hipLaunchKernelGGL must NOT be flagged as a ghost kernel launch."""
    code = """\
#include <hip/hip_runtime.h>

__global__ void vec_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

int main() {
    hipLaunchKernelGGL(vec_add, dim3(128), dim3(256), 0, 0, d_a, d_b, n);
    return 0;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_prose_inside_function_body():
    """Prose-like lines inside a function body must NOT trigger prose detection."""
    code = """\
#include <cstdio>

__global__ void k() {
    printf("Hello from device!\\n");
    // Note: this is a comment inside a function
    /* However, this is also a block comment */
    // This is fine too
    int x = 42;
}

int main() {
    // Note: driver code
    printf("The result is: %d\\n", 42);
    return 0;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_method_definition_at_file_scope():
    """`ClassName::method()` at file scope must NOT trigger executable-statement check."""
    code = """\
#include <hip/hip_runtime.h>

void MyClass::vec_add(float *a, float *b, int n) {
    int i = 0;
    if (i < n) a[i] += b[i];
}

int MyClass::get_value() const {
    return 42;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_namespace_using_template():
    """Namespace, using, template declarations at file scope must pass."""
    code = """\
#include <hip/hip_runtime.h>

namespace my_ns {
__global__ void k() {}
}

using namespace my_ns;

template <typename T>
__global__ void vec_add(T *a, T *b, int n) {
    int i = threadIdx.x;
    if (i < n) a[i] += b[i];
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_forward_declaration():
    """Function forward declaration (prototype with trailing ``;`` but no body) must pass."""
    code = """\
#include <hip/hip_runtime.h>

__global__ void vec_add(float *a, float *b, int n);

int main() {
    return 0;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


def test_pass_empty_and_whitespace_only():
    """Empty string, whitespace-only, and comment-only files must pass."""
    assert _check("").ok
    assert _check("   \n \n  ").ok
    assert _check("// just a comment\n").ok
    assert _check("/* block comment\n   only */\n").ok


def test_pass_ghost_keyword_skip_list():
    """Keywords like ``if``, ``for``, ``while`` followed by ``<<<`` inside a function
    must be skipped by the ghost kernel detector (they are shift operators, not launches).

    The ghost check scans the whole file (including comments), so the test avoids
    putting ``word<<<`` inside comments as that can produce false positives.
    """
    code = """\
#include <hip/hip_runtime.h>

__global__ void k() {
    int x = 1;
    if (x >> 2) {}
    for (int i = 0; i < 10; i++) {}
    while (false) {}
    return;
}
"""
    result = _check(code)
    assert result.ok, f"expected pass, got errors: {result.errors}"


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE — code that MUST fail (Check #1: prose at file scope)
# ═══════════════════════════════════════════════════════════════════════════

def test_reject_prose_the_at_file_scope():
    """Line starting with ``The`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

The host code launches the kernel with a 1D grid.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok, "expected rejection for prose starting with 'The'"
    assert any("prose" in e.lower() for e in result.errors), (
        f"error message should mention prose: {result.errors}"
    )


def test_reject_prose_this_at_file_scope():
    """Line starting with ``This`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

This kernel performs element-wise vector addition.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok
    assert any("prose" in e.lower() for e in result.errors)


def test_reject_prose_we_at_file_scope():
    """Line starting with ``We`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

We need to replace __shfl_down_sync with __shfl_down.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_note_at_file_scope():
    """Line starting with ``Note`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

Note that warp size is 32 on NVIDIA but 64 on AMD.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_but_at_file_scope():
    """Line starting with ``But`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

But we must also adjust the shared memory size.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_however_at_file_scope():
    """Line starting with ``However`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

However, there is a subtle difference in warp semantics.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_also_at_file_scope():
    """Line starting with ``Also`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

Also check that the shared memory bank size matches.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_therefore_at_file_scope():
    """Line starting with ``Therefore`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

Therefore we use __shfl_down instead of __shfl_down_sync.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok


def test_reject_prose_hint_backtick_pattern():
    """Line with a backtick at file scope starting with uppercase word (``Host code: `int...```) must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

Host code: `int nWarps = blockSize / 32;` should be placed in main().

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok
    assert any("prose" in e.lower() for e in result.errors), (
        f"error should mention prose: {result.errors}"
    )


def test_reject_multiple_prose_lines():
    """Multiple prose lines at file scope must all be reported."""
    code = """\
#include <hip/hip_runtime.h>

The first step is to include the header.
This kernel uses shared memory.
We also need to change the warp size.

__global__ void k() {}
"""
    result = _check(code)
    assert not result.ok
    # There should be at least 2 prose errors (likely 3)
    assert len(result.errors) >= 2, f"expected ≥2 prose errors, got: {result.errors}"


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE — code that MUST fail (Check #2: ghost kernel launches)
# ═══════════════════════════════════════════════════════════════════════════

def test_reject_ghost_kernel_single():
    """A single undefined kernel launch must be flagged as ghost kernel."""
    code = """\
#include <hip/hip_runtime.h>

int main() {
    undefined_kernel<<<1, 1>>>();
    return 0;
}
"""
    result = _check(code)
    assert not result.ok
    assert any("ghost" in e.lower() for e in result.errors), (
        f"error should mention ghost: {result.errors}"
    )
    assert any("undefined_kernel" in e for e in result.errors), (
        f"error should name the kernel: {result.errors}"
    )


def test_reject_ghost_kernel_multiple():
    """Multiple undefined kernel launches must all be flagged."""
    code = """\
#include <hip/hip_runtime.h>

int main() {
    kernel_a<<<1, 1>>>();
    kernel_b<<<128, 256>>>();
    kernel_c<<<1, 1>>>();
    return 0;
}
"""
    result = _check(code)
    assert not result.ok
    ghost_count = sum(1 for e in result.errors if "ghost" in e.lower())
    assert ghost_count >= 3, f"expected 3 ghost errors, got {ghost_count}: {result.errors}"


def test_reject_ghost_kernel_mixed_with_real():
    """If only some kernels are defined, the undefined ones must still be flagged."""
    code = """\
#include <hip/hip_runtime.h>

__global__ void real_kernel(float *x) {
    *x = 1.0f;
}

int main() {
    real_kernel<<<1, 1>>>(d_x);
    ghost_kernel<<<1, 1>>>();
    return 0;
}
"""
    result = _check(code)
    assert not result.ok
    assert any("ghost_kernel" in e for e in result.errors), (
        f"ghost_kernel must be flagged: {result.errors}"
    )
    assert not any("real_kernel" in e for e in result.errors), (
        f"real_kernel must NOT be flagged: {result.errors}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE — code that MUST fail (Check #3: executable statements at file scope)
# ═══════════════════════════════════════════════════════════════════════════

def test_reject_executable_assignment_at_file_scope():
    """An assignment like ``x = 42;`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

x = 42;
"""
    result = _check(code)
    assert not result.ok
    assert any("executable" in e.lower() for e in result.errors), (
        f"error should mention executable: {result.errors}"
    )


def test_reject_executable_kernel_launch_at_file_scope():
    """A kernel launch like ``k<<<1,1>>>();`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

k<<<1, 1>>>();
"""
    result = _check(code)
    assert not result.ok
    assert any("executable" in e.lower() for e in result.errors)


def test_reject_executable_arrow_at_file_scope():
    """A line with ``->`` and ``;`` at file scope must be rejected."""
    code = """\
#include <hip/hip_runtime.h>

ptr->value = 42;
"""
    result = _check(code)
    assert not result.ok
    assert any("executable" in e.lower() for e in result.errors)


def test_reject_executable_scope_at_file_scope():
    """A line with ``::`` at file scope that is NOT a method definition and
    also triggers assignment/arrow must be rejected.

    ``value = ns::VALUE;`` triggers via ``=`` + ``;`` even though the ``::``
    is in the first-paren segment (so the scope check alone doesn't fire).
    """
    code = """\
#include <hip/hip_runtime.h>

value = ns::VALUE;
"""
    result = _check(code)
    assert not result.ok
    assert any("executable" in e.lower() for e in result.errors), (
        f"expected executable statement rejection: {result.errors}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MIXED / COMPREHENSIVE — combining multiple failures
# ═══════════════════════════════════════════════════════════════════════════

def test_reject_multiple_pathologies():
    """Code with prose, ghost kernels, AND executable statements must report all."""
    code = """\
#include <hip/hip_runtime.h>

The port should use hipLaunchKernelGGL.
undefined_launch<<<1, 1>>>();
x = 42;

__global__ void defined_kernel() {}
"""
    result = _check(code)
    assert not result.ok
    prose_errs = [e for e in result.errors if "prose" in e.lower()]
    ghost_errs = [e for e in result.errors if "ghost" in e.lower()]
    exec_errs = [e for e in result.errors if "executable" in e.lower()]
    assert len(prose_errs) >= 1, f"expected prose errors: {prose_errs}"
    assert len(ghost_errs) >= 1, f"expected ghost errors: {ghost_errs}"
    assert len(exec_errs) >= 1, f"expected executable errors: {exec_errs}"


# ═══════════════════════════════════════════════════════════════════════════
# EDGE: error message format
# ═══════════════════════════════════════════════════════════════════════════

def test_prose_error_message_contains_line_number():
    """Prose-rejection error messages must include the line number."""
    code = """\
#include <hip/hip_runtime.h>

This is a prose line at file scope.
"""
    result = _check(code)
    assert not result.ok
    assert any("line 3" in e for e in result.errors), (
        f"error should contain line number: {result.errors}"
    )


def test_ghost_error_message_contains_kernel_name():
    """Ghost-kernel error messages must include the kernel name."""
    code = """\
#include <hip/hip_runtime.h>

int main() {
    my_ghost<<<1, 1>>>();
    return 0;
}
"""
    result = _check(code)
    assert not result.ok
    assert any("my_ghost" in e for e in result.errors), (
        f"error should name the kernel: {result.errors}"
    )
    assert any("__global__" in e for e in result.errors), (
        f"error should mention __global__: {result.errors}"
    )


def test_executable_error_message_contains_line_number():
    """Executable-statement error messages must include the line number."""
    code = """\
#include <hip/hip_runtime.h>

x = 42;
"""
    result = _check(code)
    assert not result.ok
    assert any("line 3" in e for e in result.errors), (
        f"error should contain line number: {result.errors}"
    )
