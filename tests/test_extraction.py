"""Tests for verification.extraction — the provider-agnostic code extractor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from verification.extraction import extract_code  # noqa: E402
from verification.lexical import validate_lexical  # noqa: E402


CLEAN_KERNEL = """\
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vec_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
"""


# ── Extraction: happy paths ────────────────────────────────────────────────

def test_markdown_hip_fence():
    resp = f"Here is the port:\n\n```hip\n{CLEAN_KERNEL}```\n\nHope it helps!"
    r = extract_code(resp)
    assert r.ok
    assert r.strategy == "markdown-fence"
    assert "__global__" in r.code
    assert "Hope it helps" not in r.code


def test_markdown_cpp_fence():
    resp = f"```cpp\n{CLEAN_KERNEL}```"
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code


def test_untagged_fence_still_scored():
    resp = f"```\n{CLEAN_KERNEL}```"
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code


def test_multiple_fences_prefers_code_shaped():
    resp = f"""\
Here's the plan:

```json
{{"summary": "swap warp for wavefront"}}
```

And the port:

```cpp
{CLEAN_KERNEL}```
"""
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code
    assert "summary" not in r.code


def test_json_ported_code_field():
    payload = json.dumps({"ported_code": CLEAN_KERNEL, "confidence": 88})
    r = extract_code(payload)
    assert r.ok
    assert r.strategy == "json-field"
    assert "__global__" in r.code


def test_json_field_inside_markdown_json_block():
    resp = "```json\n" + json.dumps({"code": CLEAN_KERNEL}) + "\n```"
    r = extract_code(resp)
    assert r.ok
    assert r.strategy == "json-field"


def test_json_truncated_still_recovers_code_field():
    """Truncated JSON: strict parse fails, hand-extract wins."""
    body = json.dumps(CLEAN_KERNEL)  # a JSON-encoded string
    # Deliberately truncate the outer object before the closing brace.
    truncated = '{"ported_code": ' + body + ', "confidence": 88'
    r = extract_code(truncated)
    assert r.ok
    assert "__global__" in r.code


def test_reasoning_before_code_raw():
    resp = f"""\
Okay, so I need to port this kernel.  Let's think about what changes.
First, warpSize→64.  Second, adjust the masks.

Here is the ported kernel:

{CLEAN_KERNEL}
"""
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code
    # The reasoning preamble must not be in the extracted code.
    assert "Let's" not in r.code
    assert "Okay" not in r.code


def test_reasoning_after_code_raw():
    resp = f"""\
{CLEAN_KERNEL}

That should compile fine.  Let me know if you need adjustments.
"""
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code
    assert "let me know" not in r.code.lower()


# ── Extraction: adversarial cases ──────────────────────────────────────────

def test_pure_prose_rejects():
    resp = "This kernel adds two vectors element-wise.  It uses 256 threads."
    r = extract_code(resp)
    assert r.code is None
    assert r.strategy == "rejected"


def test_json_field_with_prose_body_is_still_extracted_but_lexical_rejects():
    """The exact failure mode from the bug report: JSON puts prose in the code field.

    Extraction correctly returns the field body, but the lexical gate
    downstream will refuse to write it.  This test asserts BOTH: the
    extractor is honest (it returns what the JSON said), and the lexical
    layer catches it.
    """
    prose = (
        "#include <iostream>\n"
        "#include <hip/hip_runtime.h>\n"
        "\n"
        "local headers\\\". So we need inline the cuh.\n"
        "\n"
        "Let's search memory more concretely.\n"
        "\n"
        "I think...\n"
        "\n"
        "Wait maybe...\n"
        "\n"
        "Let's design our own kernels...\n"
    )
    resp = '{"ported_code": "' + prose.replace("\n", "\\n") + '"}'
    r = extract_code(resp)
    assert r.ok  # extractor honestly returns the field body
    lex = validate_lexical(r.code)
    assert not lex.ok, "lexical gate must reject the extracted prose"


def test_truncated_response_no_close_fence():
    """A response that started a fence and got cut off mid-body."""
    resp = f"```cpp\n{CLEAN_KERNEL}"  # no closing ```
    r = extract_code(resp)
    # The fence never closed, so strategy 2 finds no block.  Strategy 3
    # (raw window) picks up the code from the #include.
    assert r.ok
    assert "__global__" in r.code


def test_malformed_code_fence_backtick_count():
    """Four backticks in a row — not a valid fence."""
    resp = f"````cpp\n{CLEAN_KERNEL}```"
    r = extract_code(resp)
    # Fenced-block regex won't match cleanly, but raw window still picks up
    # the code — the extractor prefers "we got something" over "we got nothing".
    assert r.ok
    assert "__global__" in r.code


def test_streaming_interruption_mid_kernel():
    """Only the first half of the kernel arrived."""
    resp = "```cpp\n#include <hip/hip_runtime.h>\n\n__global__ void k(float *"
    r = extract_code(resp)
    # No closing fence and no obvious end-of-code; the raw window path may
    # still return something.  The important guarantee is: whatever comes
    # back is not full of prose.
    if r.ok:
        assert "Let's" not in r.code
        assert "I think" not in r.code


def test_duplicate_code_blocks_picks_larger():
    small = "```cpp\nint x = 1;\n```"
    big = f"```cpp\n{CLEAN_KERNEL}```"
    resp = f"{small}\n\nAnd the full port:\n\n{big}"
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code


def test_conversation_mixed_with_code():
    resp = f"""\
Sure!  Here is what I recommend.

```cpp
{CLEAN_KERNEL}```

Let me know if you'd like a different tile size.  I can also add shared
memory blocking if that's useful.  Just say the word.
"""
    r = extract_code(resp)
    assert r.ok
    assert "__global__" in r.code
    assert "Let me know" not in r.code
    assert "Just say the word" not in r.code


def test_diagnostics_are_populated():
    resp = f"```cpp\n{CLEAN_KERNEL}```"
    r = extract_code(resp)
    assert r.response_length == len(resp)
    assert r.code_length > 0
    assert r.candidates_considered >= 1


def test_empty_response():
    assert not extract_code("").ok
    assert not extract_code(None).ok
    assert extract_code("").strategy == "rejected"


# ── End-to-end: extractor + lexical form a safe pipe ───────────────────────

def test_pipe_bug_report_response_never_writes_prose():
    """End-to-end: extractor → lexical.

    A response identical in shape to the observed failure must not survive
    the two-stage pipe: whatever comes out of extraction must be rejected
    by the lexical gate.
    """
    resp = (
        '{"ported_code": "'
        "#include <iostream>\\n"
        "#include <hip/hip_runtime.h>\\n"
        "\\n"
        "local headers. So we need inline the cuh.\\n"
        "\\n"
        "Let's search memory more concretely.\\n"
        '"}'
    )
    r = extract_code(resp)
    # Extraction returned SOMETHING (the JSON field body), and now the
    # lexical layer has to catch it.
    if r.ok:
        assert not validate_lexical(r.code).ok
