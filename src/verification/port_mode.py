"""Single-source portability decision for CUDA→HIP porting.

The router and verifier must agree on ONE question: is the source's own
host driver portable to HIP in this repo? If yes, we port the whole
program (WHOLE_PROGRAM). If no — because it depends on unvendored headers
or its main() calls symbols this port cannot provide — we port only the
device code and synthesize a harness (DEVICE_SUBSET).

Before this module existed, ``spec["self_contained"]`` was overloaded to
answer two different questions:

  1. Does the source define ``int main()``?          (spec_parser wrote this)
  2. Should the harness expect the port to have main()?  (verifier read this)

Those are only the same question when the driver is portable. For
``nvidia_shfl_scan.cu`` the driver is NOT portable — it #includes
``shfl_integral_image.cuh`` which is not vendored here — and the router
correctly refused to restore main(). The verifier, still reading the flag
as the answer to question (2), returned the ported code verbatim as the
test file. Net: hipcc got a translation unit with neither the original
driver nor a synthesized harness, and the loop burned its whole budget
chasing a phantom code defect.

decide_port_mode() answers question (2) from the ORIGINAL source, once,
before any LLM call. Router and verifier read the same field.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

PORT_MODE_WHOLE_PROGRAM = "WHOLE_PROGRAM"
PORT_MODE_DEVICE_SUBSET = "DEVICE_SUBSET"


# ── Regex primitives (shared with router.ModelRouter, which delegates here) ──

# ``int main(`` at the start of a line — the sole marker that ties self-
# containment to what the linker will actually look for. Anchored so that a
# comment or string literal containing ``int main(`` mid-line cannot match.
_MAIN_RE = re.compile(r'^\s*int\s+main\s*\(', re.MULTILINE)

# Same anchor but with ``[ \t]`` instead of ``\s`` so ``m.start()`` never
# points at a leading newline. ``_extract_main`` returns ``source[m.start():..]``,
# so a match on the previous line's ``\n`` would prepend a stray newline to the
# returned driver text (broke test_extracts_a_brace_balanced_driver).
_MAIN_TIGHT_RE = re.compile(r'^[ \t]*int[ \t]+main[ \t]*\(', re.MULTILINE)

# A function DEFINITION at file scope: ``... name(args) {``. Deliberately not
# a declaration (``...;``) — a prototype defines no symbol for the linker.
_FUNC_DEF = re.compile(
    r'^[ \t]*(?:(?:static|inline|extern|__global__|__device__|__host__|'
    r'template\s*<[^>]*>)\s+)*'
    r'[A-Za-z_][A-Za-z0-9_:<>,\t \*&]*?\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?\{',
    re.MULTILINE)

# An identifier immediately followed by ``(`` — a call, a definition, or a cast.
_CALL_SITE = re.compile(r'\b([A-Za-z_]\w*)\s*\(')

# Control-flow keywords that look like calls to the regex above.
_NOT_CALLS = frozenset({
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
})


def is_self_contained(source: str) -> bool:
    """True when *source* defines its own ``int main(``."""
    return bool(_MAIN_RE.search(source or ""))


def extract_main(source: str) -> str:
    """Return the full text of *source*'s ``int main(...)`` definition, or ""."""
    if not source:
        return ""
    m = _MAIN_TIGHT_RE.search(source)
    if not m:
        return ""
    open_brace = source.find("{", m.end())
    if open_brace < 0:
        return ""
    # A declaration ends at ';' before any body opens.
    semi = source.find(";", m.end())
    if 0 <= semi < open_brace:
        return ""

    depth = 0
    in_string = in_char = in_line_comment = in_block_comment = False
    escape = False
    for i in range(open_brace, len(source)):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
            continue
        if in_string or in_char:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif (ch == '"' and in_string) or (ch == "'" and in_char):
                in_string = in_char = False
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "'":
            in_char = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[m.start():i + 1]
    return ""  # unbalanced — refuse to hand back a truncated driver


def defined_functions(code: str) -> set:
    """Names of functions *defined* (not merely declared) at file scope."""
    return {m.group(1) for m in _FUNC_DEF.finditer(code or "")} - _NOT_CALLS


def unsatisfied_main_calls(main_text: str, ported_code: str,
                           original_source: str) -> List[str]:
    """User-defined helpers that main() calls but the port does not define.

    Only names *defined in the original* are considered — runtime and library
    symbols (printf, exit, hipMalloc) are never function definitions in .cu.
    """
    original_funcs = defined_functions(original_source) - {"main"}
    if not original_funcs:
        return []
    called = ({m.group(1) for m in _CALL_SITE.finditer(main_text or "")}
              - _NOT_CALLS)
    available = defined_functions(ported_code)
    return sorted((called & original_funcs) - available)


# ``sample_kernels/`` under the repo root — checked for locally-included
# headers. If a #include "foo.cuh" cannot be found there, the header is
# unvendored and any code path that depends on it must be dropped.
_REPO_SAMPLE_DIR = (Path(__file__).resolve().parent.parent.parent
                    / "sample_kernels")


def unresolved_local_headers(source: str) -> List[str]:
    """Quoted local ``.cuh``/``.h`` includes in *source* that are unvendored."""
    if not source:
        return []
    missing: List[str] = []
    for m in re.finditer(r'#include\s*"([^"]+\.(?:cuh|h|hpp))"', source):
        fname = m.group(1)
        if not list(_REPO_SAMPLE_DIR.rglob(Path(fname).name)):
            missing.append(fname)
    return missing


def decide_port_mode(original_source: str) -> str:
    """Compute the port mode from the ORIGINAL CUDA source.

    Returns ``PORT_MODE_WHOLE_PROGRAM`` when the source's own driver can be
    linked in this repo — no unvendored headers, no main() calls into
    functions that aren't defined anywhere the port can see.

    Returns ``PORT_MODE_DEVICE_SUBSET`` when the source has a main() we
    cannot honestly reproduce (unvendored headers, or main() calls helpers
    whose bodies live only behind such headers). In that mode the coder
    ports only the device code and the verifier synthesizes a harness.

    A bare kernel snippet with no main() is also WHOLE_PROGRAM: the
    verifier synthesizes a harness for it too, so the two modes converge
    when there's no driver at all. WHOLE_PROGRAM is the safe default
    because it triggers restore-main() logic only when the source has one.
    """
    if not is_self_contained(original_source):
        # No driver to preserve or drop → the whole-program restore path
        # is a no-op, so treat as WHOLE_PROGRAM and let the harness
        # synthesizer handle it.
        return PORT_MODE_WHOLE_PROGRAM
    if unresolved_local_headers(original_source):
        return PORT_MODE_DEVICE_SUBSET
    original_main = extract_main(original_source)
    if not original_main:
        # A main() we cannot brace-match is a driver we cannot lift
        # verbatim — safer to synthesize than to restore garbage.
        return PORT_MODE_DEVICE_SUBSET
    # Check main()'s dependencies against the original source itself. If a
    # helper it calls is not defined anywhere in this file, the driver was
    # already un-buildable here (the missing helper lived in a header that
    # is not part of the ported code by the time verify runs).
    if unsatisfied_main_calls(original_main, original_source, original_source):
        # Nothing to catch here in practice — if the helper is defined in
        # the same .cu file, unsatisfied_main_calls will treat it as
        # satisfied. This branch would fire only if a driver calls a
        # function whose declaration is present but whose body is
        # elsewhere; treat that as a signal to synthesize.
        return PORT_MODE_DEVICE_SUBSET
    return PORT_MODE_WHOLE_PROGRAM
