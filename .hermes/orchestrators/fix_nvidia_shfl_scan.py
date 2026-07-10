#!/usr/bin/env python3
"""
Kernel Olympics — nvidia_shfl_scan Fix Orchestrator

Pythonic AI orchestrator: dispatches subagents per skill, validates
preconditions, gates progression. Implements Part A + Part B of the
Portable Subset fix plan (merged per TRIZ #10/#20 — budget re-allocation
IS a precondition for Part A).

Phases:
  1. PREFLIGHT — validate current state (spec, code structure)
  2. PART_A — PortMode enum, spec fix, harness update, coder prompt
  3. PART_B — budget re-allocation (cap initial gen, protect repair reserve)
  4. PART_C — honest logging, stage contracts, symbol-diff scope
  5. VERIFY — run tests, check nothing regressed
  6. COMMIT — commit and push

Each phase is a subagent with its own skill(s), running independently.
"""
import json
import re
import sys
from pathlib import Path

from hermes_tools import (
    delegate_task,
    terminal,
    read_file,
    write_file,
    patch,
    search_files,
    web_search,
    web_extract,
)


def preflight_check() -> dict:
    """Validate repo state before dispatching."""
    issues = []
    spec_path = Path("src/verification/specs/nvidia_shfl_scan.json")
    if not spec_path.exists():
        issues.append("MISSING: nvidia_shfl_scan.json spec")
    else:
        spec = json.loads(spec_path.read_text())
        if spec.get("output_readback", {}).get("element_type") == "float":
            issues.append("SPEC BUG: output_readback.element_type is 'float', should be 'int'")
        if spec.get("self_contained") is not True:
            issues.append("SPEC BUG: self_contained should be True (original source has main())")

    # Check _generate_harness uses self_contained
    vpath = "src/verification/verifier.py"
    content = Path(vpath).read_text()
    if 'spec.get("self_contained", False)' in content:
        issues.append("VERIFIER: _generate_harness reads spec.self_contained — needs port_mode")

    # Check router has _unsatisfied_main_calls
    rpath = "src/router.py"
    rcontent = Path(rpath).read_text()
    if "_unsatisfied_main_calls" not in rcontent:
        issues.append("ROUTER: missing _unsatisfied_main_calls")

    return {
        "spec_path": str(spec_path),
        "spec": spec if spec_path.exists() else None,
        "issues": issues,
        "pass": len(issues) == 0,
    }


def phase_part_a(preflight: dict) -> str:
    """Part A: PortMode enum, spec fix, DEVICE_SUBSET coder prompt, harness fix."""
    return delegate_task(
        goal="""Implement PART A of the nvidia_shfl_scan fix plan:
1. Add PortMode enum (WHOLE_PROGRAM, DEVICE_SUBSET) to the router
2. Compute port_mode once at route() start from _is_self_contained(original) + _unsatisfied_main_calls() + _unresolved_local_headers()
3. Fix nvidia_shfl_scan.json spec: output_readback.element_type → int, format → int_per_line, port_mode → DEVICE_SUBSET
4. Update _generate_harness in verifier.py to read port_mode instead of self_contained
5. Update coder prompt for DEVICE_SUBSET mode: "port ONLY __global__/__device__ functions, drop host driver"
6. Ensure _ensure_main_preserved skips restoration in DEVICE_SUBSET mode
7. Create test fixture: hand-written wavefront64-correct HIP port of shfl_scan_test + uniform_add

IMPORTANT CONTEXT:
- Working dir: /root/Kernel-Olympics
- Files you touch: src/router.py, src/verification/verifier.py, src/verification/specs/nvidia_shfl_scan.json, tests/
- Current model roles: GLM=coder (16384 tok), Kimi=evaluator (1024 tok), DeepSeek=planner
- self_contained field in spec means "original CUDA source has main()" — it's TRUE for nvidia_shfl_scan
- But the ported code WILL NOT have main() because the host driver depends on missing headers
- _unsatisfied_main_calls() already exists and returns blockers for missing headers
- _generate_harness() at line 236 reads spec.get("self_contained") — change to spec.get("port_mode")
- port_mode enum values: "WHOLE_PROGRAM" or "DEVICE_SUBSET"
- Compute port_mode in route() before building any prompt, persist in result["port_mode"]

CODE EXAMPLES:
```python
from enum import Enum

class PortMode(str, Enum):
    WHOLE_PROGRAM = "WHOLE_PROGRAM"
    DEVICE_SUBSET = "DEVICE_SUBSET"

    @classmethod
    def compute(cls, original_source: str, router) -> "PortMode":
        \"\"\"Single-source portability decision.\"\"\"
        if not cls._is_self_contained(original_source):
            return cls.WHOLE_PROGRAM  # no main → need harness
        # Has main() — check if driver is portable
        main_text = cls._extract_main(original_source)
        unsatisfied = cls._unsatisfied_main_calls(
            main_text, original_source, original_source
        )
        if unsatisfied:
            return cls.DEVICE_SUBSET
        return cls.WHOLE_PROGRAM
```

INSERT PortMode class near top of router.py (after imports, before Deadline class).
Add a `_compute_port_mode` static method.
Call it in route() right after kernel_source is available, store in result.
Update _generate_harness to read port_mode from spec or result dict.

DO NOT: touch budget code, timeout code, or architecture hardening — those are Part B/C.
DO: write tests for PortMode.compute() and _generate_harness(port_mode=DEVICE_SUBSET).
RUN: pytest tests/ -q --tb=line after changes.

CRITICAL: Do NOT touch _call_model call sites, model keys, or SYSTEM_PROMPTS — those were already swapped.
""",
        role="leaf",
    )


def phase_part_b() -> str:
    """Part B: budget re-allocation — cap initial gen, protect repair reserve."""
    return delegate_task(
        goal="""Implement PART B of the nvidia_shfl_scan fix plan:
Re-allocate the 180s pipeline budget so a full compile→patch→recompile cycle can run.

Observed timeline (from failing trace):
- Plan: 36s cap, 22.2s used
- Codegen: ~132s cap (remaining - 25s reserve), 89.9s used  
- Refine: remaining - 25s, 43.4s used
- Left: ~24s → "no room to retry"

REQUIRED CHANGES:
1. Add per-stage budget constants at top of route():
   - MAX_PIPELINE_SECONDS = 180 (keep existing)
   - PLAN_CAP = 30
   - CODEGEN_CAP = 70  (was effectively ~132)
   - COMPILE_RESERVE = 25 (keep existing)
   - REPAIR_RESERVE = 60  (NEW — protected, cannot be borrowed by codegen)
   - VERIFY_CAP = 15
2. Implement stage budget enforcement: each phase gets a max_seconds computed from its cap, NOT from `deadline.remaining() - COMPILE_RESERVE_SECONDS`
3. The COMPILE_RESERVE_SECONDS is always protected — no stage may borrow it
4. The REPAIR_RESERVE is protected until first compile failure — then released
5. Add adaptive stop: if same error signature repeats twice in a row (structural or compile), abort early with "same_error_repeated" reason instead of burning another iteration

KEY CONSTRAINTS:
- Do NOT raise MAX_PIPELINE_SECONDS
- Do NOT change max_tokens, model selection, or prompt logic
- The repair reserve ensures at least one real compile→fix→recompile cycle
- After compile passes, release the repair reserve for verification

FILES TO TOUCH:
- src/router.py — budget constants, _compute_stage_budget() method, stage cap enforcement in route()

Write tests:
- test that codegen cannot borrow from repair reserve
- test that repair reserve releases after first compile
- test that duplicate error signature triggers early abort

CRITICAL: Do NOT touch Part A code (PortMode, spec, harness) or Part C (logging, stage contracts). Only budget/timing code.
""",
        role="leaf",
    )


def phase_part_c(preflight: dict) -> str:
    """Part C: honest logging, stage contracts, symbol-diff scope."""
    return delegate_task(
        goal="""Implement PART C of the nvidia_shfl_scan fix plan:
Honest logging, stage contracts, and symbol-diff scoping.

REQUIRED CHANGES:

C.1 Honest logging — fix log messages that lie:
- In route(), any "refining with compile errors" or similar log must check: was hipcc actually called this iteration?
- If structural reject → "refining from structural feedback" 
- If lexical reject → "refining from lexical feedback"
- Only if compile ran → "refining from compiler diagnostics"
- Search for all print() f-strings mentioning "compile" and add the gate check

C.2 Stage contracts — IterationState already exists; extend it:
- Add repair_mode field: str = "lexical" | "structural" | "compiler"
- Set it when the stage is entered (before the repair)
- Print it in debug display

C.3 Symbol-diff scope (DEVICE_SUBSET):
- In symbols.py or wherever the symbol diff lives: when port_mode is DEVICE_SUBSET,
  exclude intentionally-dropped host symbols (main, shuffle_*_test, etc.)
- Add test: symbol diff on DEVICE_SUBSET kernel does NOT flag missing host functions
- Only device symbols (__global__, __device__) are in the expected set

FILES TO TOUCH:
- src/router.py — honest logging, repair_mode in IterationState
- src/verification/symbols.py — scope symbol diff by port_mode
- tests/ — test for symbol diff scoping

Write tests:
- test_repair_mode_reflects_real_source
- test_symbol_diff_excludes_host_code_in_device_subset

CRITICAL: Do NOT touch Part A (PortMode, spec, harness) or Part B (budget). Only logging/stage contracts/symbols.
""",
        role="leaf",
    )


def verify_all() -> dict:
    """Run all tests and check spec integrity."""
    result = terminal("cd /root/Kernel-Olympics && python3 -m pytest tests/ -q --tb=line 2>&1")
    test_output = result.get("output", "")

    # Parse test results
    passed = failed = 0
    if match := re.search(r"(\d+) passed", test_output):
        passed = int(match.group(1))
    if match := re.search(r"(\d+) failed", test_output):
        failed = int(match.group(1))

    # Check spec integrity
    spec_path = Path("src/verification/specs/nvidia_shfl_scan.json")
    spec_ok = False
    spec_output = ""
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        readback_type = spec.get("output_readback", {}).get("element_type", "")
        has_port_mode = "port_mode" in spec
        spec_output = f"readback_type={readback_type}, port_mode={has_port_mode}"
        spec_ok = readback_type == "int" and has_port_mode

    return {
        "tests": {"passed": passed, "failed": failed, "output": test_output},
        "spec": {"ok": spec_ok, "details": spec_output},
        "pass": failed == 0 and spec_ok,
    }


def commit_and_push() -> dict:
    """Commit and push all changes."""
    result = terminal(
        'cd /root/Kernel-Olympics && '
        'git add -A && '
        'git commit -m "fix: PortMode + budget re-allocation + honest logging for nvidia_shfl_scan" '
        '&& git push 2>&1'
    )
    output = result.get("output", "")
    commit_hash = ""
    if m := re.search(r"\[main ([a-f0-9]+)\]", output):
        commit_hash = m.group(1)
    return {
        "output": output,
        "commit": commit_hash,
        "pass": bool(commit_hash),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🧠 KERNEL OLYMPICS — nvidia_shfl_scan Fix Orchestrator")
    print("=" * 70)

    # ── Phase 0: Preflight ──
    print("\n📋 PHASE 0: PREFLIGHT")
    pre = preflight_check()
    print(f"  Issues: {len(pre['issues'])}")
    for issue in pre["issues"]:
        print(f"    ⚠️  {issue}")

    # ── Phase 1-3: Dispatch subagents in parallel ──
    print("\n🚀 PHASE 1-3: DISPATCHING SUBAGENTS")
    a_handle = phase_part_a(pre)
    b_handle = phase_part_b()
    c_handle = phase_part_c(pre)

    print(f"  PART A (PortMode+spec+harness):   dispatched → {a_handle}")
    print(f"  PART B (budget re-allocation):     dispatched → {b_handle}")
    print(f"  PART C (logging+contracts+symbols): dispatched → {c_handle}")

    # Subagents run in background — results arrive as separate messages.
    # When they're done, the user (or a follow-up orchestrator) can call verify_all()
    # and commit_and_push().
    print("\n⏳ Subagents are running in the background...")
    print("💡 Their results will appear as separate messages here.")
    print("   After all three complete, run:")
    print("     python3 .hermes/orchestrators/fix_nvidia_shfl_scan.py --verify")
    print("   to run tests and commit.")
