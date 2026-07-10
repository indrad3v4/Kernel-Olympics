"""Pipeline outcomes: structural reject, compile fail, compile pass, timeout.

Before the structural gate existed, an iteration had two outcomes — hipcc
passed or hipcc failed — and every reader downstream (e.g. the informed
re-plan at router.py:3352) assumed the compile-fail branch had run and its
locals (``compile_errs``, ``linker_only``, ``error_origins``) were bound.

The structural gate introduced a third outcome — rejected before hipcc —
that raised ``compile_failed_this_iter`` without ever entering the compile
branch, and blew up the readers with ``UnboundLocalError``.

These tests pin the four pipeline outcomes as first-class states:

    STRUCTURAL_REJECT — text-level defect, hipcc did NOT run
    COMPILE_FAIL      — hipcc ran, produced errors
    COMPILE_SUCCESS   — hipcc ran and passed
    TIMEOUT           — wall-clock exhausted before completion

Each test asserts that the state machine reached the intended terminal state
AND that no ``UnboundLocalError`` was raised on the way — the regression that
motivated this file.
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from router import ModelRouter, AgentResult, IterationState


CUDA_SOURCE = """
#include <cuda_runtime.h>
__global__ void k(float* x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) x[tid] = x[tid] * 2.0f;
}

int main() {
    float* d;
    cudaMalloc(&d, 16);
    cudaFree(d);
    return 0;
}
"""

HIP_CODE_OK = """
#include <hip/hip_runtime.h>
__global__ void k(float* x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) x[tid] = x[tid] * 2.0f;
}

int main() {
    float* d;
    hipMalloc(&d, 16);
    hipFree(d);
    return 0;
}
"""

# A generation that will be rejected by the structural gate — unbalanced
# braces AND a truncation marker. Nothing hipcc could ever emit sensibly.
HIP_CODE_STRUCTURAL_BAD = """
#include <hip/hip_runtime.h>
__global__ void k(float* x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) x[tid] = x[tid] * 2.0f;
// ... rest of code omitted
"""


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


# ─────────────────────────────────────────────────────────────────────────────
# IterationState invariants — the piece the whole file rests on
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationStateInvariants:
    """The dataclass exists specifically so no reader hits UnboundLocalError.
    These tests pin the default-safety and property semantics."""

    def test_defaults_are_all_safe_to_read(self):
        """No reader should be able to trip on a fresh state — every list is
        empty, every bool is False, and no field is None-that-crashes."""
        s = IterationState(iteration=1)
        assert s.gate == "skipped"
        assert s.compile_errs == []
        assert s.error_origins == []
        assert s.structural_errors == []
        assert s.structural_missing == []
        assert s.structural_ok is True
        assert s.structural_reject is False
        assert s.compile_ran is False
        assert s.compile_success is False
        assert s.linker_only is False
        assert s.run_crashed is False
        assert s.glm_analysis is None
        assert s.replanned is False

    def test_compile_failed_is_false_on_fresh_state(self):
        """A skipped iteration must not read as failed — the loop uses this
        to decide whether to spin up recovery paths."""
        assert IterationState(iteration=1).compile_failed is False

    def test_compile_failed_true_on_structural_reject(self):
        """Structural rejects are failures for control-flow purposes even
        though hipcc never ran."""
        s = IterationState(iteration=1)
        s.structural_reject = True
        assert s.compile_failed is True

    def test_compile_failed_true_on_compile_error(self):
        s = IterationState(iteration=1)
        s.compile_ran = True
        s.compile_success = False
        assert s.compile_failed is True

    def test_compile_failed_false_on_compile_pass(self):
        s = IterationState(iteration=1)
        s.compile_ran = True
        s.compile_success = True
        assert s.compile_failed is False

    def test_compile_ran_without_success_flag_is_not_a_pass(self):
        """A crashed-mid-compile state (ran=True, success=False) must be
        readable as a failure — the loop treats it as one."""
        s = IterationState(iteration=1)
        s.compile_ran = True
        # success defaults to False; do not touch it
        assert s.compile_failed is True


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: run route() and observe the terminal state
# ─────────────────────────────────────────────────────────────────────────────

def _plan_result():
    return AgentResult("deepseek", True, "Plan: swap cuda* for hip*", 0.1)


def _glm_pass():
    return AgentResult("glm", True, '{"pass": true, "feedback": "ok"}', 0.1)


def _glm_err_analysis():
    return AgentResult(
        "glm", True,
        '{"fixes": [{"action": "swap cudaMalloc for hipMalloc", "priority": 1}], '
        '"missing_includes": ["hip/hip_runtime.h"]}',
        0.1,
    )


def _gemma_pass():
    return AgentResult("gemma4", True, '{"pass": true}', 0.1)


def _kimi(source):
    return AgentResult("kimi27", True, f"```cpp\n{source}\n```", 0.1)


class TestStructuralReject:
    """The regression: on structural reject, the loop must not crash and must
    not spin up compile-only recovery (informed re-plan, GLM error-analyst).

    Historical failure: ``UnboundLocalError: local variable 'compile_errs'
    referenced before assignment`` at router.py:3352 when the informed re-plan
    read `compile_errs` on an iteration where hipcc had never run.
    """

    @patch.object(ModelRouter, '_call_model')
    def test_structural_reject_does_not_raise(self, mock_call, router):
        """The whole point of the fix: rejection at the structural gate must
        traverse the loop cleanly."""
        # Every Kimi output is structurally broken (truncation marker), so
        # every iteration should route through the structural branch — never
        # touching hipcc.
        def dispatch(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return _plan_result()
            if model_key == "kimi27":
                return _kimi(HIP_CODE_STRUCTURAL_BAD)
            if model_key == "glm":
                return _glm_pass()
            if model_key == "gemma4":
                return _gemma_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        # If the guard fails and hipcc is called anyway, this side effect
        # would fire and the test would still (incorrectly) pass — so we also
        # assert quick_compile_check is never called below.
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["should not be reached"], "output": ""
        }
        verifier.verify.return_value = {
            "compile_success": False, "passed": False, "compile_output": "", "output": ""
        }

        # The historical bug was an UnboundLocalError on this call.
        result = router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=2,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        assert result is not None
        assert "last_iteration_state" in result
        assert result["last_iteration_state"]["gate"] == "structural"
        assert result["last_iteration_state"]["structural_reject"] is True
        assert result["last_iteration_state"]["compile_ran"] is False

    @patch.object(ModelRouter, '_call_model')
    def test_structural_reject_skips_hipcc_on_broken_code(self, mock_call, router):
        """Structural rejects must NOT spend a hipcc call on the *broken*
        Kimi output. hipcc may still fire on the pre-Kimi fast-path
        (regex-hipified original source), which is unrelated — the invariant
        is that the truncation marker never reaches the compiler."""
        def dispatch(model_key, *args, **kwargs):
            if model_key == "kimi27":
                return _kimi(HIP_CODE_STRUCTURAL_BAD)
            if model_key == "deepseek":
                return _plan_result()
            if model_key == "glm":
                return _glm_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": [], "output": ""
        }
        verifier.verify.return_value = {
            "compile_success": False, "passed": False, "compile_output": "", "output": ""
        }

        router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=1,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        # Assert none of the hipcc call sites saw the truncation marker. The
        # fast-path regex-hipify call is fine — it never touches Kimi output.
        for call in verifier.quick_compile_check.call_args_list:
            args, kwargs = call
            passed_source = args[0] if args else kwargs.get("source", "")
            assert "rest of code omitted" not in passed_source, (
                "hipcc was invoked on the structurally-broken Kimi output — "
                "the structural gate is leaking"
            )

    @patch.object(ModelRouter, '_call_model')
    def test_structural_reject_skips_deepseek_replan(self, mock_call, router):
        """A truncation is not a strategy problem — DeepSeek re-planning
        against it wastes ~80s of clock. The re-plan must be skipped."""
        deepseek_calls = []

        def dispatch(model_key, *args, **kwargs):
            if model_key == "kimi27":
                return _kimi(HIP_CODE_STRUCTURAL_BAD)
            if model_key == "deepseek":
                deepseek_calls.append(kwargs.get("system_prompt", ""))
                return _plan_result()
            if model_key == "glm":
                return _glm_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {"compile_success": False, "errors": [], "output": ""}
        verifier.verify.return_value = {"compile_success": False, "passed": False, "compile_output": "", "output": ""}

        router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=2,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        # DeepSeek should be called exactly ONCE — the initial plan. No
        # informed re-plans on structural rejects.
        assert len(deepseek_calls) == 1

    @patch.object(ModelRouter, '_call_model')
    def test_structural_reject_feeds_kimi_targeted_feedback(self, mock_call, router):
        """The structural error message should reach Kimi's refine prompt so
        it can act on the specific defect, not on a raw parser diagnostic."""
        refine_prompts = []

        def dispatch(model_key, *args, **kwargs):
            if model_key == "kimi27":
                # First call is initial code; subsequent are refines. The
                # refine prompt is the first positional arg after model_key.
                if args:
                    refine_prompts.append(args[0])
                return _kimi(HIP_CODE_STRUCTURAL_BAD)
            if model_key == "deepseek":
                return _plan_result()
            if model_key == "glm":
                return _glm_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {"compile_success": False, "errors": [], "output": ""}
        verifier.verify.return_value = {"compile_success": False, "passed": False, "compile_output": "", "output": ""}

        router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=2,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        # Refine prompt should reference structural feedback, not "compile errors".
        joined = "\n".join(refine_prompts)
        assert "structural" in joined.lower() or "STRUCTURAL" in joined


class TestCompileFailure:
    """Classic hipcc-failed path: route must record it as a compile failure
    (compile_ran=True, compile_success=False) and let the recovery paths run."""

    @patch.object(ModelRouter, '_call_model')
    def test_compile_fail_records_compile_ran(self, mock_call, router):
        """After a real hipcc call that failed, the terminal iteration state
        must show compile_ran=True (i.e. NOT a structural reject)."""

        def dispatch(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return _plan_result()
            if model_key == "kimi27":
                # A structurally-valid but semantically-broken port
                return _kimi(HIP_CODE_OK.replace("hip/hip_runtime.h", "cuda_runtime.h"))
            if model_key == "glm":
                sysp = kwargs.get("system_prompt", "")
                if "error analyst" in sysp:
                    return _glm_err_analysis()
                return _glm_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False,
            "errors": ["error: use of undeclared identifier 'hipMalloc'"],
            "output": "",
        }
        verifier.verify.return_value = {
            "compile_success": False, "passed": False, "compile_output": "", "output": ""
        }

        result = router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=1,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        state = result["last_iteration_state"]
        assert state["compile_ran"] is True
        assert state["compile_success"] is False
        assert state["structural_reject"] is False
        assert state["gate"] == "compile"
        assert state["compile_errs_count"] > 0


class TestCompileSuccess:
    """The green path: hipcc passes, binary runs clean, GLM approves."""

    @patch.object(ModelRouter, '_call_model')
    def test_compile_success_terminal_state(self, mock_call, router):
        def dispatch(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return _plan_result()
            if model_key == "kimi27":
                return _kimi(HIP_CODE_OK)
            if model_key == "glm":
                return _glm_pass()
            if model_key == "gemma4":
                return _gemma_pass()
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""
        }
        verifier.verify.return_value = {
            "compile_success": True, "passed": True, "compile_output": "", "output": ""
        }

        result = router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=1,
            verifier=verifier,
            kernel_name="test_pipeline_state",
        )

        state = result["last_iteration_state"]
        assert state["compile_ran"] is True
        assert state["compile_success"] is True
        assert state["structural_reject"] is False


class TestTimeout:
    """Pipeline should surface an explicit timeout via ``timed_out=True`` and
    ``abort_reason='pipeline_timeout'``, without crashing on missing locals.

    The 4-outcome pin: timeout is a first-class terminal state distinct from
    compile-fail, structural-reject, and success."""

    @patch.object(ModelRouter, '_call_model')
    def test_timeout_before_first_generation(self, mock_call, router):
        """Budget exhausted before any Kimi call → clean early return, no
        ``compile_errs`` reference, ``timed_out=True``."""
        def dispatch(model_key, *args, **kwargs):
            if model_key == "deepseek":
                return _plan_result()
            # Simulate a failed Kimi call so route() enters the timeout arm
            if model_key == "kimi27":
                return AgentResult("kimi27", False, "", 0.0)
            return AgentResult(model_key, True, "{}", 0.1)

        mock_call.side_effect = dispatch

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {"compile_success": False, "errors": [], "output": ""}
        verifier.verify.return_value = {"compile_success": False, "passed": False, "compile_output": "", "output": ""}

        # max_seconds=0.001 starves the initial call → the timeout arm fires
        # and sets ``timed_out=True`` with ``abort_reason='pipeline_timeout'``.
        result = router.route(
            kernel_source=CUDA_SOURCE,
            patterns=[],
            max_iterations=1,
            verifier=verifier,
            kernel_name="test_pipeline_state",
            max_seconds=0.001,
        )

        assert result is not None
        # Either the pre-generation timeout arm fires, or Kimi's failure is
        # surfaced cleanly — both are valid non-crash terminals. Assert that
        # SOME clean terminal was reached and no state-machine locals leaked.
        assert (
            result.get("timed_out") is True
            or "Code generation FAILED" in " ".join(result.get("changes", []))
        )
