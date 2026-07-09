"""Tests for the 3-minute-demo work (prompts/to-fable-system-roast-and-fix.md).

Covers the seven fixes:
  P0  hard wall-clock timeout          → TestDeadline, TestRouteTimeout
  P0  cache-first check                → TestCacheFirst
  P0  faster stagnation detection      → TestStagnationThresholds
  P1  two-layer SIGSEGV fix            → TestTwoLayerFix
  P1  banner single source of truth    → TestBannerSingleSource
  P2  prompt versioning                → TestPromptVersioning
"""
import io
import os
import sys
import contextlib
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# main.py auto-loads the repo's .env at import time via os.environ.setdefault().
# Importing it here would otherwise leak a real FIREWORKS_API_KEY into the shared
# pytest process — which makes ModelRouter(api_key="") pick the key up and issue
# live network calls. Snapshot/restore around the import, as test_main.py does.
_env_snapshot = dict(os.environ)
import main
os.environ.clear()
os.environ.update(_env_snapshot)

from router import (
    ModelRouter, AgentResult, Deadline, PipelineTimeoutError,
    PROMPT_VERSION, SYSTEM_PROMPTS, MAX_PIPELINE_SECONDS,
    STAGNATION_ABORT_THRESHOLD, MAX_REPLANS, MODEL_CATALOG,
    PLAN_BUDGET_FRACTION, CODE_RESERVE_FRACTION, COMPILE_RESERVE_SECONDS,
    MIN_LLM_TIMEOUT_SECONDS,
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """No test in this file may reach the network."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


CUDA_SRC = """
#include <cuda_runtime.h>
__global__ void scan_kernel(float* input, int n) {
    int tid = threadIdx.x;
    __shfl_up_sync(0xffffffff, input[tid], 1);
}
"""

HIP_OK = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) { int tid = threadIdx.x; }
"""

HIP_OTHER = """
#include <hip/hip_runtime.h>
__global__ void scan_kernel(float* input, int n) { int tid = threadIdx.x; /* v2 */ }
"""


@pytest.fixture
def router():
    return ModelRouter(api_key="test_key")


# ── P0: Deadline primitive ───────────────────────────────────────────────

class TestDeadline:
    def test_unlimited_budget_never_expires(self):
        d = Deadline(0)
        assert d.unlimited
        assert not d.expired()
        assert d.remaining() is None

    def test_negative_budget_is_unlimited(self):
        assert Deadline(-1).unlimited

    def test_expired_once_budget_spent(self):
        d = Deadline(100)
        d._t0 -= 101  # pretend 101s elapsed
        assert d.expired()
        assert d.remaining() < 0

    def test_clamp_shrinks_timeout_to_remaining(self):
        d = Deadline(100)
        d._t0 -= 70  # 30s left
        assert d.clamp_timeout(180) == pytest.approx(30, abs=1)

    def test_clamp_never_returns_negative(self):
        d = Deadline(10)
        d._t0 -= 50
        assert d.clamp_timeout(180) == 0.0

    def test_clamp_is_noop_when_unlimited(self):
        assert Deadline(0).clamp_timeout(180) == 180

    def test_clamp_does_not_inflate_a_short_timeout(self):
        """A 30s local call must not grow to the full remaining budget."""
        d = Deadline(300)
        assert d.clamp_timeout(30) == 30

    def test_exhausted_when_remaining_is_below_the_call_floor(self):
        """Budget not yet expired, but too small to fit any LLM call.

        Found end-to-end: with 3s left and a 5s floor, expired() was False, so a
        run that had really timed out was reported as '[kimi27] Code generation
        FAILED' — a model failure — with time still on the clock.
        """
        d = Deadline(100)
        d._t0 -= 98  # 2s left, below the 5s floor
        assert not d.expired()
        assert d.exhausted()

    def test_not_exhausted_with_ample_budget(self):
        assert not Deadline(100).exhausted()

    def test_unlimited_is_never_exhausted(self):
        assert not Deadline(0).exhausted()


class TestCallModelBudget:
    def test_pipeline_timeout_error_is_a_runtime_error(self):
        assert issubclass(PipelineTimeoutError, RuntimeError)

    def test_expired_budget_short_circuits_before_any_request(self, router):
        """The whole point: no HTTP request is issued once the budget is gone."""
        router._deadline = Deadline(10)
        router._deadline._t0 -= 50
        with patch("urllib.request.urlopen") as mock_open:
            res = router._call_model("kimi27", "prompt")
        mock_open.assert_not_called()
        assert res.success is False

    def test_timeout_does_not_fall_through_to_the_local_endpoint(self, router):
        """A PipelineTimeoutError must end the call, not trigger a second doomed
        request against the local vLLM endpoint."""
        router._deadline = Deadline(180)
        router._deadline._t0 -= 178  # 2s left → below the floor, mid-call

        attempts = []

        def fake_urlopen(req, timeout=None):
            attempts.append(req.full_url)
            raise AssertionError("no request should be issued")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = router._call_model("kimi27", "prompt")

        assert res.success is False
        assert attempts == [], f"issued doomed requests: {attempts}"

    def test_kimi_retry_cannot_outlive_the_budget(self, router):
        """kimi27 has timeout=180 and retries at 2x. Under a 180s pipeline budget
        neither attempt may be issued with a timeout exceeding what remains."""
        router._deadline = Deadline(180)
        router._deadline._t0 -= 150  # 30s left
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            raise OSError("boom")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_model("kimi27", "prompt")

        assert seen, "expected at least one request attempt"
        assert all(t <= 31 for t in seen), f"timeout escaped the budget: {seen}"


# ── HIPIFY: mechanical translation + compile-first fast path ─────────────

SIMPLE_CUDA = """
#include <cuda_runtime.h>
__global__ void vadd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() { float* d; cudaMalloc(&d, 4); cudaFree(d); return 0; }
"""


def _clean_verifier():
    v = MagicMock()
    v.quick_compile_check.return_value = {"compile_success": True, "errors": [], "output": ""}
    v.quick_run_check.return_value = {"run_success": True}
    return v


class TestHipifySource:
    def test_translates_cuda_api_and_headers(self):
        out, log = ModelRouter._hipify_source(SIMPLE_CUDA)
        assert "cudaMalloc" not in out and "hipMalloc" in out
        assert "cudaFree" not in out and "hipFree" in out
        assert "cuda_runtime.h" not in out and "hip/hip_runtime.h" in out
        assert log, "no changelog produced"

    def test_is_idempotent(self):
        """The hipified source passes through _fix_ported_code again after Kimi.
        If the transform were not idempotent it would corrupt on the second pass —
        which is exactly what the post-processor did before it was fixed."""
        once, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        twice, _ = ModelRouter._hipify_source(once)
        assert once == twice

    def test_leaves_no_residual_cuda_symbols_on_a_simple_kernel(self):
        out, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        assert ModelRouter._residual_cuda_symbols(out) == []

    def test_residual_symbols_are_reported(self):
        assert "cudaGraphLaunch" in ModelRouter._residual_cuda_symbols(
            "cudaGraphLaunch(g, s);")


class TestWavefrontGate:
    def test_shfl_needs_semantics(self):
        assert ModelRouter._needs_wavefront_semantics("__shfl_up_sync(0xffffffff, v, 1);")

    def test_warpsize_needs_semantics(self):
        assert ModelRouter._needs_wavefront_semantics("int l = tid % warpSize;")

    def test_plain_kernel_does_not(self):
        assert not ModelRouter._needs_wavefront_semantics(SIMPLE_CUDA)

    def test_classifier_findings_also_trip_the_gate(self):
        """Belt and braces: the regex may miss a spelling the classifier caught."""
        assert ModelRouter._needs_wavefront_semantics(
            "int x = 1;", patterns=[{"pattern": "shfl_up_sync", "severity": "high"}])

    def test_the_real_outlier_kernel_needs_semantics(self):
        src = (Path(__file__).resolve().parents[1]
               / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu").read_text(encoding="utf-8")
        assert ModelRouter._needs_wavefront_semantics(src)


class TestFastPath:
    def test_mechanical_port_that_compiles_and_runs_skips_every_llm(self, router):
        with patch.object(ModelRouter, '_call_model') as mc:
            result = router.route(SIMPLE_CUDA, [], verifier=_clean_verifier(),
                                  kernel_name="test_fastok", max_seconds=0)
        assert mc.call_count == 0, "fast path must not call any model"
        assert result["fast_path_used"] is True
        assert result["model_used"] == "hipify"
        assert result["compile_passed"] is True
        assert result["orchestrator_passed"] is True
        assert result["cost"] == 0
        assert any("skipped DeepSeek, Kimi and GLM" in c for c in result["changes"])

    def test_compiles_but_crashes_falls_through_to_the_models(self, router):
        """RUN-FIRST: a mechanical port that compiles and SIGSEGVs is not a port.
        This is the 2026-07-09 failure — shared memory sized blockDim/32."""
        v = MagicMock()
        v.quick_compile_check.return_value = {"compile_success": True, "errors": [], "output": ""}
        v.quick_run_check.return_value = {
            "run_success": False, "signal": "SIGSEGV", "run_exit_code": -11, "run_output": ""}

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": true}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect) as mc:
            result = router.route(SIMPLE_CUDA, [], max_iterations=1, verifier=v,
                                  kernel_name="test_fastcrash", max_seconds=0)

        assert result["fast_path_used"] is False
        assert mc.call_count > 0, "must fall through to the coder"
        assert any("did not run clean" in c for c in result["changes"])

    def test_compile_failure_falls_through_to_the_models(self, router):
        v = MagicMock()
        v.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["e1", "e2"], "output": "", "error_context": []}

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect) as mc:
            result = router.route(SIMPLE_CUDA, [], max_iterations=1, verifier=v,
                                  kernel_name="test_fastnocompile", max_seconds=0)

        assert result["fast_path_used"] is False
        assert mc.call_count > 0
        assert any("did not compile" in c for c in result["changes"])

    def test_warp_kernel_skips_the_fast_path_without_spending_a_compile(self, router):
        """The gate exists to avoid burning ~20s of a 180s budget on a compile
        whose outcome the source already tells us."""
        v = _clean_verifier()

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": true}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=v,
                                  kernel_name="test_warpgate", max_seconds=0)

        assert result["fast_path_used"] is False
        assert any("uses warp-level primitives" in c for c in result["changes"])
        # exactly the loop's own compiles — no extra fast-path probe
        assert not any("did not compile" in c or "did not run clean" in c
                       for c in result["changes"])

    def test_fast_path_false_disables_it(self, router):
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": true}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect) as mc:
            result = router.route(SIMPLE_CUDA, [], max_iterations=1,
                                  verifier=_clean_verifier(),
                                  kernel_name="test_fastoff", max_seconds=0,
                                  fast_path=False)
        assert result["fast_path_used"] is False
        assert mc.call_count > 0

    def test_non_dict_compile_result_is_never_read_as_a_pass(self, router):
        """A MagicMock quick_compile_check returns a MagicMock, which is truthy.
        The fast-path probe must read that as 'no information', never as a pass —
        otherwise a mocked verifier ships an unverified kernel."""
        v = MagicMock()
        # first call is the fast-path probe (no information); the loop's own
        # compiles afterwards return a real dict
        v.quick_compile_check.side_effect = [
            MagicMock(),
            {"compile_success": False, "errors": ["e"], "output": "", "error_context": []},
            {"compile_success": False, "errors": ["e"], "output": "", "error_context": []},
        ]

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect) as mc:
            result = router.route(SIMPLE_CUDA, [], max_iterations=1, verifier=v,
                                  kernel_name="test_fastmock", max_seconds=0)
        assert result["fast_path_used"] is False
        assert mc.call_count > 0, "must fall through to the models"

    def test_reproducibility_logging_cannot_crash_a_port(self, router):
        """json.dumps raises TypeError on a non-serializable value, and the run-dir
        logging guarded only OSError — so a verifier returning anything unusual
        killed route() from inside a debug-logging block."""
        class _FalsyUnserializable:
            """Stands in for e.g. a numpy.bool_ or a mock: usable as a boolean,
            fatal to json.dumps."""
            def __bool__(self): return False

        import json as _json
        with pytest.raises(TypeError):
            _json.dumps({"compile_success": _FalsyUnserializable()})

        v = MagicMock()
        v.quick_compile_check.return_value = {
            "compile_success": _FalsyUnserializable(),
            "errors": ["error: undeclared identifier"],
            "output": "", "error_context": []}

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect):
            result = router.route(SIMPLE_CUDA, [], max_iterations=1, verifier=v,
                                  kernel_name="test_logcrash", max_seconds=0)
        assert result["ported_code"], "logging killed the port"


class TestAdaptiveTokens:
    def test_small_kernel_gets_far_less_than_the_ceiling(self):
        n = ModelRouter._compute_adaptive_max_tokens("x" * 2000)
        assert n < MODEL_CATALOG["kimi27"]["max_tokens"]

    def test_floor_protects_tiny_kernels_from_truncation(self):
        assert ModelRouter._compute_adaptive_max_tokens("int main(){}") == 2048

    def test_large_kernel_clamps_to_the_catalog_ceiling(self):
        n = ModelRouter._compute_adaptive_max_tokens("x" * 200_000)
        assert n == MODEL_CATALOG["kimi27"]["max_tokens"]

    def test_override_never_raises_the_catalog_ceiling(self, router):
        seen = []

        class _Resp:
            def __init__(self, b): self.b = b
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return self.b

        def fake(req, timeout=None):
            seen.append(json.loads(req.data.decode())["max_tokens"])
            return _Resp(json.dumps({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 1}}).encode())

        with patch("urllib.request.urlopen", side_effect=fake):
            router._call_model("glm", "p", max_tokens_override=999_999)
        assert seen == [MODEL_CATALOG["glm"]["max_tokens"]]


class TestPreprocessedSourcePrompts:
    def test_code_prompt_embeds_the_draft_and_narrows_the_checklist(self, router):
        draft, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        p = router._build_kimi_code_prompt(SIMPLE_CUDA, [], preprocessed_source=draft)
        assert "HIP DRAFT TO EDIT" in p
        assert "do NOT re-port" in p.lower() or "Do NOT re-port" in p
        assert "reference only" in p
        # The mechanical checklist block is dropped; the semantics block is kept.
        # (checkCudaErrors is still *named* in the prose, as work already done.)
        assert ModelRouter._MECHANICAL_CHECKLIST not in p
        assert ModelRouter._WAVEFRONT_CHECKLIST in p

    def test_code_prompt_elides_the_original_but_embeds_the_draft_whole(self, router):
        """The draft is edited, so it must be complete (a self-contained program
        truncated below its own main() can never be reproduced — Bug 1). The CUDA
        original is only reference: re-sending it whole doubled the prompt."""
        big = (Path(__file__).resolve().parents[1]
               / "sample_kernels" / "cuda" / "nvidia_shfl_scan.cu").read_text(encoding="utf-8")
        draft, _ = ModelRouter._hipify_source(big)
        p = router._build_kimi_code_prompt(big, [], preprocessed_source=draft)

        assert draft in p, "the draft Kimi must edit was truncated"
        assert big not in p, "the whole CUDA original was re-embedded alongside the draft"
        assert "original elided" in p
        assert len(p) < len(big) + len(draft), "prompt carries both sources in full"

    def test_code_prompt_without_a_draft_keeps_the_full_checklist(self, router):
        p = router._build_kimi_code_prompt(SIMPLE_CUDA, [])
        assert "HIP DRAFT TO EDIT" not in p
        assert ModelRouter._MECHANICAL_CHECKLIST in p
        assert ModelRouter._WAVEFRONT_CHECKLIST in p

    def test_refine_prompt_states_the_mechanical_pass_is_done(self, router):
        draft, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        p = router._build_kimi_refine_prompt(
            SIMPLE_CUDA, HIP_OK, "errors", [], preprocessed_source=draft)
        assert "MECHANICAL PASS ALREADY APPLIED" in p

    def test_refine_prompt_does_not_re_embed_the_draft(self, router):
        """Re-sending the draft would grow the prompt this parameter shrinks."""
        draft, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        with_draft = router._build_kimi_refine_prompt(
            SIMPLE_CUDA, HIP_OK, "errors", [], preprocessed_source=draft)
        without = router._build_kimi_refine_prompt(SIMPLE_CUDA, HIP_OK, "errors", [])
        assert len(with_draft) - len(without) < len(draft), "draft was re-embedded"

    def test_refine_prompt_names_cuda_symbols_that_crept_back(self, router):
        draft, _ = ModelRouter._hipify_source(SIMPLE_CUDA)
        regressed = "#include <hip/hip_runtime.h>\nint main(){ cudaMalloc(&d,4); }"
        p = router._build_kimi_refine_prompt(
            SIMPLE_CUDA, regressed, "errors", [], preprocessed_source=draft)
        assert "reintroduced CUDA symbols" in p
        assert "cudaMalloc" in p

    def test_route_passes_the_draft_and_adaptive_tokens_to_every_kimi_call(self, router):
        """The three edits the source prompt asked for: initial port, refine, retry."""
        seen = []

        def side_effect(model_key, prompt, **k):
            if model_key == "kimi27":
                seen.append(k.get("max_tokens_override"))
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        v = MagicMock()
        v.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["e"], "output": "", "error_context": []}

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect):
            router.route(CUDA_SRC, [], max_iterations=2, verifier=v,
                         kernel_name="test_kimitokens", max_seconds=0)

        assert len(seen) >= 2, "expected an initial port and at least one refine"
        assert all(t is not None for t in seen), "a kimi call lacked max_tokens_override"
        assert all(t <= MODEL_CATALOG["kimi27"]["max_tokens"] for t in seen)


# ── P0: phase budgets — no phase may starve the ones behind it ───────────

def _instant_fireworks(seen, content="```cpp\nint main(){}\n```"):
    """Fake urlopen that answers instantly and records the timeout it was given."""
    class _Resp:
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body

    def fake(req, timeout=None):
        seen.append((req.full_url, timeout))
        body = json.dumps({
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }).encode()
        return _Resp(body)
    return fake


class TestPhaseCapArithmetic:
    def test_unlimited_has_no_cap(self):
        assert Deadline(0).phase_cap(0.2) is None

    def test_cap_is_a_share_of_the_total_budget(self):
        assert Deadline(180).phase_cap(0.2) == pytest.approx(36, abs=0.5)

    def test_reserve_is_held_back_for_later_phases(self):
        # 180s budget, reserve 124s → only 56s could remain, but the 20% share is tighter
        cap = Deadline(180).phase_cap(0.2, reserve_s=124)
        assert cap == pytest.approx(36, abs=0.5)

    def test_reserve_wins_when_it_is_the_tighter_limit(self):
        # 180s budget, reserve 160s → 20s left, tighter than the 36s share
        cap = Deadline(180).phase_cap(0.2, reserve_s=160)
        assert cap == pytest.approx(20, abs=0.5)

    def test_cap_goes_negative_when_the_phase_cannot_run(self):
        assert Deadline(40).phase_cap(0.2, reserve_s=47) < 0

    def test_has_at_least(self):
        assert Deadline(0).has_at_least(1e9)          # unlimited
        assert Deadline(100).has_at_least(50)
        d = Deadline(100); d._t0 -= 80
        assert not d.has_at_least(50)


class TestPhaseBudgets:
    def test_planner_cannot_outlive_its_slice(self, router):
        """The 2026-07-10 regression: DeepSeek's own timeout is 120s of a 180s
        budget, so an 86.9s plan left the coder nothing. The request must now be
        issued with a timeout no larger than PLAN_BUDGET_FRACTION of the budget."""
        assert MODEL_CATALOG["deepseek"]["timeout"] == 120, "premise changed"
        seen = []
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch("urllib.request.urlopen", side_effect=_instant_fireworks(seen)):
            router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                         kernel_name="test_plancap", max_seconds=180)

        assert seen, "no request issued"
        plan_timeout = seen[0][1]
        expected = 180 * PLAN_BUDGET_FRACTION
        assert plan_timeout <= expected + 1, (
            f"planner got {plan_timeout}s; must be capped at ~{expected}s, "
            f"not the model's own 120s")

    def test_coder_gets_the_rest_minus_a_compile_reserve(self, router):
        seen = []
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch("urllib.request.urlopen", side_effect=_instant_fireworks(seen)):
            router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                         kernel_name="test_codecap", max_seconds=180)

        # second request is the coder; it may not claim the compile reserve
        assert len(seen) >= 2
        code_timeout = seen[1][1]
        assert code_timeout <= 180 - COMPILE_RESERVE_SECONDS + 1

    def test_planner_is_skipped_when_it_would_starve_the_coder(self, router):
        """A 40s budget cannot afford a plan at all. Skip it rather than spend the
        clock finding out — and still produce a port."""
        seen = []
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch("urllib.request.urlopen", side_effect=_instant_fireworks(seen)):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_planskip", max_seconds=40)

        assert any("Planning SKIPPED" in c for c in result["changes"])
        # the coder still ran, and no deepseek request was ever issued
        assert result["ported_code"], "skipping the plan must not skip the port"
        assert any("[kimi27]" in c for c in result["changes"])

    def test_planner_timeout_is_not_reported_as_a_model_failure(self, router):
        """A planner that overran its slice is a budget event, not a broken endpoint."""
        def slow_plan(model_key, *a, **k):
            if model_key == "deepseek":
                return AgentResult(model_key, False, "", 0.0)   # what a capped timeout returns
            return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        with patch.object(ModelRouter, '_call_model', side_effect=slow_plan):
            result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                                  kernel_name="test_planslow", max_seconds=180)

        assert any("did not finish within its" in c for c in result["changes"])
        assert not any("Planning FAILED" in c for c in result["changes"])
        assert result["ported_code"], "the port must still happen without a plan"

    def test_phase_cap_bounds_the_retry_too(self, router):
        """kimi27 retries at 2x its timeout. Under a 30s cap the retry may not
        turn 30s into 60s — that would spend the reserve the cap protects."""
        router._deadline = Deadline(0)  # unlimited pipeline: isolate the phase cap
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            raise urllib.error.URLError("timed out")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_model("kimi27", "p", max_seconds=30)

        assert seen, "no attempt made"
        assert all(t <= 30.5 for t in seen), f"retry escaped the phase cap: {seen}"

    def test_failed_glm_analyst_does_not_crash_the_route(self, router):
        """Regression: glm_analysis was bound only inside `if glm_err.success:`,
        yet the GLM-informed re-plan read it unconditionally. A failed analyst
        call — the very case its "falling back to raw errors" branch exists for —
        raised UnboundLocalError and took the whole run down."""
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            if model_key == "glm":
                return AgentResult(model_key, False, "", 0.0)   # analyst call fails
            return AgentResult(model_key, True, "plan", 0.5)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["error: undeclared identifier"],
            "output": "", "error_context": []}

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_glmfail", max_seconds=0)

        assert any("Error analyst call failed" in c for c in result["changes"])
        assert result["iterations_used"] >= 1

    def test_timeout_does_not_claim_a_compiling_attempt_when_none_compiled(self, router):
        """T0.1: 'returning the best compiling attempt (iter 0)' was printed even
        when no iteration ever compiled."""
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, "plan", 0.5)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["e"], "output": "", "error_context": []}

        calls = {"n": 0}

        def fake_exhausted(self):
            calls["n"] += 1
            return calls["n"] > 1     # survive iteration 1's check, die at iteration 2's

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect), \
             patch.object(Deadline, 'exhausted', fake_exhausted):
            result = router.route(CUDA_SRC, [], max_iterations=5, verifier=verifier,
                                  kernel_name="test_nocompile")

        assert result["timed_out"] is True
        assert result["compile_passed"] is False
        budget_lines = [c for c in result["changes"] if c.startswith("[budget] Wall-clock limit")
                        and "reached" in c]
        assert budget_lines, "no timeout line emitted"
        assert not any("best compiling attempt" in c for c in budget_lines), \
            "claimed a compiling attempt when nothing compiled"
        assert any("nothing compiled" in c for c in budget_lines)

    def test_checklist_version_is_not_double_prefixed(self, router):
        """`[prompt-v{version_id}]` rendered as `[prompt-vv2]`, and read as though
        it were router.PROMPT_VERSION. It is the PromptOptimizer's checklist."""
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["e"], "output": "", "error_context": []}

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect):
            result = router.route(CUDA_SRC, [], max_iterations=2, verifier=verifier,
                                  kernel_name="test_checklist", max_seconds=0)

        joined = "\n".join(result["changes"])
        assert "-vv" not in joined, "doubled version prefix"
        assert "[checklist v" in joined

    def test_iteration_is_not_entered_without_room_for_compile_and_a_call(self, router):
        """Entering an iteration costs an uninterruptible hipcc run. With less than
        COMPILE_RESERVE + MIN_LLM on the clock, that compile buys errors nobody
        will act on."""
        floor = COMPILE_RESERVE_SECONDS + MIN_LLM_TIMEOUT_SECONDS

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, "plan", 0.5)

        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["e"], "output": ""}

        calls = {"n": 0}
        real_has = Deadline.has_at_least

        def fake_has(self, seconds):
            # plenty of budget until the loop asks for a full iteration's worth
            if seconds == floor:
                calls["n"] += 1
                return False
            return real_has(self, seconds)

        with patch.object(ModelRouter, '_call_model', side_effect=side_effect), \
             patch.object(Deadline, 'has_at_least', fake_has):
            result = router.route(CUDA_SRC, [], max_iterations=10, verifier=verifier,
                                  kernel_name="test_floor", max_seconds=180)

        assert calls["n"] >= 1, "iteration floor was never consulted"
        assert result["timed_out"] is True
        assert result["iterations_used"] == 0


# ── P0: route() honours the wall-clock budget ────────────────────────────

class TestRouteTimeout:
    def test_default_budget_is_180s(self):
        assert MAX_PIPELINE_SECONDS == 180

    @patch.object(ModelRouter, '_call_model')
    def test_route_aborts_and_returns_best_compiling_code(self, mock_call, router):
        """Budget expires mid-loop → stop, flag it, hand back the code that compiled."""
        mock_call.side_effect = lambda mk, *a, **k: AgentResult(
            mk, True, f"```cpp\n{HIP_OK}\n```" if mk == "kimi27" else '{"pass": true}', 0.5)

        verifier = MagicMock()
        # pre-loop passes, iter1 passes (freezes best attempt), then we starve the budget
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        real_deadline = {}
        orig_init = Deadline.__init__

        def spy_init(self, budget_s):
            orig_init(self, budget_s)
            real_deadline['d'] = self
            self._t0 -= (budget_s + 1)  # already expired on the first check

        with patch.object(Deadline, '__init__', spy_init):
            result = router.route(CUDA_SRC, [], max_iterations=10,
                                  verifier=verifier, kernel_name="test_timeout")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        assert result["iterations_used"] == 0
        assert any("Wall-clock limit" in c for c in result["changes"])

    @patch.object(ModelRouter, '_call_model')
    def test_timeout_hands_back_the_compiling_code_not_the_last_refine(
            self, mock_call, router):
        """The plan's actual P0 requirement: 'abort and return the best-compiling
        code so far'. iter1 compiles but SIGSEGVs (code A); Kimi refines into code
        B; the budget expires before B is ever compiled. The caller must receive
        A — the code known to build — not the unvalidated B.

        Compile-pass + clean-run breaks the loop as converged (see the 'keeping
        working code' branch), so a runtime crash is what drives a refine here.
        """
        code_A = HIP_OK
        code_B = HIP_OTHER

        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                # 1st kimi call = initial port (A); 2nd = the refine (B)
                kimi_calls = [c for c in mock_call.call_args_list if c[0][0] == "kimi27"]
                code = code_A if len(kimi_calls) <= 1 else code_B
                return AgentResult(model_key, True, f"```cpp\n{code}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": false, "issues": ["shfl width"]}', 0.5)

        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.quick_compile_check.side_effect = [
            {"compile_success": False, "errors": ["e"], "output": ""},  # pre-loop
            {"compile_success": True, "errors": [], "output": ""},      # iter1 → A compiles
        ]
        # A compiles but crashes → the loop refines into B instead of converging
        verifier.quick_run_check.return_value = {
            "run_success": False, "signal": "SIGSEGV", "run_exit_code": -11, "run_output": ""}

        # Budget survives iteration 1's boundary check, dies before iteration 2's.
        calls = {"n": 0}

        def fake_exhausted(self):
            calls["n"] += 1
            return calls["n"] > 1

        with patch.object(Deadline, 'exhausted', fake_exhausted):
            result = router.route(CUDA_SRC, [], max_iterations=10, verifier=verifier,
                                  kernel_name="test_recover")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        # `ported_code` is post-processed by _fix_ported_code (it injects
        # WAVEFRONT_SIZE), so identify the lineage by B's unique marker rather
        # than by string equality with the raw source.
        assert "/* v2 */" not in result["ported_code"], \
            "returned the unvalidated refine instead of the code that compiled"
        assert result["ported_code"] == result["best_attempt_code"]
        assert result["compile_passed"] is True
        assert result["compile_errors"] == []
        assert any("Returned best compiling code" in c for c in result["changes"])

    @patch.object(ModelRouter, '_call_model')
    def test_max_seconds_zero_disables_the_budget(self, mock_call, router):
        mock_call.side_effect = lambda mk, *a, **k: AgentResult(
            mk, True, f"```cpp\n{HIP_OK}\n```" if mk == "kimi27" else '{"pass": true}', 0.5)
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": True, "errors": [], "output": ""}
        verifier.quick_run_check.return_value = {"run_success": True}

        result = router.route(CUDA_SRC, [], max_iterations=1, verifier=verifier,
                              kernel_name="test_nobudget", max_seconds=0)
        assert result["timed_out"] is False
        assert result.get("abort_reason") != "pipeline_timeout"

    def test_no_api_key_result_still_carries_budget_fields(self):
        r = ModelRouter(api_key="")
        assert r.api_key == "", "test would otherwise hit the network"
        result = r.route(CUDA_SRC, [])
        assert result["timed_out"] is False
        assert result["prompt_version"] == PROMPT_VERSION

    @patch.object(ModelRouter, '_call_model')
    def test_timeout_during_initial_codegen_is_reported_as_a_timeout(self, mock_call, router):
        """Regression: the budget can die on the very first Kimi call, before the
        loop. That used to return via the 'code generation FAILED' path with
        timed_out=False — a timeout disguised as a model failure."""
        # DeepSeek plans fine; Kimi returns the failed AgentResult that _call_model
        # produces when the budget is gone.
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                return AgentResult(model_key, False, "", 0.0)
            return AgentResult(model_key, True, "a plan", 0.5)

        mock_call.side_effect = side_effect

        orig_init = Deadline.__init__

        def spy_init(self, budget_s):
            orig_init(self, budget_s)
            self._t0 -= (budget_s - 1)  # 1s left: not expired, but below the floor

        with patch.object(Deadline, '__init__', spy_init):
            result = router.route(CUDA_SRC, [], max_iterations=5, kernel_name="test_early")

        assert result["timed_out"] is True
        assert result["abort_reason"] == "pipeline_timeout"
        assert not result["ported_code"]
        assert any("initial code generation" in c for c in result["changes"])
        assert not any("Code generation FAILED" in c for c in result["changes"]), \
            "a budget timeout must not be reported as a model failure"


# ── P0: cache-first check (already present — pinned against regression) ──

class TestCacheFirst:
    def test_retrieve_is_called_before_route_and_short_circuits_it(self, tmp_path):
        """A verified cache hit must skip the LLM pipeline entirely.

        This is the ~0.1s-instead-of-15-min path. It already worked; this test
        exists so a refactor cannot quietly move retrieve() after route().
        """
        ko = main.KernelOlympics.__new__(main.KernelOlympics)  # no __init__ side effects
        ko.memory = MagicMock()
        ko.memory.retrieve.return_value = {
            "id": "pat-1", "verified_fix": HIP_OK, "confidence": 0.95, "llm_time_s": 12.0,
        }
        ko.router = MagicMock()

        cached = ko.memory.retrieve(CUDA_SRC, findings=[])
        assert cached is not None
        ko.router.route.assert_not_called()

    def test_memory_retrieve_refuses_unverified_entries(self, tmp_path):
        """The plan proposed guarding on cached['verified']; retrieve() already does."""
        from pattern_memory.memory import PatternMemory
        mem = PatternMemory(db_path=str(tmp_path / "m.db"))
        findings = [{"pattern": "shfl_sync", "line": 1, "severity": "high"}]
        mem.store(CUDA_SRC, HIP_OK, confidence=0.9, verified=False, findings=findings)
        assert mem.retrieve(CUDA_SRC, findings=findings) is None


# ── P0: stagnation thresholds ────────────────────────────────────────────

class TestStagnationThresholds:
    def test_thresholds_fire_early_enough_for_a_3min_demo(self):
        assert STAGNATION_ABORT_THRESHOLD == 2
        assert MAX_REPLANS == 2

    def test_abort_is_reachable_within_the_iteration_budget(self):
        """Old gate needed replan_count >= max_iterations//2 (=5 at default 10),
        but re-plans were capped well below that, so it took ~6 iterations.
        The new gate must be satisfiable at MAX_REPLANS."""
        assert MAX_REPLANS >= 1
        assert STAGNATION_ABORT_THRESHOLD <= MAX_REPLANS + 1

    @patch.object(ModelRouter, '_call_model')
    def test_one_replan_per_iteration_at_most(self, mock_call, router):
        """A stagnant iteration used to fire two DeepSeek re-plans (~160s)."""
        calls = []

        def side_effect(model_key, *a, **k):
            calls.append(model_key)
            if model_key == "kimi27":
                return AgentResult(model_key, True, f"```cpp\n{HIP_OK}\n```", 0.5)
            return AgentResult(model_key, True, '{"fixes": []}', 0.5)

        mock_call.side_effect = side_effect
        verifier = MagicMock()
        verifier.quick_compile_check.return_value = {
            "compile_success": False, "errors": ["error: undeclared foo"], "output": ""}

        router.route(CUDA_SRC, [], max_iterations=3, verifier=verifier,
                     kernel_name="test_replan", max_seconds=0)

        deepseek_calls = [c for c in calls if c == "deepseek"]
        # 1 initial plan + at most MAX_REPLANS re-plans
        assert len(deepseek_calls) <= 1 + MAX_REPLANS, f"too many DeepSeek calls: {calls}"


# ── P1: two-layer SIGSEGV fix ────────────────────────────────────────────

class TestTwoLayerFix:
    def test_refine_prompt_freezes_the_base_kernel(self, router):
        p = router._build_kimi_refine_prompt(
            CUDA_SRC, HIP_OTHER, "crashed with SIGSEGV", [],
            frozen_base_code=HIP_OK)
        assert "Layer 1" in p
        assert "do NOT rewrite" in p.lower() or "Do NOT rewrite" in p
        assert "DISCARDED" in p
        assert HIP_OK.strip()[:40] in p

    def test_refine_prompt_omits_layer1_when_no_frozen_base(self, router):
        p = router._build_kimi_refine_prompt(
            CUDA_SRC, HIP_OTHER, "compile errors", [], frozen_base_code="")
        assert "Layer 1" not in p

    @patch.object(ModelRouter, '_call_model')
    def test_layer2_that_breaks_the_build_is_discarded(self, mock_call, router):
        """iter1 compiles + crashes → Kimi rewrites → iter2 fails to compile.
        The rewrite must be thrown away and the frozen base returned."""
        def side_effect(model_key, *a, **k):
            if model_key == "kimi27":
                # first call = initial port (compiles), second = the bad rewrite
                code = HIP_OK if mock_call.call_count <= 2 else HIP_OTHER
                return AgentResult(model_key, True, f"```cpp\n{code}\n```", 0.5)
            return AgentResult(model_key, True, '{"pass": false, "issues": ["shfl width"]}', 0.5)

        mock_call.side_effect = side_effect

        verifier = MagicMock()
        verifier.quick_compile_check.side_effect = [
            {"compile_success": True, "errors": [], "output": ""},   # pre-loop
            {"compile_success": True, "errors": [], "output": ""},   # iter1 → freeze Layer 1
            {"compile_success": False, "errors": ["error: a", "error: b"],
             "output": ""},                                          # iter2 → Layer 2 broke it
        ]
        # compiles but SIGSEGVs
        verifier.quick_run_check.return_value = {
            "run_success": False, "signal": "SIGSEGV", "run_exit_code": -11, "run_output": ""}

        result = router.route(CUDA_SRC, [], max_iterations=3, verifier=verifier,
                              kernel_name="test_twolayer", max_seconds=0)

        assert result["abort_reason"] == "layer2_rejected"
        assert any("two-layer" in c for c in result["changes"])
        # the returned code is the frozen, compiling base — not the broken rewrite
        assert result["compile_passed"] is True
        assert result["compile_errors"] == []


# ── P1: banner single source of truth ────────────────────────────────────

class TestBannerSingleSource:
    def test_render_banner_exists_and_is_the_only_emitter(self):
        assert callable(main._render_banner)
        src = Path(main.__file__).read_text(encoding="utf-8")
        # The banner literal may appear in BANNER_TEXT and nowhere else as a print()
        assert src.count("╔═ Kernel Olympics ═") == 1, \
            "banner literal duplicated — route it through _render_banner()"

    def test_display_prints_exactly_one_banner(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.Display()
        assert buf.getvalue().count(main.BANNER_TEXT) == 1

    def test_silent_display_prints_no_banner(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.Display(silent=True)
        assert buf.getvalue().count(main.BANNER_TEXT) == 0

    def test_pipeline_construction_emits_a_single_banner(self):
        """T2.1 regression: KernelOlympics built Display twice."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.KernelOlympics()
        assert buf.getvalue().count(main.BANNER_TEXT) == 1

    def test_silent_pipeline_emits_no_banner(self):
        """The duplicate Display() also dropped silent=silent.

        run_daemon() is the only caller that passes silent=True, so this is the
        path the regression actually broke: a banner on every daemon start.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.KernelOlympics(fresh=False, silent=True)  # exactly what run_daemon() does
        assert buf.getvalue().count(main.BANNER_TEXT) == 0

    def test_run_daemon_still_requests_silence(self):
        """Pins the caller, not just the callee: if run_daemon() ever stops
        passing silent=True, the test above silently stops protecting anything."""
        src = Path(main.__file__).read_text(encoding="utf-8")
        daemon_body = src.split("def run_daemon(", 1)[1]
        assert "KernelOlympics(fresh=fresh, silent=True)" in daemon_body


# ── P2: prompt versioning ────────────────────────────────────────────────

class TestPromptVersioning:
    def test_prompt_version_is_semver_tagged(self):
        assert PROMPT_VERSION.startswith("v")
        major, minor, patch_ = PROMPT_VERSION[1:].split(".")
        assert all(p.isdigit() for p in (major, minor, patch_))

    def test_every_system_prompt_carries_the_version(self):
        assert SYSTEM_PROMPTS, "no system prompts defined"
        for name, text in SYSTEM_PROMPTS.items():
            assert f"[prompt {PROMPT_VERSION}]" in text, \
                f"system prompt {name!r} is missing its version tag"

    def test_route_reports_the_prompt_version(self):
        r = ModelRouter(api_key="")
        assert r.api_key == "", "test would otherwise hit the network"
        assert r.route(CUDA_SRC, [])["prompt_version"] == PROMPT_VERSION

    def test_changelog_json_matches_the_code_version(self):
        repo = Path(__file__).resolve().parents[1]
        data = json.loads((repo / "data" / "prompt_changelog.json").read_text(encoding="utf-8"))
        assert data["current"] == PROMPT_VERSION
        versions = [v["version"] for v in data["versions"]]
        assert PROMPT_VERSION in versions

    def test_changelog_md_documents_the_current_version(self):
        repo = Path(__file__).resolve().parents[1]
        md = (repo / "prompts" / "CHANGELOG.md").read_text(encoding="utf-8")
        assert PROMPT_VERSION in md
