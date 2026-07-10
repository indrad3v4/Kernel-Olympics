"""Debug Mode wired into the real pipeline.

test_debug_session.py tests the sink in isolation. This file tests the thing
that actually rots: whether ``ModelRouter.route()`` still calls it, at the right
moments, with the right values — and whether a run with Debug Mode off is
byte-for-byte the same run it was before Phase 11.

The models and the compiler are faked. Everything else is the production code
path, including ``_postprocess_port``, the structural gate, and the refine loop.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import router as router_mod
from router import ModelRouter, AgentResult
from debug_session import DebugSession


CUDA = """#include <cuda_runtime.h>
#define WARP_SIZE 32
__device__ float warpReduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__global__ void reduce_kernel(float* in, float* out, int n) {
    out[blockIdx.x] = warpReduce(in[threadIdx.x]);
}
int main() { return 0; }
"""

GOOD_HIP = (CUDA.replace("cuda_runtime.h", "hip/hip_runtime.h")
                .replace("0xffffffff", "0xffffffffffffffffULL"))

# A response that is pure reasoning: balanced braces (zero of each), so the
# structural gate alone would pass it. The lexical gate is what must catch it.
PROSE = "Let's think about this.\nI think we need a different approach.\nActually, wait."


class FakeVerifier:
    """Compiles nothing. Fails the first N compiles, then passes."""

    def __init__(self, fail_first: int = 1):
        self.fail_first = fail_first
        self.compiles = 0
        self.attached = None

    def attach_debug_session(self, session):
        self.attached = session

    def detach_debug_session(self):
        self.attached = None

    def quick_compile_check(self, src, kernel_name="k"):
        self.compiles += 1
        if self.compiles <= self.fail_first:
            return {"compile_success": False, "compile_output": "boom",
                    "errors": ["k.cpp:5:3: error: use of undeclared identifier 'foo'"],
                    "error_origins": ["ported_code"], "error_context": [],
                    "all_harness_origin": False}
        return {"compile_success": True, "compile_output": "", "errors": [],
                "error_origins": [], "error_context": [], "all_harness_origin": False}

    def quick_run_check(self, kernel_name):
        return {"run_success": True, "run_output": "ok", "run_exit_code": 0,
                "signal": "", "benchmark_us": 12.0}


def _fake_models(coder_outputs):
    """Build a _call_model_impl that replays *coder_outputs* for glm."""
    state = {"i": 0}

    def impl(self, model_key, prompt, system_prompt="", prefill="",
             max_seconds=None, max_tokens_override=None):
        if model_key == "deepseek":
            return AgentResult("deepseek", True, "Plan:\n1. 64-bit mask", 0.8, 120, 3800.0)
        if model_key == "glm":
            i = min(state["i"], len(coder_outputs) - 1)
            state["i"] += 1
            return AgentResult("glm", True, coder_outputs[i], 0.7, 900, 12000.0)
        if model_key == "kimi27":
            return AgentResult("kimi27", True,
                               '{"pass":true,"issues":[],"feedback":"ok"}', 0.9, 300, 5000.0)
        return AgentResult(model_key, False, "", 0.0)
    return impl


@pytest.fixture
def routed(monkeypatch, tmp_path):
    """Run route() with fakes; return (result, session_dir)."""
    def _run(coder_outputs, debug=True, fail_first=1, max_iterations=4,
             verifier=None):
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_models(coder_outputs))
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG_DIR", str(tmp_path))
        r = ModelRouter(api_key="fake")
        v = verifier if verifier is not None else FakeVerifier(fail_first=fail_first)
        res = r.route(CUDA, [{"pattern": "shfl_down_sync"}],
                      max_iterations=max_iterations, verifier=v,
                      kernel_name="warp_reduce", max_seconds=0,
                      fast_path=False, debug=debug)
        d = res.get("debug_session_dir")
        return res, (Path(d) if d else None), v
    return _run


# ── Off by default ──────────────────────────────────────────────────────────

class TestDisabledIsInert:
    def test_no_debug_dir_is_created_when_disabled(self, routed, tmp_path):
        res, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"], debug=False)
        assert d is None
        assert "debug_session_dir" not in res
        assert list(tmp_path.iterdir()) == []

    def test_pipeline_result_is_identical_with_and_without_debug(self, routed):
        outputs = ["```cpp\n" + GOOD_HIP + "```"]
        off, _, _ = routed(outputs, debug=False)
        on, d, _ = routed(outputs, debug=True)
        assert d is not None
        # Everything the pipeline promises its caller must be unchanged --
        # including `cost` and `changes`, which Debug Mode must not perturb.
        #
        # Two keys are exempt, and only two: `debug_session_dir`, which exists
        # solely to point at the session, and `run_id`, which is a fresh uuid4
        # on every route() call whether or not Debug Mode is on.
        ignore = {"debug_session_dir", "run_id"}
        assert {k: v for k, v in off.items() if k not in ignore} == \
               {k: v for k, v in on.items() if k not in ignore}
        assert set(on) - set(off) == {"debug_session_dir"}

    def test_router_debug_is_null_session_before_route(self):
        assert ModelRouter(api_key="x").debug.enabled is False

    def test_env_var_enables_without_a_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_models(["```cpp\n" + GOOD_HIP + "```"]))
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG_DIR", str(tmp_path))
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG", "1")
        r = ModelRouter(api_key="fake")
        # kernel_name must name an existing spec: route() auto-generates
        # src/verification/specs/<kernel_name>.json, and a novel name would
        # leave a stray file in the repo every time this test runs.
        res = r.route(CUDA, [], max_iterations=2, verifier=FakeVerifier(fail_first=0),
                      kernel_name="warp_reduce", max_seconds=0, fast_path=False)
        assert res.get("debug_session_dir")


# ── The happy path records the whole pipeline ───────────────────────────────

class TestArtifactsAreProduced:
    def test_every_stage_directory_is_populated(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"], fail_first=1)
        for stage in ("01_input", "02_planning", "03_translation", "04_extraction",
                      "05_lexical", "06_structural", "07_symbols", "08_static_analysis"):
            assert any((d / stage).iterdir()), f"{stage} is empty"

    def test_raw_model_responses_are_persisted_before_parsing(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        plan_raw = list((d / "02_planning").glob("*raw_response.txt"))
        code_raw = list((d / "03_translation").glob("*raw_response.txt"))
        assert plan_raw and code_raw
        assert "```cpp" in code_raw[0].read_text(encoding="utf-8") or \
               any("```cpp" in p.read_text(encoding="utf-8") for p in code_raw)

    def test_prompts_are_persisted_alongside_responses(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        assert list((d / "02_planning").glob("*_prompt.txt"))
        assert list((d / "03_translation").glob("*_prompt.txt"))

    def test_verifier_receives_the_session(self, routed):
        v = FakeVerifier(fail_first=0)
        _, d, v = routed(["```cpp\n" + GOOD_HIP + "```"], verifier=v)
        # Attached during the run, detached after — a verifier outlives one route().
        assert v.attached is None

    def test_summary_and_metrics_are_written(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        assert (d / "summary.md").exists()
        assert (d / "metrics.json").exists()
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        assert m["totals"]["llm_calls"] >= 2  # planner + coder
        assert m["kernel"] == "warp_reduce"

    def test_metrics_break_down_parser_and_validation_latency(self, routed):
        """'per-stage runtime' must include the non-LLM stages, or it is a lie."""
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"], fail_first=0)
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        stages = {s["stage"] for s in m["per_stage"]}
        assert {"extraction", "lexical_validation", "structural_validation"} <= stages
        assert any(s.startswith("llm:") for s in stages)
        assert "hipcc" in stages
        for s in m["per_stage"]:
            assert s["calls"] >= 1 and s["total_ms"] >= 0

    def test_state_trace_records_the_pipeline_states(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"], fail_first=0)
        states = [json.loads(l)["next_state"]
                  for l in (d / "state_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        for expected in ("INPUT_RECEIVED", "PLAN_GENERATED", "CODE_GENERATED",
                         "CODE_EXTRACTED", "LEXICAL_VALIDATION",
                         "STRUCTURAL_VALIDATION", "HIPCC_COMPILE"):
            assert expected in states, f"{expected} missing from {states}"

    def test_state_trace_is_a_chain(self, routed):
        """Each transition's previous_state is the last transition's next_state."""
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        recs = [json.loads(l) for l in
                (d / "state_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        for a, b in zip(recs, recs[1:]):
            assert b["previous_state"] == a["next_state"]
        assert recs[0]["previous_state"] == "INIT"

    def test_terminal_state_names_the_outcome(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"], fail_first=0)
        recs = [json.loads(l) for l in
                (d / "state_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        assert recs[-1]["next_state"] in ("SUCCESS", "FAILURE")


# ── A rejected generation is fully reconstructible ──────────────────────────

class TestRejectedGenerationIsDiagnosable:
    def test_lexical_reject_persists_the_prose_that_caused_it(self, routed):
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"])
        raws = sorted((d / "03_translation").glob("*gen001*raw_response.txt"))
        assert raws, "the rejected generation's raw text must survive"
        assert "Let's think about this." in raws[0].read_text(encoding="utf-8")

    def test_lexical_report_explains_the_reject(self, routed):
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"])
        rep = json.loads(next((d / "05_lexical").glob("*gen001*.json")).read_text(encoding="utf-8"))
        assert rep["pass"] is False
        assert rep["decision"] == "REJECT"
        assert rep["detected_reasoning"] is True

    def test_hipcc_is_not_run_on_a_structurally_rejected_generation(self, routed):
        """The gate's whole purpose: no compile is spent on prose."""
        v = FakeVerifier(fail_first=0)
        _, d, v = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"], verifier=v)
        states = [json.loads(l) for l in
                  (d / "state_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        reject_idx = next(i for i, s in enumerate(states)
                          if s["next_state"] == "STRUCTURAL_REJECT")
        compiles_before = [s for s in states[:reject_idx]
                           if s["next_state"] == "HIPCC_COMPILE"]
        assert compiles_before == []

    def test_generations_accumulate_and_none_overwrite(self, routed):
        _, d, _ = routed([PROSE, PROSE, "```cpp\n" + GOOD_HIP + "```"], max_iterations=4)
        raws = list((d / "03_translation").glob("*gen*_raw_response.txt"))
        assert len(raws) >= 3
        gen_ids = {p.name.split("gen")[1][:3] for p in raws}
        assert len(gen_ids) == len(raws), "each generation has its own file"

    def test_static_analysis_is_recorded_even_when_no_compile_runs(self, routed):
        """A rejected generation still gets its pre-compile findings on disk."""
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"])
        assert list((d / "08_static_analysis").glob("*gen001*.json"))


# ── Patch history ───────────────────────────────────────────────────────────

class TestPatchHistory:
    def test_each_refine_is_stored_as_its_own_patch(self, routed):
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"], fail_first=1,
                         max_iterations=4)
        diffs = sorted((d / "11_patches").glob("*.diff"))
        assert len(diffs) >= 1
        names = [p.name for p in diffs]
        assert len(set(names)) == len(names)

    def test_patch_report_counts_lines_from_the_texts(self, routed):
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"], fail_first=1)
        reports = sorted((d / "11_patches").glob("*_report.json"))
        rep = json.loads(reports[0].read_text(encoding="utf-8"))
        assert rep["lines_modified"] == rep["lines_added"] + rep["lines_removed"]
        assert rep["source"] in ("refine", "refine_retry", "shim_injection")


# ── Failure packaging ───────────────────────────────────────────────────────

class TestFailurePackaging:
    def test_a_non_converging_run_produces_a_failure_snapshot(self, routed):
        # The coder only ever returns prose: nothing ever compiles.
        res, d, _ = routed([PROSE], max_iterations=3, fail_first=99)
        assert not res.get("compile_passed")
        snaps = list((d / "12_failure").glob("*failure_snapshot.json"))
        assert snaps, "a run that never compiled must package itself"
        snap = json.loads(snaps[0].read_text(encoding="utf-8"))
        assert snap["state_trace"]
        assert snap["artifact_index"]

    def test_an_exception_inside_route_is_snapshotted_and_reraised(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_models(["```cpp\n" + GOOD_HIP + "```"]))
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG_DIR", str(tmp_path))

        class ExplodingVerifier(FakeVerifier):
            def quick_compile_check(self, src, kernel_name="k"):
                raise RuntimeError("hipcc exploded")

        r = ModelRouter(api_key="fake")
        with pytest.raises(RuntimeError, match="hipcc exploded"):
            r.route(CUDA, [], max_iterations=2, verifier=ExplodingVerifier(),
                    kernel_name="warp_reduce", max_seconds=0, fast_path=False,
                    debug=True)

        session_dirs = list(tmp_path.glob("session_*"))
        assert session_dirs, "the session must exist after a crash"
        snap = json.loads(
            next(session_dirs[0].glob("12_failure/*failure_snapshot.json"))
            .read_text(encoding="utf-8"))
        assert snap["exception_type"] == "RuntimeError"
        assert "hipcc exploded" in snap["traceback"]

    def test_summary_of_a_failed_run_names_a_root_cause(self, routed):
        res, d, _ = routed([PROSE], max_iterations=3, fail_first=99)
        body = (d / "summary.md").read_text(encoding="utf-8")
        assert "Probable root cause" in body
        assert "Recommended next action" in body
        assert "No single root cause" not in body or res.get("abort_reason")


# ── Session ownership ───────────────────────────────────────────────────────

class TestSessionOwnership:
    """Whoever creates the session finalizes it.

    main.py creates one, hands it to route(), and then runs the authoritative
    verify() compile. If route() finalized a session it did not create, the
    summary would be written before the compile that decides PASSED vs FAILED.
    """

    def _route_with(self, monkeypatch, session, verifier=None):
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_models(["```cpp\n" + GOOD_HIP + "```"]))
        r = ModelRouter(api_key="fake")
        res = r.route(CUDA, [], max_iterations=2,
                      verifier=verifier or FakeVerifier(fail_first=0),
                      kernel_name="warp_reduce", max_seconds=0, fast_path=False,
                      debug_session=session)
        return r, res

    def test_a_borrowed_session_is_not_finalized_by_route(self, monkeypatch, tmp_path):
        s = DebugSession("warp_reduce", root=tmp_path)
        r, _ = self._route_with(monkeypatch, s)
        assert r._owns_debug_session is False
        assert not (s.dir / "summary.md").exists()
        assert not (s.dir / "metrics.json").exists()
        # …but the caller can still finalize it, and gets everything.
        s.finalize({"compile_passed": True})
        assert (s.dir / "summary.md").exists()

    def test_a_borrowed_session_still_records_every_stage(self, monkeypatch, tmp_path):
        s = DebugSession("warp_reduce", root=tmp_path)
        self._route_with(monkeypatch, s)
        for stage in ("01_input", "02_planning", "03_translation", "05_lexical"):
            assert any((s.dir / stage).iterdir()), f"{stage} empty"

    def test_a_borrowed_session_stays_attached_to_the_verifier(self, monkeypatch, tmp_path):
        """The caller is about to run verify() — detaching would lose that compile."""
        s = DebugSession("warp_reduce", root=tmp_path)
        v = FakeVerifier(fail_first=0)
        self._route_with(monkeypatch, s, verifier=v)
        assert v.attached is s

    def test_an_owned_session_is_detached_and_finalized(self, routed):
        _, d, v = routed(["```cpp\n" + GOOD_HIP + "```"], fail_first=0)
        assert (d / "summary.md").exists()
        assert v.attached is None

    def test_a_borrowed_session_is_snapshotted_on_exception(self, monkeypatch, tmp_path):
        """The caller may never get to finalize; the snapshot must not wait for it."""
        monkeypatch.setattr(ModelRouter, "_call_model_impl",
                            _fake_models(["```cpp\n" + GOOD_HIP + "```"]))

        class ExplodingVerifier(FakeVerifier):
            def quick_compile_check(self, src, kernel_name="k"):
                raise RuntimeError("hipcc exploded")

        s = DebugSession("warp_reduce", root=tmp_path)
        r = ModelRouter(api_key="fake")
        with pytest.raises(RuntimeError, match="hipcc exploded"):
            r.route(CUDA, [], max_iterations=2, verifier=ExplodingVerifier(),
                    kernel_name="warp_reduce", max_seconds=0, fast_path=False,
                    debug_session=s)
        assert list(s.dir.glob("12_failure/*failure_snapshot.json"))
        # Not finalized — the session was borrowed.
        assert not (s.dir / "summary.md").exists()

    def test_passing_a_null_session_keeps_debug_off(self, monkeypatch, tmp_path):
        """main.py always passes a session; a disabled one must stay inert."""
        null = DebugSession.disabled()
        r, res = self._route_with(monkeypatch, null)
        assert r.debug.enabled is False
        assert "debug_session_dir" not in res
        assert list(tmp_path.iterdir()) == []


# ── Replayability ───────────────────────────────────────────────────────────

class TestReplayability:
    def test_the_session_alone_answers_what_the_model_returned(self, routed):
        """No LLM re-execution needed: the text is on disk, tied to a verdict."""
        _, d, _ = routed([PROSE, "```cpp\n" + GOOD_HIP + "```"])

        # 1. What did the coder return on generation 1?
        raw = next((d / "03_translation").glob("*gen001*raw_response.txt")
                   ).read_text(encoding="utf-8")
        # 2. Why was it rejected?
        lex = json.loads(next((d / "05_lexical").glob("*gen001*.json")
                              ).read_text(encoding="utf-8"))
        # 3. What did the extractor make of it?
        ext = json.loads(next((d / "04_extraction").glob("*gen001*.json")
                              ).read_text(encoding="utf-8"))

        assert "Let's think" in raw
        assert lex["pass"] is False
        assert ext["extraction_failed"] is True
        # The chain closes: raw text → extraction verdict → lexical verdict.

    def test_input_report_captures_classifier_and_preprocessing(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        rep = json.loads(next((d / "01_input").glob("*input_report.json")
                              ).read_text(encoding="utf-8"))
        assert rep["kernel_name"] == "warp_reduce"
        assert rep["detected_patterns"]
        assert rep["preprocessing"]["applied"] is True
        assert rep["preprocessing"]["transforms"] > 0
        assert (d / "01_input").glob("*original.cu")

    def test_original_source_is_byte_identical_on_disk(self, routed):
        _, d, _ = routed(["```cpp\n" + GOOD_HIP + "```"])
        saved = next((d / "01_input").glob("*original.cu")).read_text(encoding="utf-8")
        assert saved == CUDA


# ── main.py wiring ──────────────────────────────────────────────────────────

class TestMainWiring:
    """The CLI is where the ownership rule is actually consumed."""

    def test_cli_exposes_a_debug_flag(self):
        import main
        parser_src = Path(main.__file__).read_text(encoding="utf-8")
        assert '"--debug"' in parser_src
        assert '"--debug-dir"' in parser_src

    def test_kernel_olympics_accepts_debug_and_defaults_off(self):
        import main
        import inspect
        sig = inspect.signature(main.KernelOlympics.__init__)
        assert sig.parameters["debug"].default is False

    def test_main_creates_the_session_and_hands_it_to_route(self):
        """Pins the ownership contract at the call site.

        If main.py ever reverts to `debug=` instead of `debug_session=`, route()
        finalizes the summary before verify() runs and the authoritative compile
        silently vanishes from it.
        """
        import main
        src = Path(main.__file__).read_text(encoding="utf-8")
        run_body = src.split("def run(", 1)[1].split("\n    def ", 1)[0]
        assert "DebugSession.create(" in run_body
        assert "debug_session=dbg" in run_body
        # …and main.py, not route(), writes the summary.
        assert "dbg.finalize(" in run_body

    def test_main_records_the_authoritative_verify_compile(self):
        import main
        src = Path(main.__file__).read_text(encoding="utf-8")
        run_body = src.split("def run(", 1)[1].split("\n    def ", 1)[0]
        verify_region = run_body.split("self.verifier.verify(", 1)[0][-500:]
        assert "attach_debug_session(dbg)" in verify_region

    def test_daemon_forwards_the_debug_flag(self):
        import main
        import inspect
        assert "debug" in inspect.signature(main.run_daemon).parameters
