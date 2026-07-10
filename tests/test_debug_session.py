"""Tests for Phase 11 Debug Mode.

The contract under test, restated:

  * disabled by default, and a no-op when disabled
  * append-only — no artifact is ever overwritten, no response ever lost
  * deterministic — the same inputs produce the same tree and the same reports
  * a failed run yields a package sufficient to diagnose it offline

The last property is the one worth testing hardest: it is the reason the module
exists, and it is the one that silently rots.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from debug_session import (
    DebugSession, debug_enabled, discarded_text, _slug, _jsonable, _dur,
    STAGE_TRANSLATION, STAGE_PATCHES, STAGE_COMPILER, STAGE_FAILURE,
)
from verification.extraction import extract_code
from verification.lexical import validate_lexical
from verification.structural import validate_structure


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

HIP = (CUDA.replace("cuda_runtime.h", "hip/hip_runtime.h")
           .replace("0xffffffff", "0xffffffffffffffffULL")
           .replace("#define WARP_SIZE 32", "#define WAVEFRONT_SIZE 64"))


@pytest.fixture
def session(tmp_path):
    s = DebugSession("test_kernel", root=tmp_path)
    yield s


# ── Enablement ──────────────────────────────────────────────────────────────

class TestEnablement:
    def test_off_by_default(self, monkeypatch):
        for var in ("KERNEL_OLYMPICS_DEBUG", "KERNEL_DEBUG_MODE"):
            monkeypatch.delenv(var, raising=False)
        assert debug_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "y"])
    def test_env_truthy_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG", val)
        assert debug_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_env_falsy_values_do_not_enable(self, monkeypatch, val):
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG", val)
        assert debug_enabled() is False

    def test_explicit_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("KERNEL_OLYMPICS_DEBUG", "1")
        assert debug_enabled(False) is False
        monkeypatch.delenv("KERNEL_OLYMPICS_DEBUG")
        assert debug_enabled(True) is True

    def test_create_returns_null_session_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KERNEL_OLYMPICS_DEBUG", raising=False)
        s = DebugSession.create("k", enabled=False, root=tmp_path)
        assert s.enabled is False
        assert bool(s) is False
        assert list(Path(tmp_path).iterdir()) == []

    def test_create_returns_live_session_when_enabled(self, tmp_path):
        s = DebugSession.create("k", enabled=True, root=tmp_path)
        assert s.enabled is True
        assert s.dir.is_dir()


class TestNullSession:
    """Disabled mode must be a total no-op that never raises and never writes."""

    def test_every_logging_method_is_a_safe_noop(self, tmp_path):
        s = DebugSession.disabled()
        s.log_input(CUDA, classifier_results={}, patterns=[])
        s.log_planning(raw_response="x")
        assert s.log_generation("raw") == 0
        s.log_extraction(None)
        s.log_lexical(None)
        s.log_structural(None)
        s.log_symbols(CUDA, HIP)
        s.log_static_analysis(HIP)
        s.log_compile(["hipcc"], stdout="", stderr="")
        s.log_evaluation(raw_response="x")
        s.log_patch(before="a", after="b")
        s.transition("X", reason="y")
        s.event("z")
        s.count("c")
        s.record_llm_call("m", tokens=1)
        s.write_text("s", "n", "t")
        s.write_json("s", "n", {})
        assert s.snapshot_failure(ValueError("x")) is None
        assert s.finalize({}) is None
        with s.stage("phase"):
            pass
        assert s.dir is None

    def test_unknown_future_method_does_not_crash(self):
        """A method added to DebugSession but not stubbed here must not explode."""
        s = DebugSession.disabled()
        assert s.log_some_method_added_next_year(1, 2, x=3) is None

    def test_stage_contextmanager_propagates_exceptions(self):
        s = DebugSession.disabled()
        with pytest.raises(RuntimeError):
            with s.stage("phase"):
                raise RuntimeError("boom")


# ── Append-only ─────────────────────────────────────────────────────────────

class TestAppendOnly:
    def test_same_artifact_name_never_overwrites(self, session):
        p1 = session.write_text(STAGE_TRANSLATION, "gen.txt", "first")
        p2 = session.write_text(STAGE_TRANSLATION, "gen.txt", "second")
        assert p1 != p2
        assert p1.read_text() == "first"
        assert p2.read_text() == "second"

    def test_sequence_numbers_are_monotonic(self, session):
        paths = [session.write_text(STAGE_TRANSLATION, "a.txt", str(i)) for i in range(5)]
        seqs = [int(p.name.split("_")[0]) for p in paths]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 5

    def test_every_generation_stored_separately(self, session):
        g1 = session.log_generation("raw one", extracted_code="code1", iteration=1)
        g2 = session.log_generation("raw two", extracted_code="code2", iteration=2)
        assert (g1, g2) == (1, 2)
        raws = sorted((session.dir / STAGE_TRANSLATION).glob("*_raw_response.txt"))
        assert len(raws) == 2
        bodies = {p.read_text() for p in raws}
        assert bodies == {"raw one", "raw two"}

    def test_every_patch_stored_separately(self, session):
        session.log_patch(before="a\n", after="b\n", iteration=1)
        session.log_patch(before="b\n", after="c\n", iteration=2)
        diffs = sorted((session.dir / STAGE_PATCHES).glob("*.diff"))
        assert len(diffs) == 2
        assert "patch001" in diffs[0].name and "patch002" in diffs[1].name

    def test_manifest_records_every_artifact(self, session):
        session.write_text(STAGE_TRANSLATION, "a.txt", "x")
        session.write_json(STAGE_TRANSLATION, "b", {"k": 1})
        lines = [json.loads(l) for l in
                 (session.dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
        artifacts = [l for l in lines if l.get("event") == "artifact"]
        assert len(artifacts) == 2
        assert {a["kind"] for a in artifacts} == {"text", "json"}

    def test_traces_are_jsonl_append_only(self, session):
        session.transition("A", reason="one")
        session.transition("B", reason="two")
        lines = (session.dir / "state_trace.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["next_state"] == "A"
        assert json.loads(lines[1])["previous_state"] == "A"

    def test_finalize_twice_does_not_overwrite_summary(self, session):
        first = session.finalize({"compile_passed": True})
        second = session.finalize({"compile_passed": False})
        assert first.name == "summary.md"
        assert second != first
        assert first.exists() and second.exists()

    def test_session_dir_collision_gets_new_directory(self, tmp_path):
        a = DebugSession("k", root=tmp_path, session_id="fixed")
        b = DebugSession("k", root=tmp_path, session_id="fixed")
        assert a.dir != b.dir
        assert a.dir.exists() and b.dir.exists()


# ── Determinism ─────────────────────────────────────────────────────────────

class TestDeterminism:
    """The analyzers' own determinism is pinned in test_symbols.py and
    test_static_analysis.py. What is tested here is the session: that the same
    pipeline produces the same *tree*, so two sessions can be diffed."""

    def test_reports_written_twice_are_byte_identical(self, tmp_path):
        bodies = []
        for i in range(2):
            s = DebugSession("k", root=tmp_path, session_id=f"s{i}")
            s.log_symbols(CUDA, HIP, generation=1)
            s.log_static_analysis(HIP, generation=1)
            bodies.append([p.read_text(encoding="utf-8")
                           for p in sorted(s.dir.rglob("*.json"))])
        assert bodies[0] == bodies[1]

    def test_artifact_names_do_not_embed_wall_clock(self, session):
        p = session.write_text(STAGE_TRANSLATION, "x.txt", "y")
        # The sequence number, not a timestamp, is what orders the tree.
        assert p.name.startswith("0001_")

    def test_two_sessions_produce_the_same_tree_shape(self, tmp_path):
        shapes = []
        for i in range(2):
            s = DebugSession("k", root=tmp_path, session_id=f"s{i}")
            s.log_generation("raw", extracted_code=HIP, iteration=1)
            s.log_static_analysis(HIP, generation=1)
            s.finalize({})
            shapes.append(sorted(
                str(p.relative_to(s.dir)).replace("\\", "/")
                for p in s.dir.rglob("*") if p.is_file()))
        assert shapes[0] == shapes[1]

    def test_clock_is_injectable(self, tmp_path):
        s = DebugSession("k", root=tmp_path, clock=lambda: 0.0)
        assert s.session_id.startswith("session_19700101T000000_")


# ── Artifact content ────────────────────────────────────────────────────────

class TestArtifactContent:
    def test_raw_response_is_never_truncated(self, session):
        huge = "x" * 500_000
        session.log_generation(huge, extracted_code="", iteration=1)
        raw = next((session.dir / STAGE_TRANSLATION).glob("*_raw_response.txt"))
        assert len(raw.read_text()) == 500_000

    def test_compiler_output_is_never_truncated(self, session):
        stderr = "error: boom\n" * 50_000
        session.log_compile(["hipcc", "-o", "k"], stdout="", stderr=stderr, returncode=1)
        out = next((session.dir / STAGE_COMPILER).glob("*_stderr.txt"))
        assert out.read_text() == stderr

    def test_compile_report_captures_exact_command_and_version(self, session):
        cmd = ["hipcc", "-o", "k", "k.cpp", "--offload-arch=gfx942"]
        session.log_compile(cmd, returncode=0, compiler_version="HIP 6.2.0")
        rep = json.loads(next((session.dir / STAGE_COMPILER).glob("*_report.json")).read_text())
        assert rep["command"] == cmd
        assert rep["command_line"] == " ".join(cmd)
        assert rep["compiler_version"] == "HIP 6.2.0"
        assert rep["success"] is True

    def test_compile_report_excludes_secrets_from_environment(self, session, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "sk-super-secret")
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
        session.log_compile(["hipcc"], returncode=0)
        body = next((session.dir / STAGE_COMPILER).glob("*_report.json")).read_text()
        assert "sk-super-secret" not in body
        assert "/opt/rocm" in body

    def test_patch_diff_and_line_counts_come_from_the_texts(self, session):
        rep = session.log_patch(before="a\nb\nc\n", after="a\nZ\nc\n", iteration=1)
        assert rep["lines_added"] == 1
        assert rep["lines_removed"] == 1
        assert rep["lines_modified"] == 2
        diff = next((session.dir / STAGE_PATCHES).glob("*.diff")).read_text()
        assert "-b" in diff and "+Z" in diff

    def test_patch_records_a_noop_refine_as_unchanged(self, session):
        rep = session.log_patch(before=HIP, after=HIP, iteration=2)
        assert rep["unchanged"] is True
        assert rep["lines_modified"] == 0

    def test_extraction_report_is_machine_readable(self, session):
        raw = "Here you go:\n```cpp\n" + HIP + "```\nHope that helps!"
        result = extract_code(raw)
        session.log_extraction(result, generation=1, iteration=1)
        rep = json.loads(next((session.dir / "04_extraction").glob("*.json")).read_text())
        assert rep["code_block_detected"] is True
        assert rep["extraction_failed"] is False
        assert 0.0 < rep["parser_confidence"] <= 1.0
        assert "confidence_formula" in rep

    def test_extraction_report_detects_markdown_and_discarded_reasoning(self, session):
        raw = "Here you go:\n```cpp\n" + HIP + "```\nHope that helps!"
        session.log_extraction(extract_code(raw), generation=1)
        rep = json.loads(next((session.dir / "04_extraction").glob("*.json")).read_text())
        assert rep["markdown_removed"] is True
        assert rep["reasoning_removed"] is True
        assert rep["discarded_length"] > 0
        assert rep["strategy_used"] == "markdown-fence"

    def test_extraction_report_marks_a_failure_as_a_failure(self, session):
        result = extract_code("Let's think. I believe the answer is unclear.")
        session.log_extraction(result, generation=1)
        rep = json.loads(next((session.dir / "04_extraction").glob("*.json")).read_text())
        assert rep["extraction_failed"] is True
        assert rep["parser_confidence"] == 0.0

    def test_lexical_report_names_the_rejection(self, session):
        prose = "Let's think about this.\nI think we should port it.\nActually, wait."
        lex = validate_lexical(prose)
        assert not lex.ok
        session.log_lexical(lex, generation=1, code=prose)
        rep = json.loads(next((session.dir / "05_lexical").glob("*.json")).read_text())
        assert rep["pass"] is False
        assert rep["decision"] == "REJECT"
        assert rep["detected_reasoning"] is True
        assert rep["rejected_phrases"]

    def test_structural_report_records_score_and_symbol_counts(self, session):
        st = validate_structure(CUDA, HIP)
        session.log_structural(st, generation=1, cuda_source=CUDA, hip_source=HIP)
        rep = json.loads(next((session.dir / "06_structural").glob("*.json")).read_text())
        assert rep["pass"] is True
        assert rep["brace_validation"] is True
        assert rep["kernel_preservation"] is True
        assert rep["symbol_counts"]["original"]["kernels"] == 1

    def test_structural_report_validates_namespaces(self, session):
        src = "namespace cg { }\nusing namespace std;\n__global__ void k(){}\n"
        prt = "__global__ void k(){}\n"
        st = validate_structure(src, prt)
        session.log_structural(st, generation=1, cuda_source=src, hip_source=prt)
        rep = json.loads(next((session.dir / "06_structural").glob("*.json")).read_text())
        ns = rep["namespace_validation"]
        assert ns["original"]["declared"] == ["cg"]
        assert ns["dropped_declarations"] == ["cg"]
        assert ns["dropped_using"] == ["std"]
        assert ns["preserved"] is False

    def test_namespace_report_says_preserved_when_kept(self, session):
        code = "namespace cg { }\n__global__ void k(){}\n"
        st = validate_structure(code, code)
        session.log_structural(st, generation=1, cuda_source=code, hip_source=code)
        rep = json.loads(next((session.dir / "06_structural").glob("*.json")).read_text())
        assert rep["namespace_validation"]["preserved"] is True
        assert rep["namespace_validation"]["dropped_declarations"] == []

    def test_structural_report_flags_unbalanced_braces(self, session):
        broken = "__global__ void k(float* a) { if (a) {"
        st = validate_structure(CUDA, broken)
        session.log_structural(st, generation=1, cuda_source=CUDA, hip_source=broken)
        rep = json.loads(next((session.dir / "06_structural").glob("*.json")).read_text())
        assert rep["pass"] is False
        assert rep["brace_validation"] is False


# ── Failure snapshot ────────────────────────────────────────────────────────

class TestFailureSnapshot:
    def test_snapshot_contains_everything_needed_to_diagnose_offline(self, session):
        session.log_input(CUDA, patterns=[{"p": "shfl"}])
        session.transition("INPUT_RECEIVED", reason="loaded")
        gen = session.log_generation("raw model text", extracted_code=HIP, iteration=1)
        session.transition("CODE_GENERATED", reason="coder ok")
        session.log_compile(["hipcc", "k.cpp"], stderr="k.cpp:1:1: error: boom", returncode=1)
        session.event("compile_failed", reason="1 error", iteration=1)
        session.count("compile_failures")

        session.snapshot_failure(RuntimeError("kaboom"), reason="test failure")

        snap = json.loads(next((session.dir / STAGE_FAILURE).glob("*failure_snapshot.json")).read_text())
        assert snap["reason"] == "test failure"
        assert snap["exception_type"] == "RuntimeError"
        assert "kaboom" in snap["traceback"]
        assert snap["state_trace"], "state trace must be embedded"
        assert snap["retry_history"], "retry history must be embedded"
        assert snap["counters"]["compile_failures"] == 1
        # The index must point at the artifacts that hold the raw evidence.
        paths = [a["path"].replace("\\", "/") for a in snap["artifact_index"]]
        assert any(p.startswith("01_input/") for p in paths)
        assert any(p.startswith("03_translation/") for p in paths)
        assert any(p.startswith("09_compiler/") for p in paths)

    def test_snapshot_without_an_exception_still_packages(self, session):
        session.snapshot_failure(reason="did not converge")
        snap = json.loads(next((session.dir / STAGE_FAILURE).glob("*failure_snapshot.json")).read_text())
        assert snap["exception_type"] is None
        assert snap["reason"] == "did not converge"

    def test_raw_response_survives_a_failure(self, session):
        """The whole point: the text that caused the failure is still on disk."""
        session.log_generation("Let's think. I believe...", extracted_code="", iteration=1)
        session.snapshot_failure(reason="lexical reject")
        raws = list((session.dir / STAGE_TRANSLATION).glob("*_raw_response.txt"))
        assert raws and raws[0].read_text() == "Let's think. I believe..."


# ── Summary ─────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_is_written_and_names_the_outcome(self, session):
        session.transition("INPUT_RECEIVED", reason="loaded")
        session.transition("HIPCC_COMPILE", reason="3 errors", validation_result=False)
        path = session.finalize({"abort_reason": "max_iterations_exhausted",
                                 "compile_errors": ["k.cpp:1:1: error: boom"],
                                 "compile_passed": False})
        body = path.read_text(encoding="utf-8")
        assert "# Debug session — test_kernel" in body
        assert "max_iterations_exhausted" in body
        assert "Probable root cause" in body
        assert "Recommended next action" in body
        assert "k.cpp:1:1: error: boom" in body

    def test_summary_reports_the_timeline_and_transitions(self, session):
        session.transition("A", reason="first")
        session.transition("B", reason="second")
        session.event("generation", reason="coder", iteration=1)
        body = session.finalize({}).read_text(encoding="utf-8")
        assert "`A`" in body and "`B`" in body
        assert "generation" in body

    def test_root_cause_is_inferred_per_abort_reason(self, session):
        body = session.finalize({"abort_reason": "harness_origin"}).read_text(encoding="utf-8")
        assert "harness" in body.lower()
        assert "specs/test_kernel.json" in body

    def test_success_summary_does_not_claim_a_failure(self, session):
        session.transition("SUCCESS", reason="converged", validation_result=True)
        body = session.finalize({"compile_passed": True}).read_text(encoding="utf-8")
        assert "**PASSED**" in body
        assert "No failure detected" in body

    def test_summary_does_not_point_at_an_empty_compiler_dir(self, session):
        """A run where hipcc never ran must say so, not cite compiler output."""
        session.log_generation("prose", extracted_code="", iteration=1)
        session.count("lexical_rejects")
        body = session.finalize({
            "abort_reason": "max_iterations_exhausted",
            "compile_errors": ["[structural] [lexical] reasoning at top level"],
        }).read_text(encoding="utf-8")
        assert "hipcc never ran" in body
        assert "Full, untruncated" not in body
        # …and the structural rejections are labeled as gate rejections, not
        # as things the compiler said.
        assert "not by hipcc" in body

    def test_summary_separates_gate_rejections_from_compiler_errors(self, session):
        session.log_compile(["hipcc"], stderr="k.cpp:1:1: error: real", returncode=1)
        body = session.finalize({
            "compile_errors": ["k.cpp:1:1: error: real",
                               "[structural] unbalanced braces"],
        }).read_text(encoding="utf-8")
        assert "k.cpp:1:1: error: real" in body
        assert "1 gate rejection(s)" in body
        assert "1 diagnostic(s) on the final attempt" in body

    def test_root_cause_prefers_evidence_over_abort_reason(self, session):
        """`max_iterations_exhausted` says how the loop quit, not why it failed."""
        for i in (1, 2, 3):
            session.log_generation("Let's think.", extracted_code="", iteration=i)
            session.count("lexical_rejects")
        body = session.finalize(
            {"abort_reason": "max_iterations_exhausted"}).read_text(encoding="utf-8")
        assert "returned reasoning/prose instead of source code" in body
        assert "prompting or model-selection failure" in body
        assert "The iteration ceiling was reached" not in body

    def test_root_cause_falls_back_to_abort_reason_when_compiles_happened(self, session):
        session.log_generation("code", extracted_code="int main(){}", iteration=1)
        session.log_compile(["hipcc"], returncode=1)
        body = session.finalize(
            {"abort_reason": "max_iterations_exhausted"}).read_text(encoding="utf-8")
        assert "The iteration ceiling was reached" in body

    def test_metrics_json_is_written(self, session):
        session.record_llm_call("kimi27", tokens=900, cost=0.001, latency_ms=1200.0)
        with session.stage("hipcc"):
            pass
        session.finalize({})
        m = json.loads((session.dir / "metrics.json").read_text())
        assert m["totals"]["tokens"] == 900
        assert m["totals"]["llm_calls"] == 1
        assert any(s["stage"] == "hipcc" for s in m["per_stage"])
        assert m["total_runtime_ms"] >= 0


# ── Timing / metrics ────────────────────────────────────────────────────────

class TestMetrics:
    def test_stage_records_duration_even_when_body_raises(self, session):
        with pytest.raises(ValueError):
            with session.stage("hipcc"):
                raise ValueError("boom")
        m = session.metrics()
        stage = next(s for s in m["per_stage"] if s["stage"] == "hipcc")
        assert stage["calls"] == 1
        assert stage["failures"] == 1

    def test_token_usage_accumulates_per_model(self, session):
        session.record_llm_call("kimi27", tokens=100, cost=0.001, latency_ms=10)
        session.record_llm_call("kimi27", tokens=200, cost=0.002, latency_ms=20)
        session.record_llm_call("glm", tokens=50, cost=0.0005, latency_ms=5)
        usage = {u["model"]: u for u in session.metrics()["token_usage"]}
        assert usage["kimi27"]["tokens"] == 300
        assert usage["kimi27"]["calls"] == 2
        assert usage["glm"]["tokens"] == 50

    def test_counters_accumulate(self, session):
        session.count("retries")
        session.count("retries", 2)
        assert session.metrics()["counters"]["retries"] == 3

    def test_transition_records_time_spent_in_previous_state(self, session):
        session.transition("A", reason="x")
        session.transition("B", reason="y")
        recs = [json.loads(l) for l in
                (session.dir / "state_trace.jsonl").read_text().splitlines()]
        assert recs[1]["previous_state"] == "A"
        assert recs[1]["elapsed_in_previous_ms"] >= 0
        assert recs[1]["since_start_ms"] >= recs[1]["elapsed_in_previous_ms"]


# ── Robustness: the sink must never be the thing that fails ────────────────

class TestNeverFatal:
    def test_unserializable_payload_is_recorded_not_raised(self, session):
        class Weird:
            def __repr__(self): return "<weird>"
        p = session.write_json("01_input", "weird", {"obj": Weird()})
        assert p is not None
        assert "<weird>" in p.read_text()

    def test_write_failure_is_recorded_in_errors_jsonl(self, session, monkeypatch):
        def boom(*a, **kw):
            raise OSError("disk full")
        monkeypatch.setattr(Path, "write_text", boom)
        assert session.write_text("01_input", "x.txt", "y") is None
        # errors.jsonl is written through open(), not Path.write_text, so it survives.
        errors = (session.dir / "errors.jsonl").read_text()
        assert "disk full" in errors

    def test_log_symbols_on_garbage_does_not_raise(self, session):
        assert session.log_symbols("\x00\x01", "}{}{") is not None

    def test_log_static_analysis_on_garbage_does_not_raise(self, session):
        assert session.log_static_analysis("}{ ((( ///") is not None

    def test_log_lexical_accepts_none(self, session):
        session.log_lexical(None)  # must not raise

    def test_create_falls_back_to_disabled_on_unwritable_root(self, monkeypatch, tmp_path):
        def boom(*a, **kw):
            raise OSError("read-only filesystem")
        monkeypatch.setattr(Path, "mkdir", boom)
        s = DebugSession.create("k", enabled=True, root=tmp_path / "nope")
        assert s.enabled is False


# ── Helpers ─────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_discarded_text_returns_prose_around_the_code(self):
        raw = "Here you go:\nCODE\nHope it helps."
        d = discarded_text(raw, "CODE")
        assert "Here you go:" in d and "Hope it helps." in d
        assert "CODE" not in d.replace("<<< extracted code omitted >>>", "")

    def test_discarded_text_is_empty_when_response_is_only_code(self):
        assert discarded_text("CODE", "CODE") == ""

    def test_discarded_text_returns_everything_when_no_code(self):
        assert discarded_text("Let's think.", "") == "Let's think."

    def test_discarded_text_over_reports_rather_than_lying(self):
        """When code is not a substring of raw, we must not claim nothing was discarded."""
        assert discarded_text("totally different text", "MUTATED CODE") != ""

    def test_slug_is_filesystem_safe_and_deterministic(self):
        assert _slug("a/b\\c:d*e") == "a_b_c_d_e"
        assert _slug("") == "unnamed"
        assert _slug("x" * 200, limit=10) == "x" * 10

    def test_jsonable_handles_sets_paths_and_dataclasses(self):
        from dataclasses import dataclass

        @dataclass
        class Sample:
            rule: str
            line: int

        out = _jsonable({"s": {3, 1, 2}, "p": Path("a/b"), "f": Sample("r", 1)})
        assert out["s"] == [1, 2, 3]          # sets become sorted lists
        assert out["f"] == {"rule": "r", "line": 1}
        assert isinstance(out["p"], str)

    def test_jsonable_falls_back_to_repr_rather_than_raising(self):
        class Weird:
            def __repr__(self): return "<weird>"
        assert _jsonable(Weird()) == "<weird>"
        json.dumps(_jsonable({"x": Weird()}))

    def test_dur_picks_a_unit_a_reader_can_act_on(self):
        """A 0.4ms parser stage rendered as '0.00s' tells nobody anything."""
        assert _dur(0.4).endswith("µs")
        assert _dur(12.7) == "12.7ms"
        assert _dur(38200.0) == "38.20s"
