"""
Kernel Olympics — Main orchestrator with live ANSI progress display.

Pipeline:
1. Scanner: runs hipify-clang dry-run on CUDA files
2. Risk Classifier: rule-based pattern matching for warp/wavefront divergence
3. Pattern Memory: trigram-indexed cache (0.2ms retrieval vs 12s LLM)
4. Porting Agent: skip on cache hit, Fireworks API on miss
5. Verification Agent: compile + run + diff on AMD Developer Cloud
6. Report Generator: Gemma on local ROCm for plain-English summary

Usage:
    python main.py --input sample_kernels/cuda/*.cu
"""

import argparse
import json
import logging
import os
import sys
import time
import shutil
from pathlib import Path

# Auto-load .env file if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.strip("\"'")
                os.environ.setdefault(_k.strip(), _v)

sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 stdout/stderr (with ASCII fallback) before any glyphs print.
from utf8_console import enable_utf8_console
enable_utf8_console()

from scanner.scanner import Scanner
from risk_classifier.classifier import RiskClassifier
from pattern_memory.memory import PatternMemory
from porting_agent.agent import PortingAgent
from router import ModelRouter
from verification.verifier import VerificationAgent
from report_generator.reporter import ReportGenerator


# ── Zero-dependency ANSI display helpers ──────────────────────────

def _c(s, code): return f"\033[{code}m{s}\033[0m"
def green(s): return _c(s, 92)
def yellow(s): return _c(s, 93)
def red(s): return _c(s, 91)
def cyan(s): return _c(s, 96)
def bold(s): return _c(s, 1)
def dim(s): return _c(s, 2)


def verification_failure_label(ver_result: dict) -> str:
    """Pick the failure label for a verify() result that did not pass.

    Bug 0: compile_success / run_success / output_match are three DIFFERENT
    failure points, and verify() (verifier.py) returns early on a run
    failure — before any diff ever executes. The old code derived the label
    from compile_success alone, so a crashed binary and a genuine diff
    failure printed identically as "Output mismatch", and the label implied
    a comparison had happened when it hadn't. Call only when
    ver_result["passed"] is falsy and it isn't the no-GPU case.
    """
    if not ver_result.get("compile_success"):
        return "Not compiled — saved for manual hipcc"
    if not ver_result.get("run_success"):
        exit_code = ver_result.get("run_exit_code")
        if exit_code is not None:
            return f"Compiled, but crashed at runtime (exit {exit_code})"
        return "Compiled, but failed to run"
    return "Output mismatch"

SPINNER = "|/-\\"

class Display:
    """Live-updating terminal display. Zero dependencies, pure ANSI."""

    def __init__(self, silent: bool = False):
        try:
            self.width = min(shutil.get_terminal_size().columns, 80)
        except (ValueError, OSError):
            self.width = 80
        self._is_tty = sys.stdout.isatty() and not silent
        self._phase_lines = {}
        self._counter = 0
        self._headers_printed = 0
        self._start_time = time.time()
        if not silent:
            print(bold("╔═ Kernel Olympics ═══════════════════════════════╗"))

    def phase(self, name: str, icon: str):
        self._counter += 1
        self._phase_lines[name] = self._counter
        line = f"║ {icon} {bold(name)}..."
        print(f"{line:<68}║")

    def status(self, phase: str, text: str, ok: bool = True):
        mark = green("●") if ok else yellow("●")
        ts = dim(f"{time.time()-self._start_time:.1f}s")
        print(f"║ {mark} {bold(phase):<16} {text:<40} {ts:<6}║")

    def file_done(self, name: str, info: str, ok: bool = True):
        mark = green("✓") if ok else yellow("→")
        print(f"║  {mark} {dim(name+':')} {info:<58}║")

    def cache_hit(self):
        self._cache_hits = getattr(self, '_cache_hits', 0) + 1
        hits = self._cache_hits
        calls = getattr(self, '_llm_calls', 0)
        print(f"║ {green('⏩')} Cache: {bold(str(hits))} hits  LLM: {bold(str(calls))} calls          ║")

    def llm_call(self):
        self._llm_calls = getattr(self, '_llm_calls', 0) + 1

    def divider(self):
        print(f"╠{'═'*66}╣")

    def final_summary(self, mem_stats: dict, pipeline_state: dict):
        elapsed = time.time() - self._start_time
        hits = pipeline_state.get("cache_hits", 0)
        calls = pipeline_state.get("llm_calls", 0)
        total = hits + calls
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        cache_ms = mem_stats.get("last_cache_time_ms", 0)
        llm_s = mem_stats.get("last_llm_time_s", 0)
        speedup = f"~{llm_s / (max(cache_ms, 0.1)/1000):.0f}×" if cache_ms > 0 and llm_s > 0 else "N/A"
        
        total_cost = pipeline_state.get("total_cost", 0.0)
        cost_str = f"${total_cost:.4f}" if total_cost > 0 else f"$0.0000 ({calls} LLM call{'s' if calls != 1 else ''})"
        
        print(f"║{'═'*66}║")
        print(f"║ {bold('Summary')}")
        print(f"║ {green('●')} Cache: {bold(str(hits))} hits  LLM: {bold(str(calls))} calls  {cyan(f'{hit_rate:.0f}%')} hit rate")
        print(f"║ {green('●')} Fastest: {cache_ms}ms  LLM avg: {llm_s}s  {cyan(speedup)} faster with cache")
        print(f"║ {green('●')} Patterns: {pipeline_state.get('patterns_before',0)} → {bold(str(pipeline_state.get('patterns_after',0)))} stored")
        print(f"║ {green('●')} Cost: {bold(cost_str)}")
        print(f"║ {green('●')} Elapsed: {elapsed:.1f}s total")
        print(f"╚{'═'*66}╝")

    def _flush(self):
        import sys as _sys
        _sys.stdout.flush()


class KernelOlympics:
    """Orchestrates the full CUDA→ROCm migration pipeline."""

    def __init__(self, fresh: bool = False, silent: bool = False):
        self.fresh = fresh
        self.silent = silent
        self.disp = Display(silent=silent)
        self.scanner = Scanner()
        self.classifier = RiskClassifier()
        self.memory = PatternMemory()
        if fresh:
            # Start with an empty pattern cache (T3.2: --fresh)
            self.memory.clear()
        self.porting_agent = PortingAgent(
            deepseek_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        )
        self.router = ModelRouter()
        self.verifier = VerificationAgent()
        self.reporter = ReportGenerator()
        self.disp = Display()

    def run(self, input_paths: list[str], reference_dir: str = "sample_kernels/reference") -> dict:
        pipeline_state = {"phase": "initializing", "patterns_before": 0, "patterns_after": 0,
                          "cache_hits": 0, "llm_calls": 0, "total_cost": 0.0}

        # Phase 1: Scanner
        self.disp.phase("Scanning", "🔍")
        scan_results = self.scanner.scan_batch(input_paths)
        for r in scan_results:
            cov = r.get('hipify_coverage_pct', 0)
            ok = cov > 80 #50 is a little low, I wouldn't trust an AI written kernel
            self.disp.file_done(Path(r['file']).name, f"coverage: {cov}%", ok=ok)
        self.disp.status("Scanning", f"{len(scan_results)} files scanned", ok=True)

        # Phase 2: Risk Classifier
        self.disp.phase("Classifying", "⚠️")
        file_sources = {}
        for fp in input_paths:
            try:
                file_sources[fp] = Path(fp).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logging.debug("Failed to read %s: %s", fp, e)

        if not file_sources:
            print("\033[91m✖ ERROR: no readable input files found — nothing scanned.\033[0m")
            print("\033[91m  Check the path(s) passed to --input.\033[0m")
            return {"error": "no_input", "files_scanned": 0}

        classifier_results = self.classifier.classify_batch(file_sources)
        red_kernels = [r for r in classifier_results if r.get("risk_level") == "red"]
        ylw = [r for r in classifier_results if r.get("risk_level") == "yellow"]
        for cr in classifier_results:
            findings = cr.get("findings", [])
            if findings:
                levels = ", ".join(f"[{f['severity']}] L{f['line']}: {f['pattern']}" for f in findings[:3])
                self.disp.file_done(Path(cr['file']).name, levels, ok=cr.get("risk_level") != "red")
        self.disp.status("Classifying", f"RED: {len(red_kernels)}  YELLOW: {len(ylw)}  GREEN: {len(classifier_results)-len(red_kernels)-len(ylw)}")
#since there is a potential correlation between coverage and the ris k atribution, maybe we can try to make the risk assesment dynamic later on if we get some good runs. 
        
        # Phase 3: Pattern Memory
        self.disp.phase("Memory Cache", "🧠")
        pipeline_state["patterns_before"] = self.memory.count()
        count = self.memory.count()
        self.disp.status("Memory Cache", f"{count} cached patterns ready" if count > 0 else "0 cached patterns (first run)", ok=count > 0)

        # Phase 4: Porting (or skip if cached)
        self.disp.phase("Porting", "🤖")
        verification_results = []
        total_llm_time = 0.0

        for cr in classifier_results:
            if cr.get("risk_level") == "red":
                source = file_sources.get(cr["file"], "")
                if not source:
                    continue

                # Pass classifier findings so store/retrieve key on the SAME
                # full-source pattern signature (source is truncated on store).
                cached = self.memory.retrieve(source, findings=cr.get("findings", []))
                if cached:
                    pipeline_state["cache_hits"] += 1
                    self.disp.cache_hit()
                    llm_saved = cached.get("llm_time_s", 12.0)
                    self.disp.file_done(
                        Path(cr['file']).name,
                        f"{green('CACHED')} {cached.get('confidence',0)*100:.0f}%  {dim(f'saved ~{llm_saved:.0f}s')}",
                        ok=True
                    )
                    port_result = {
                        "ported_code": cached["verified_fix"],
                        "confidence": cached["confidence"] * 100,
                        "changes": [f"Applied cached fix from pattern {cached['id']} "
                                    f"(LLM call skipped — saved ~{llm_saved:.0f}s)"],
                        "from_cache": True,
                        "llm_time_s": 0
                    }
                else:
                    pipeline_state["llm_calls"] += 1
                    self.disp.llm_call()
                    # Show which verifier is actually being used
                    import urllib.request
                    gemma_online = False
                    try:
                        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=1)
                        gemma_online = True
                    except Exception:
                        pass
                    verifier_name = "Gemma 4(AMD)" if gemma_online else "DeepSeek V4 Pro(Gemma fallback)"
                    self.disp.status("Porting", "DeepSeek-v4-pro (plan) → Kimi K2.7 (code) → GLM-5.2 (evaluate) ⟲ loop")
                    t0 = time.perf_counter()

                    # Live progress: phase callback only (no spinner thread)
                    import threading
                    _phase_state = {"phase": None, "model": "", "detail": "", "phase_t0": t0}

                    def _on_phase(phase, model, detail):
                        # Close out previous phase with its duration
                        if _phase_state["phase"] is not None:
                            prev_dur = time.perf_counter() - _phase_state["phase_t0"]
                            prev_icon = icons.get(_phase_state["phase"], "●")
                            prev_model = _phase_state["model"]
                            prev_detail = _phase_state["detail"]
                            print(f"║  {prev_icon} {bold(prev_model):<16} {prev_detail:<38} {cyan(f'{prev_dur:.1f}s')}")
                        # Record new phase start (no print yet — duration unknown)
                        _phase_state["phase_t0"] = time.perf_counter()
                        _phase_state["phase"] = phase
                        _phase_state["model"] = model
                        _phase_state["detail"] = detail

                    icons = {"plan": "🧠", "code": "⚡", "evaluate": "🔬",
                             "refine": "🔁", "verify": "✅", "compile": "🔨"}

                    port_result = self.router.route(
                        source, cr.get("findings", []),
                        on_phase=_on_phase,
                        verifier=self.verifier,
                        kernel_name=Path(cr['file']).stem
                    )

                    # Close out the LAST phase with its duration
                    if _phase_state["phase"] is not None:
                        last_dur = time.perf_counter() - _phase_state["phase_t0"]
                        last_icon = icons.get(_phase_state["phase"], "●")
                        last_model = _phase_state["model"]
                        last_detail = _phase_state["detail"]
                        print(f"║  {last_icon} {bold(last_model):<16} {last_detail:<38} {cyan(f'{last_dur:.1f}s')}")

                    llm_elapsed = time.perf_counter() - t0
                    pipeline_state["total_cost"] += port_result.get("cost", 0)
                    if not port_result.get("ported_code"):
                        port_result = self.porting_agent.port_kernel(source)
                        pipeline_state["total_cost"] += port_result.get("cost", 0)
                        llm_elapsed = time.perf_counter() - t0
                    # Show orchestrator loop details
                    iters = port_result.get("iterations_used", 1)
                    orch_passed = port_result.get("orchestrator_passed", False)
                    orch_changes = [c for c in port_result.get("changes", []) if "[deepseek]" in c or "[glm]" in c or "[kimi27]" in c or "[hipcc]" in c or "orchestrator" in str(c).lower()]
                    if orch_changes:
                        # Show the LAST 5 changes, not the first 5 — on a multi-iteration
                        # run the final iteration's state is what matters, not iteration 1's.
                        for ch in orch_changes[-5:]:
                            print(f"║  🧠 {dim(ch[:70]):<64}║")
                    compile_ok = port_result.get("compile_passed", False)
                    tag = "✅ PASSED" if orch_passed else (f"✅ COMPILED" if compile_ok else f"🔁 {iters}/10 iterations")
                    self.disp.file_done(Path(cr['file']).name, f"GLM-eval {tag} ({port_result.get('confidence', 0)}%, {llm_elapsed:.0f}s)", ok=orch_passed)
                    save_path = Path.cwd() / "ported_kernels" / (Path(cr["file"]).stem + ".hip.cpp")
                    print(f"║  📁 Ported kernel → {bold(str(save_path)):<47}║")
                    self.memory.record_llm_time(llm_elapsed)
                    total_llm_time += llm_elapsed
                    port_result["from_cache"] = False
                    port_result["llm_time_s"] = round(llm_elapsed, 1)

                # Phase 5: Verification (with 0-100% compile progress bar)
                self.disp.phase("Verifying", "✅")
                ref_path = Path(reference_dir) / f"{Path(cr['file']).stem}_output.txt"
                reference_output = ref_path.read_text(encoding="utf-8") if ref_path.exists() else ""

                # Live compile progress bar callback
                _compile_pct = [0]
                _compile_stage = [""]
                _compile_stop = threading.Event()
                _compile_spin = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

                def _compile_progress(pct: int, stage: str):
                    _compile_pct[0] = pct
                    _compile_stage[0] = stage

                def _compile_bar():
                    si = 0
                    while not _compile_stop.is_set():
                        si = (si + 1) % len(_compile_spin)
                        pct = _compile_pct[0]
                        stage = _compile_stage[0]
                        # Progress bar: [████████░░░░░░░░] 45%
                        filled = pct // 5
                        bar = "█" * filled + "░" * (20 - filled)
                        sys.stdout.write(
                            f"\r║  {_compile_spin[si]} {bold('hipcc'):<10} "
                            f"[{cyan(bar)}] {cyan(f'{pct:3d}%')} "
                            f"{dim(stage[:28]):<30}"
                        )
                        sys.stdout.flush()
                        time.sleep(0.1)

                ct = threading.Thread(target=_compile_bar, daemon=True)
                ct.start()

                ver_result = self.verifier.verify(
                    hip_source=port_result.get("ported_code", source),
                    cuda_reference_output=reference_output,
                    kernel_name=Path(cr['file']).stem,
                    on_progress=_compile_progress
                )

                _compile_stop.set()
                ct.join(timeout=1)
                sys.stdout.write("\r" + " " * 80 + "\r")  # clear progress bar
                sys.stdout.flush()

                ver_result["confidence"] = port_result.get("confidence", 0)
                # Attach in-loop compile errors to verification result
                if port_result.get("compile_errors"):
                    ver_result["in_loop_compile_errors"] = port_result["compile_errors"]
                verification_results.append(ver_result)

                # Always save ported kernel to ported_kernels/
                manual_dir = Path.cwd() / "ported_kernels"
                manual_dir.mkdir(parents=True, exist_ok=True)
                manual_path = manual_dir / (Path(cr["file"]).stem + ".hip.cpp")
                kernel_code = port_result.get("ported_code", source)
                # Auto-detect kernel function name from ported code
                import re as _re
                _km = _re.search(r'__global__\s+void\s+(\w+)\s*\(', kernel_code)
                _kname = _km.group(1) if _km else "warp_reduce_kernel"
                # Wrap in minimal harness so it's compilable standalone
                lines = [
                    "#include <iostream>",
                    "#include <hip/hip_runtime.h>",
                    "#include <cmath>",
                    "#include <vector>",
                    "",
                    kernel_code,
                    "",
                    "int main() {",
                    "    const int N = 256;",
                    "    std::vector<float> input(N, 1.0f);",
                    "    std::vector<float> output(N, 0.0f);",
                    "    float *d_in, *d_out;",
                    "    hipMalloc(&d_in, N * sizeof(float));",
                    "    hipMalloc(&d_out, N * sizeof(float));",
                    "    hipMemcpy(d_in, input.data(), N * sizeof(float), hipMemcpyHostToDevice);",
                    f"    {_kname}<<<4, 64>>>(d_in, d_out, N);",
                    "    hipDeviceSynchronize();",
                    "    hipMemcpy(output.data(), d_out, 4 * sizeof(float), hipMemcpyDeviceToHost);",
                    "    for (int i = 0; i < 4; i++) {",
                    '        printf("Block %d sum: %.0f\\n", i, output[i]);',
                    "    }",
                    "    bool pass = true;",
                    "    for (int i = 0; i < 4; i++) {",
                    "        if (fabs(output[i] - 64.0f) > 0.001f) pass = false;",
                    "    }",
                    '    printf("TEST: %s\\n", pass ? "PASSED ✅" : "FAILED ❌");',
                    "    hipFree(d_in); hipFree(d_out);",
                    "    return pass ? 0 : 1;",
                    "}",
                ]
                harness = "\n".join(lines)
                try:
                    manual_path.write_text(harness, encoding="utf-8")
                except Exception as e:
                    print(f"  ⚠️ Failed to save: {e}")

                if ver_result.get("passed"):
                    self.memory.store(
                        pattern_snippet=source[:500],
                        verified_fix=port_result.get("ported_code", "")[:500],
                        confidence=port_result.get("confidence", 80) / 100.0,
                        verification_run_id=ver_result.get("compile_output", "")[:20],
                        llm_time_s=port_result.get("llm_time_s", 0.0),
                        findings=cr.get("findings", [])
                    )
                    self.disp.status("Verifying", f"{Path(cr['file']).name} {green('VERIFIED')}", ok=True)
                elif not ver_result.get("compile_success") and "hipcc not found" in ver_result.get("compile_output", ""):
                    # GPU unavailable — store as unverified template fix for the
                    # "second kernel is faster" demo. Uses lower confidence.
                    self.memory.store(
                        pattern_snippet=source[:500],
                        verified_fix=port_result.get("ported_code", "")[:500],
                        confidence=0.70,  # lower confidence — unverified
                        verification_run_id="template_unverified",
                        llm_time_s=port_result.get("llm_time_s", 0.0),
                        findings=cr.get("findings", [])
                    )
                    self.disp.status("Verifying", f"{Path(cr['file']).name} {yellow('stored (unverified — no GPU)')}", ok=False)
                else:
                    reason = verification_failure_label(ver_result)
                    self.disp.status("Verifying", f"{Path(cr['file']).name} {yellow(reason)}", ok=False)
                    # Show compile errors or hipcc status.
                    # verifier._compile() already strips the temp build-dir prefix from
                    # every line, so a naive line[:N] truncation no longer eats the
                    # path instead of the message — but prioritize actual "error:"
                    # lines over the first 3 raw lines (which used to be path/caret noise).
                    compile_output = ver_result.get("compile_output", "")
                    if compile_output and "hipcc not found" not in compile_output:
                        all_lines = compile_output.strip().splitlines()
                        error_lines = [l for l in all_lines if "error:" in l] or all_lines
                        for line in error_lines[:3]:
                            print(f"║  ⚠️  {red(line[:65]):<64}║")
                        log_path = ver_result.get("compile_log_path", "")
                        if log_path:
                            print(f"║  ℹ️ Full compile log: {dim(log_path):<44}║")
                    if not ver_result.get("hipcc_available", True):
                        print(f"║  ⚠️  {'hipcc not found — export PATH=/opt/rocm-7.2.1/bin:$PATH':<64}║")
                    # Bug 0: the binary's own stdout/stderr is the actual explanation
                    # for a runtime failure and was captured but never printed —
                    # the same class of blindness Bug 1 (in the harness fix plan)
                    # found in the compile path.
                    if ver_result.get("compile_success") and not ver_result.get("run_success"):
                        run_output = ver_result.get("run_output", "")
                        if run_output:
                            for line in run_output.strip().splitlines()[:5]:
                                print(f"║  ⚠️  {red(line[:65]):<64}║")
                    # S3: Cache best-attempt code even when verification fails, so
                    # re-runs can rebuild from the closest-working version instead
                    # of starting from scratch (which costs ~10 min per kernel).
                    # This solves the 0% cache-hit-rate problem for long-running loops.
                    best_code = port_result.get("best_attempt_code", "")
                    best_iter = port_result.get("best_attempt_iteration", 0)
                    if best_code:
                        best_confidence = port_result.get("best_attempt_confidence", 0.15)
                        self.memory.store(
                            pattern_snippet=source[:500],
                            verified_fix=best_code[:500],
                            confidence=best_confidence,
                            verification_run_id=f"best_attempt_iter_{best_iter}",
                            llm_time_s=port_result.get("llm_time_s", 0.0),
                            findings=cr.get("findings", [])
                        )
                        print(f"║  {'📦 Cached best attempt (iter ' + str(best_iter) + ') @ ' + str(best_confidence*100) + '% confidence':<64}║")
                    else:
                        # No best-attempt code either — still cache the ported_code
                        # with a token confidence so re-runs have a starting point.
                        fallback_code = port_result.get("ported_code", "")
                        if fallback_code:
                            self.memory.store(
                                pattern_snippet=source[:500],
                                verified_fix=fallback_code[:500],
                                confidence=0.10,
                                verification_run_id="best_attempt_fallback",
                                llm_time_s=port_result.get("llm_time_s", 0.0),
                                findings=cr.get("findings", [])
                            )
                            print(f"║  {'📦 Cached fallback code @ 10% confidence (no compile success)':<64}║")
            else:
                self.disp.file_done(Path(cr['file']).name, f"{cr.get('risk_level')} — no porting needed", ok=True)

        pipeline_state["patterns_after"] = self.memory.count()

        # Phase 6: Report Generator
        self.disp.divider()
        self.disp.phase("Report", "📊")
        report = self.reporter.generate(
            scan_results=scan_results,
            classifier_results=classifier_results,
            verification_results=verification_results,
            memory_stats=self.memory.get_stats(),
            hours_per_fix=4.0
        )
        report["pipeline_state"] = pipeline_state
        self.disp.status("Report", "Generated", ok=True)

        # Final display
        self.disp.final_summary(self.memory.get_stats(), pipeline_state)
        return report


def doctor():
    """Pre-flight check: validate environment, dependencies, and configuration."""
    import subprocess
    import shutil
    import sys
    import platform

    checks = []
    all_ok = True

    def _check(name, ok, detail="", warn=False):
        """Record a check. warn=True marks an optional check: it shows a
        yellow '!' when absent but does NOT fail the preflight, so the
        doctor stays green on a machine without a GPU or API keys."""
        nonlocal all_ok
        if ok:
            mark = green("✓")
        elif warn:
            mark = yellow("!")
        else:
            all_ok = False
            mark = red("✗")
        checks.append((name, ok, detail, warn))
        print(f"  {mark} {bold(name):<30} {dim(detail)}")

    print()
    print(bold("╔═ Kernel Olympics — Doctor ══════════════════════════════╗"))
    print()

    # 1. Python version
    py_ok = sys.version_info >= (3, 10)
    _check("Python >= 3.10", py_ok, platform.python_version())

    # 2. Critical import check
    critical_imports = [
        ("json", True),
        ("sqlite3", True),
        ("argparse", True),
        ("hashlib", True),
        ("re", True),
        ("pathlib", True),
        ("subprocess", True),
    ]
    for mod_name, required in critical_imports:
        try:
            __import__(mod_name)
            _check(f"stdlib: {mod_name}", True, "ok")
        except ImportError:
            level = required
            _check(f"stdlib: {mod_name}", not required, "MISSING — may affect runtime")

    # 3. Optional pip package check
    # Core pipeline is pure-stdlib; pytest is the only (dev/CI) pip dep.
    pip_packages = [
        "pytest",
    ]
    for pkg in pip_packages:
        try:
            __import__(pkg.replace("-", "_"))
            _check(f"pip: {pkg}", True, "installed")
        except ImportError:
            _check(f"pip: {pkg}", False, "not installed — 'pip install -r requirements.txt'", warn=True)

    # 4. Project module imports (from src/)
    project_modules = [
        ("scanner", "Scanner"),
        ("risk_classifier", "RiskClassifier"),
        ("pattern_memory", "PatternMemory"),
        ("porting_agent", "PortingAgent"),
        ("router", "ModelRouter"),
        ("verification", "VerificationAgent"),
        ("report_generator", "ReportGenerator"),
    ]
    sys.path.insert(0, str(Path(__file__).parent))
    for module_name, class_name in project_modules:
        try:
            mod = __import__(module_name)
            if hasattr(mod, class_name):
                _check(f"module: {module_name}", True, f"{class_name} OK")
            else:
                _check(f"module: {module_name}", True, f"loaded (no {class_name})")
        except ImportError as e:
            _check(f"module: {module_name}", False, f"ImportError: {e}")

    # 5. API keys
    api_keys = [
        ("DEEPSEEK_API_KEY", "DeepSeek (LLM fallback)", False),
        ("FIREWORKS_API_KEY", "Fireworks AI (primary LLM)", False),
        ("DEEPSEEK_MODEL", "DeepSeek model", True),
    ]
    for var_name, label, optional in api_keys:
        val = os.getenv(var_name, "")
        if val:
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            _check(f"env: {var_name}", True, f"{masked}")
        elif optional:
            _check(f"env: {var_name}", True, f"not set (optional — using default)")
        else:
            _check(f"env: {var_name}", False,
                   "not set — LLM porting disabled, template fallback used", warn=True)

    # 6. Directory structure
    required_dirs = [
        ("sample_kernels", True),
        ("sample_kernels/cuda", True),
        ("sample_kernels/reference", True),
        ("data", False),
        ("ported_kernels", False),
    ]
    root = Path(__file__).parent.parent
    for dir_name, required in required_dirs:
        d = root / dir_name
        exists = d.is_dir()
        _check(f"dir: {dir_name}", exists or not required,
               "found" if exists else ("MISSING" if required else "not found (created on demand)"))

    # 7. GPU tooling (optional — the tool runs without a GPU via template
    #    porting + unverified storage, so absence is a warning, not a failure)
    import subprocess as _sp
    hipify_path = shutil.which("hipify-clang")
    _check("hipify-clang", bool(hipify_path),
           f"at {hipify_path}" if hipify_path else "not found — scanner falls back", warn=True)

    # Thorough hipcc detection — check common paths, which, and command -v
    hipcc_candidates = ["hipcc", "/opt/rocm/bin/hipcc", "/opt/rocm/lib/llvm/bin/hipcc",
                        "/opt/rocm-7.2.1/bin/hipcc", "/opt/rocm-7.2.1/lib/llvm/bin/hipcc",
                        "/usr/bin/hipcc"]
    hipcc_found = None
    for cmd in hipcc_candidates:
        try:
            _r = _sp.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
            if _r.returncode == 0:
                hipcc_found = cmd
                break
        except (FileNotFoundError, _sp.TimeoutExpired):
            continue
    if not hipcc_found:
        try:
            _r = _sp.run("which hipcc 2>/dev/null || command -v hipcc 2>/dev/null",
                         shell=True, capture_output=True, text=True, timeout=5)
            if _r.stdout.strip():
                hipcc_found = _r.stdout.strip()
        except _sp.TimeoutExpired:
            pass
    if not hipcc_found:
        try:
            _r = _sp.run("hipcc --version", shell=True, capture_output=True, text=True, timeout=5)
            if _r.returncode == 0:
                hipcc_found = "hipcc"
        except _sp.TimeoutExpired:
            pass
    _check("hipcc (ROCm)", bool(hipcc_found),
           f"at {hipcc_found}" if hipcc_found else "not found — GPU verification unavailable", warn=True)

    # 8. Network connectivity (to Fireworks API) — skipped cleanly without a key
    if not os.getenv("FIREWORKS_API_KEY"):
        _check("network: Fireworks API", True, "skipped — no API key")
    else:
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.fireworks.ai/v1/models",
                method="HEAD",
                headers={"Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY', '')}"}
            )
            urllib.request.urlopen(req, timeout=5)
            _check("network: Fireworks API", True, "reachable")
        except Exception as e:
            _check("network: Fireworks API", False,
                   f"unreachable ({type(e).__name__})", warn=True)

    # 9. SQLite write test (pattern memory)
    try:
        import sqlite3
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            conn = sqlite3.connect(tf.name)
            conn.execute("CREATE TABLE doctor (k TEXT, v TEXT)")
            conn.execute("INSERT INTO doctor VALUES ('ping', 'pong')")
            conn.close()
        _check("sqlite: write test", True, "OK")
    except Exception as e:
        _check("sqlite: write test", False, f"FAILED: {e}")

    # 10. Disk space (data directory)
    try:
        st = shutil.disk_usage(root)
        free_gb = st.free / (1024**3)
        _check("disk: free space", free_gb > 0.1, f"{free_gb:.1f} GB free")
    except Exception:
        _check("disk: free space", True, "unable to check")

    # ── Summary ──────────────────────────────────────────────────
    failed = [n for n, ok, _, warn in checks if not ok and not warn]
    warnings = [(n, d) for n, ok, d, warn in checks if not ok and warn]
    print()
    if all_ok:
        print(f"  {bold(green('RESULT: ALL REQUIRED CHECKS PASSED'))}")
    else:
        print(f"  {bold(red(f'RESULT: {len(failed)} required check(s) FAILED'))}")
        for n in failed:
            print(f"         {red('✗')} {n}")
    if warnings:
        print(f"  {yellow(f'{len(warnings)} optional check(s) not available (safe to proceed):')}")
        for n, d in warnings:
            print(f"         {yellow('!')} {n} {dim('— ' + d) if d else ''}")
    print()
    print(bold("╚════════════════════════════════════════════════════════╝"))
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Kernel Olympics — CUDA→ROCm Migration Copilot")
    parser.add_argument("--input", nargs="+", required=False, help="CUDA kernel files to analyze")
    parser.add_argument("--reference", default="sample_kernels/reference", help="Reference outputs directory")
    parser.add_argument("--output", default="portability_report.json", help="Output path for JSON report")
    parser.add_argument("--demo", action="store_true", help="Run 'second kernel is faster' speedup demo")
    parser.add_argument("--reset", action="store_true", help="With --demo, clear pattern memory before running")
    parser.add_argument("--fresh", action="store_true", help="Start with an empty pattern memory (clears the cache DB before running)")
    parser.add_argument("--doctor", action="store_true", help="Run pre-flight environment check and exit")
    parser.add_argument("--nvidia-sample", type=str, nargs="?",
                        const="cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu",
                        help="Download and test a sample from NVIDIA/cuda-samples (default: shfl_scan)")
    parser.add_argument("--daemon", action="store_true",
                        help="Run in daemon mode: watch a directory for new .cu files and auto-process")
    parser.add_argument("--watch", default="sample_kernels/cuda",
                        help="Directory to watch for .cu files (default: sample_kernels/cuda)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Poll interval in seconds (default: 5)")
    args = parser.parse_args()

    if args.doctor:
        return doctor()

    if args.demo:
        return run_demo(reset=args.reset)

    if args.daemon:
        return run_daemon(
            watch_dir=args.watch,
            interval=args.interval,
            reference_dir=args.reference,
            fresh=args.fresh,
        )

    if not args.input and not args.nvidia_sample:
        parser.error("--input or --nvidia-sample is required unless --demo or --doctor is used")
        return 1

    if args.nvidia_sample:
        sample_path = args.nvidia_sample
        url = f"https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/{sample_path}"
        filename = Path(sample_path).name
        local_path = Path(f"/tmp/nvidia_{filename}")
        
        print(green(f"\n  ┌─ NVIDIA CUDA SAMPLE ─────────────────────────────────────┐"))
        print(green(f"  │ Source: NVIDIA/cuda-samples"))
        print(green(f"  │ File:   {sample_path}"))
        print(green(f"  │ URL:    {url}"))
        print(green(f"  └──────────────────────────────────────────────────────────┘\n"))
        
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Kernel-Olympics/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                local_path.write_bytes(resp.read())
            lines = len(local_path.read_text(encoding="utf-8").splitlines())
            print(cyan(f"  ↓ Downloaded: {filename} ({lines} lines)\n"))
            args.input = [str(local_path)]
        except Exception as e:
            print(yellow(f"  ⚠️ Download failed ({e}) — trying local copy..."))
            # Fallback: use local copy from repo
            local_repo_path = Path(f"sample_kernels/cuda/nvidia_{filename}")
            if local_repo_path.exists():
                lines = len(local_repo_path.read_text(encoding="utf-8").splitlines())
                print(cyan(f"  ✓ Using local copy: {local_repo_path} ({lines} lines)\n"))
                args.input = [str(local_repo_path)]
            else:
                print(red(f"  ✗ No local copy either. Download manually and save to sample_kernels/cuda/"))
                return 1

    ko = KernelOlympics(fresh=args.fresh)
    report = ko.run(args.input, args.reference)

    if report.get("error") == "no_input":
        print(red("✖ Pipeline failed — no input files."))
        return 1

    output_path = Path(args.output)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {output_path}")


def run_demo(reset: bool = False):
    """Demo: 'second kernel is faster' — pattern memory speedup showcase.

    Args:
        reset: If True, clear all existing patterns and start fresh.
               If False (default), use any existing patterns for immediate speedup.
    """
    from pattern_memory.memory import PatternMemory
    from porting_agent.agent import PortingAgent
    import os

    # Decide whether we can do realistic LLM timing
    has_api = bool(os.getenv("FIREWORKS_API_KEY"))

    print(bold("╔═ Kernel Olympics — Demo Mode ══════════════════════════╗"))
    print(bold("║") + " Demonstrating: Pattern Memory 'Second Kernel is Faster'  " + bold("║"))
    print(bold("╠════════════════════════════════════════════════════════╣"))

    demo_memory = PatternMemory()
    demo_porter = PortingAgent()

    existing_count = demo_memory.count()
    if reset:
        demo_memory.clear()
        existing_count = 0
        print(f"║ {yellow('●')} Pattern memory cleared — starting fresh        ")

    mode = green("LIVE LLM") if has_api else yellow("simulated LLM (no API key)")
    if existing_count > 0:
        print(f"║ {green('●')} {bold(str(existing_count))} patterns already cached — "
              f"{green('immediate speedup available!')}  ")
    else:
        print(f"║ {green('●')} Pattern memory empty — first kernel will use LLM  ")
    print(f"║ {green('●')} Mode: {mode}                    ")
    print(f"║ {dim('Tip:')} run {bold('--demo --reset')} to start fresh            ")

    # First kernel: warp_reduce.cu
    print(bold("╠════════════════════════════════════════════════════════╣"))
    warp_source = Path("sample_kernels/cuda/warp_reduce.cu").read_text(encoding="utf-8")

    # Check if warp_reduce is already cached
    warp_cached = demo_memory.retrieve(warp_source)
    if warp_cached and not reset:
        print(f"║ {bold('Kernel 1:')} warp_reduce.cu — {green('ALREADY CACHED!')}       ")
        cache_ms = warp_cached.get("retrieval_ms", 0.3)
        jaccard_val = warp_cached.get("jaccard", 0)
        print(f"║ {green('●')} Retrieved in {green(f'{cache_ms:.1f}ms')}  "
              f"{dim(f'(jaccard: {jaccard_val:.0%})')}")
        # For the speed comparison, still show a "what if it was LLM" baseline
        llm_elapsed = warp_cached.get("llm_time_s", 0.0) or 0.0
        simulated_first = True
        warp_port_result = {
            "ported_code": warp_cached["verified_fix"],
            "confidence": warp_cached["confidence"] * 100,
            "changes": ["Retrieved from pattern memory cache"],
            "from_cache": True,
            "llm_time_s": 0
        }
    else:
        print(f"║ {bold('Kernel 1:')} warp_reduce.cu — {yellow('NO cached pattern')}     ")
        t0 = time.perf_counter()
        warp_port_result = demo_porter.port_kernel(warp_source)
        llm_elapsed = time.perf_counter() - t0

        simulated_first = False
        if not has_api:
            simulated_first = True
            print(f"║ {dim('(template port completed in {:.2f}s — simulated LLM)')}  ".format(
                llm_elapsed))

        # Store with forced LLM-time simulation
        demo_memory.record_llm_time(llm_elapsed)
        pid = demo_memory.store(
            pattern_snippet=warp_source[:500],
            verified_fix=warp_port_result["ported_code"][:500],
            confidence=warp_port_result["confidence"] / 100.0,
            verification_run_id="demo_1",
            llm_time_s=round(max(llm_elapsed, 0.001), 3)
        )
        n_changes = len(warp_port_result.get("changes", []))
        sim_tag = yellow(" (simulated)") if simulated_first else ""
        print(f"║ {green('●')} Ported in {yellow(f'{llm_elapsed:.1f}s')}{sim_tag}  {dim(str(n_changes) + ' changes')}")
        print(f"║ {green('●')} Pattern stored — id: {dim(pid)}     ")

    # Second kernel: histogram.cu (similar patterns)
    print(bold("╠════════════════════════════════════════════════════════╣"))
    hist_source = Path("sample_kernels/cuda/histogram.cu").read_text(encoding="utf-8")
    print(f"║ {bold('Kernel 2:')} histogram.cu — {green('cache lookup...')}        ")
    cached = demo_memory.retrieve(hist_source)

    t1 = time.perf_counter()
    if cached:
        cache_ms = cached.get("retrieval_ms", 0.3)
        jaccard = cached.get("jaccard", 0)
        hist_result = {
            "ported_code": cached["verified_fix"],
            "confidence": cached["confidence"] * 100,
            "changes": [f"Applied cached fix (pattern {cached['id']}) — LLM skipped"],
            "from_cache": True,
            "llm_time_s": cache_ms / 1000
        }
        time.sleep(0.001)
        llm_elapsed2 = time.perf_counter() - t1
        print(f"║ {green('●')} {green('CACHE HIT!')} Retrieved in {green(f'{cache_ms:.1f}ms')}    ")
        print(f"║ {green('●')} {dim(f'Jaccard similarity: {jaccard:.0%}')}               ")
    else:
        hist_result = demo_porter.port_kernel(hist_source)
        llm_elapsed2 = time.perf_counter() - t1
        print(f"║ {yellow('●')} Cache miss — ported in {yellow(f'{llm_elapsed2:.2f}s')}       ")
        # Store the histogram result for future runs
        demo_memory.store(
            pattern_snippet=hist_source[:500],
            verified_fix=hist_result["ported_code"][:500],
            confidence=hist_result["confidence"] / 100.0,
            verification_run_id="demo_2",
            llm_time_s=round(max(llm_elapsed2, 0.001), 3)
        )
        print(f"║ {dim('●')} Histogram pattern stored for next demo run       ")

    # Summary
    print(bold("╠════════════════════════════════════════════════════════╣"))
    print(f"║ {bold('Speed Comparison')}                                       ")
    first_ms = llm_elapsed * 1000  # ms
    second_ms = (cached.get("retrieval_ms", 0.3) if cached else llm_elapsed2 * 1000)
    speedup_val = first_ms / max(second_ms, 0.1)
    speedup_str = f"{speedup_val:.0f}×" if second_ms > 0 else "N/A"

    if simulated_first or existing_count > 0:
        print(f"║ {green('●')} Kernel 1 ({dim('LLM call, no cache')}): "
              f"{yellow(f'{first_ms:.0f}ms')} {'':>6} {dim('(simulated LLM — template port timing)')}")
    else:
        print(f"║ {green('●')} Kernel 1 ({dim('LLM call, no cache')}): "
              f"{yellow(f'{first_ms:.0f}ms')}")
    print(f"║ {green('●')} Kernel 2 ({dim('cache hit')}): "
          f"{green(f'{second_ms:.1f}ms')}                         ")
    print(f"║ {green('●')} {bold('Speedup:')} {cyan(speedup_str)} "
          f"{dim(f'(analysis: ~{llm_elapsed:.1f}s LLM → ~{second_ms:.0f}ms cache)')}")
    print(f"║                                                 ")
    if cached:
        print(f"║ {dim('Pattern memory avoided a {:.2f}s LLM call'.format(llm_elapsed))}")
    print(f"║ {dim('Demo complete. Pattern memory proves: similar kernels')}")
    print(f"║ {dim('get faster as the cache grows.')}   ")
    print(bold("╚════════════════════════════════════════════════════════╝"))

    # Save demo report
    stats = demo_memory.get_stats()
    report = {
        "demo": True,
        "mode": "simulated" if not has_api else "live_llm",
        "first_kernel": {"name": "warp_reduce.cu", "time_s": round(llm_elapsed, 3), "from_cache": bool(warp_cached and not reset)},
        "second_kernel": {"name": "histogram.cu", "time_s": round(second_ms / 1000, 4), "from_cache": bool(cached)},
        "speedup_ratio": round(speedup_val, 0),
        "speedup_label": speedup_str,
        "analysis": f"LLM: {llm_elapsed:.1f}s → Cache: {second_ms:.0f}ms",
        "memory_stats": stats,
        "patterns_in_cache": existing_count,
        "warp_first_time": not bool(warp_cached and not reset)
    }
    Path("demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"├{'─'*66}┤")
    print(f"║ {dim('Demo report saved to: demo_report.json')}")
    print(bold("╚════════════════════════════════════════════════════════╝"))
    return 0


# ── Daemon / watch mode ──────────────────────────────────────────

_DAEMON_STATE_FILE = os.path.join(
    os.path.expanduser("~"), ".kernel-olympics", "daemon_state.json"
)
_SHUTDOWN_REQUESTED = False


def _daemon_signal_handler(signum, frame):
    """Set global shutdown flag on SIGINT/SIGTERM for clean exit."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("\n[daemon] Shutdown signal received — finishing current file...",
          file=sys.stderr)


class DaemonState:
    """Tracks which .cu files have already been processed by the daemon.
    Persisted as a JSON dict on disk so the daemon survives restarts."""
    def __init__(self, state_path: str = _DAEMON_STATE_FILE):
        self._path = state_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def is_processed(self, file_path: str, mtime: float, size: int) -> bool:
        entry = self._data.get(file_path)
        if entry is None:
            return False
        return entry.get("mtime") == mtime and entry.get("size") == size

    def mark_processed(self, file_path: str, mtime: float, size: int,
                       status: str = "ok", result: str = ""):
        self._data[file_path] = {
            "mtime": mtime, "size": size,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status, "result": result,
        }
        self._save()

    def list_pending(self, watch_dir: str) -> list[str]:
        pending = []
        watch = Path(watch_dir)
        if not watch.is_dir():
            return pending
        for fpath in sorted(watch.glob("*.cu")):
            try:
                stat = fpath.stat()
                entry = self._data.get(str(fpath))
                if entry is None or entry.get("mtime") != stat.st_mtime or entry.get("size") != stat.st_size:
                    pending.append(str(fpath))
            except OSError:
                continue
        return pending


def _process_single_file(ko: "KernelOlympics", file_path: str,
                         reference_dir: str) -> dict:
    try:
        report = ko.run([file_path], reference_dir=reference_dir)
        return report
    except Exception as exc:
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc(), "file": file_path}


def run_daemon(watch_dir: str, interval: int = 5,
               reference_dir: str = "sample_kernels/reference",
               fresh: bool = False, state_path: str | None = None):
    """Watch *watch_dir* for new/edited .cu files and process them automatically."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = False
    import signal
    signal.signal(signal.SIGINT, _daemon_signal_handler)
    signal.signal(signal.SIGTERM, _daemon_signal_handler)

    state = DaemonState(state_path or _DAEMON_STATE_FILE)
    ko = KernelOlympics(fresh=fresh, silent=True)

    watch = Path(watch_dir)
    if not watch.is_dir():
        print(f"[daemon] ERROR: watch directory does not exist: {watch_dir}", file=sys.stderr)
        return 1

    print(f"[daemon] Watching {watch.resolve()} every {interval}s (PID {os.getpid()})", file=sys.stderr)
    sys.stderr.flush()

    while not _SHUTDOWN_REQUESTED:
        try:
            pending = state.list_pending(watch_dir)
            for fpath_str in pending:
                if _SHUTDOWN_REQUESTED:
                    break
                fpath = Path(fpath_str)
                try:
                    stat = fpath.stat()
                    mtime = stat.st_mtime
                    size = stat.st_size
                except OSError as e:
                    print(f"[daemon] ERROR: cannot stat {fpath}: {e}", file=sys.stderr)
                    continue
                print(f"[daemon] Processing: {fpath.name}", file=sys.stderr)
                sys.stderr.flush()
                report = _process_single_file(ko, fpath_str, reference_dir)
                if "error" in report:
                    print(f"[daemon] ERROR: {fpath.name}: {report['error']}", file=sys.stderr)
                    state.mark_processed(fpath_str, mtime, size, status="error", result=report["error"])
                else:
                    state.mark_processed(fpath_str, mtime, size, status="ok", result="done")
                    print(f"[daemon] Completed: {fpath.name}", file=sys.stderr)
                sys.stderr.flush()
        except Exception as loop_exc:
            import traceback as _tb
            print(f"[daemon] Loop error: {loop_exc}", file=sys.stderr)
        if not _SHUTDOWN_REQUESTED:
            for _ in range(interval):
                if _SHUTDOWN_REQUESTED:
                    break
                time.sleep(1)
    print(f"[daemon] Shutdown complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
