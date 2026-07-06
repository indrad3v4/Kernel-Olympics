"""
Kernel Olympics — Main orchestrator.

Pipeline:
1. Scanner: runs hipify-clang dry-run on CUDA files
2. Risk Classifier: rule-based pattern matching for warp/wavefront divergence
3. Pattern Memory: vector store for verified fixes (cached patterns speed up future runs)
4. Porting Agent: Fireworks API (or template fallback) to fix red-flagged kernels
5. Verification Agent: compile + run + diff on AMD Developer Cloud
6. Report Generator: Gemma on local ROCm for plain-English summary

Usage:
    python main.py --input sample_kernels/cuda/*.cu
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from scanner.scanner import Scanner
from risk_classifier.classifier import RiskClassifier
from pattern_memory.memory import PatternMemory
from porting_agent.agent import PortingAgent
from verification.verifier import VerificationAgent
from report_generator.reporter import ReportGenerator


class KernelOlympics:
    """Orchestrates the full CUDA→ROCm migration pipeline."""

    def __init__(self):
        self.scanner = Scanner()
        self.classifier = RiskClassifier()
        self.memory = PatternMemory()
        self.porting_agent = PortingAgent()
        self.verifier = VerificationAgent()
        self.reporter = ReportGenerator()

    def run(self, input_paths: list[str], reference_dir: str = "sample_kernels/reference") -> dict:
        """Run the full pipeline on input CUDA files."""
        
        pipeline_state = {"phase": "initializing", "patterns_before": 0, "patterns_after": 0}
        
        # Phase 1: Scanner
        pipeline_state["phase"] = "scanning"
        print(f"[1/6] Scanning {len(input_paths)} file(s)...")
        scan_results = self.scanner.scan_batch(input_paths)
        print(f"  → {len(scan_results)} files scanned")
        for r in scan_results:
            print(f"     {Path(r['file']).name}: coverage {r.get('hipify_coverage_pct', 0)}%")

        # Phase 2: Risk Classifier
        pipeline_state["phase"] = "classifying"
        print(f"\n[2/6] Classifying portability risk...")
        file_sources = {}
        for fp in input_paths:
            try:
                file_sources[fp] = Path(fp).read_text()
            except:
                pass
        
        classifier_results = self.classifier.classify_batch(file_sources)
        self._print_risk_summary(classifier_results)

        # Phase 3: Pattern Memory — check for cached fixes (SKIP LLM on cache hit)
        pipeline_state["phase"] = "checking_memory"
        pipeline_state["patterns_before"] = self.memory.count()
        pipeline_state["cache_hits"] = 0
        pipeline_state["llm_calls"] = 0
        print(f"\n[3/6] Checking pattern memory ({self.memory.count()} stored patterns)...")
        
        # Phase 4: Porting Agent (or SKIP if cached)
        pipeline_state["phase"] = "porting"
        print(f"\n[4/6] Porting red-flagged kernels...")
        verification_results = []
        total_llm_time = 0.0
        
        for cr in classifier_results:
            if cr.get("risk_level") == "red":
                source = file_sources.get(cr["file"], "")
                if not source:
                    continue
                
                # Check pattern memory — trigram-based O(1) retrieval
                cached = self.memory.retrieve(source)
                if cached:
                    pipeline_state["cache_hits"] += 1
                    llm_saved = cached.get("llm_time_s", 12.0)
                    print(f"     ⏩ CACHED — {Path(cr['file']).name} "
                          f"(confidence: {cached.get('confidence', 0)*100:.0f}%, "
                          f"retrieved in {cached.get('retrieval_ms', 0):.1f}ms)")
                    print(f"       Estimated LLM time saved: {llm_saved:.1f}s")
                    
                    # Use cached fix — skip LLM entirely
                    port_result = {
                        "ported_code": cached["verified_fix"],
                        "confidence": cached["confidence"] * 100,
                        "changes": [f"Applied cached fix from pattern {cached['id']} "
                                    f"(LLM call skipped — saved ~{llm_saved:.0f}s)"],
                        "from_cache": True
                    }
                else:
                    pipeline_state["llm_calls"] += 1
                    print(f"     → No cached pattern for {Path(cr['file']).name}, calling porting agent...")
                    import time
                    t0 = time.perf_counter()
                    port_result = self.porting_agent.port_kernel(source)
                    llm_elapsed = time.perf_counter() - t0
                    self.memory.record_llm_time(llm_elapsed)
                    total_llm_time += llm_elapsed
                    port_result["from_cache"] = False
                    port_result["llm_time_s"] = round(llm_elapsed, 1)
                    print(f"     → LLM took {llm_elapsed:.1f}s — will be CACHED for next similar kernel")
                
                print(f"     → Confidence: {port_result.get('confidence', 0)}%")
                for change in port_result.get("changes", []):
                    print(f"       • {change[:80]}")
                
                # Phase 5: Verification
                pipeline_state["phase"] = "verifying"
                print(f"\n[5/6] Verifying ported kernel {Path(cr['file']).name}...")
                
                # Use reference output if available
                ref_path = Path(reference_dir) / f"{Path(cr['file']).stem}_output.txt"
                reference_output = ref_path.read_text() if ref_path.exists() else ""
                
                ver_result = self.verifier.verify(
                    hip_source=port_result.get("ported_code", source),
                    cuda_reference_output=reference_output,
                    kernel_name=Path(cr['file']).stem
                )
                ver_result["confidence"] = port_result.get("confidence", 0)
                verification_results.append(ver_result)
                
                # Store in pattern memory if verified
                if ver_result.get("passed"):
                    self.memory.store(
                        pattern_snippet=source[:500],
                        verified_fix=port_result.get("ported_code", "")[:500],
                        confidence=port_result.get("confidence", 80) / 100.0,
                        verification_run_id=ver_result.get("compile_output", "")[:20],
                        llm_time_s=port_result.get("llm_time_s", 0.0)
                    )
                    print(f"     ✓ Kernel VERIFIED and stored in pattern memory")
                else:
                    print(f"     ✗ Kernel verification FAILED")
                    if not ver_result.get("compile_success"):
                        print(f"       Compile error (expected without AMD GPU)")
            else:
                print(f"     → {Path(cr['file']).name}: {cr.get('risk_level')} — no porting needed")

        pipeline_state["patterns_after"] = self.memory.count()
        pipeline_state["phase"] = "reporting"

        # Phase 6: Report Generator
        print(f"\n[6/6] Generating portability report...")
        report = self.reporter.generate(
            scan_results=scan_results,
            classifier_results=classifier_results,
            verification_results=verification_results,
            memory_stats=self.memory.get_stats(),
            hours_per_fix=4.0
        )
        
        report["pipeline_state"] = pipeline_state
        
        print(f"\n{'='*60}")
        print(report["summary"])
        print(f"{'='*60}")
        print(f"\nEstimated engineer-hours saved: {report['engineer_hours_saved']}h")
        mem = self.memory.get_stats()
        print(f"Pattern memory: {pipeline_state['patterns_before']} → {pipeline_state['patterns_after']} patterns")
        print(f"Cache performance: {pipeline_state.get('cache_hits', 0)} hits, "
              f"{pipeline_state.get('llm_calls', 0)} LLM calls")
        if mem.get('last_cache_time_ms', 0) > 0:
            print(f"Fastest cache retrieval: {mem['last_cache_time_ms']}ms vs LLM: {mem['last_llm_time_s']}s "
                  f"(~{mem['last_llm_time_s'] / (max(mem['last_cache_time_ms'], 1)/1000):.0f}× faster)")
        
        return report

    def _print_risk_summary(self, results: list) -> None:
        """Print a summary of risk classification."""
        red = [r for r in results if r.get("risk_level") == "red"]
        yellow = [r for r in results if r.get("risk_level") == "yellow"]
        green = [r for r in results if r.get("risk_level") == "green"]
        
        print(f"  → RED: {len(red)} file(s) — would silently produce wrong output")
        print(f"  → YELLOW: {len(yellow)} file(s) — ported, needs review")
        print(f"  → GREEN: {len(green)} file(s) — safe for auto-port")
        
        for r in results:
            findings = r.get("findings", [])
            if findings:
                print(f"\n    {Path(r['file']).name}:")
                for f in findings:
                    print(f"      [{f['severity']}] L{f['line']}: {f['pattern']}")


def main():
    parser = argparse.ArgumentParser(description="Kernel Olympics — CUDA→ROCm Migration Copilot")
    parser.add_argument("--input", nargs="+", required=True,
                        help="CUDA kernel files to analyze")
    parser.add_argument("--reference", default="sample_kernels/reference",
                        help="Directory with reference outputs for verification")
    parser.add_argument("--output", default="portability_report.json",
                        help="Output path for JSON report")
    args = parser.parse_args()

    ko = KernelOlympics()
    report = ko.run(args.input, args.reference)

    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
