"""
Report Generator — Gemma on local ROCm for cheap text generation.

Input: aggregated results from scanner + classifier + verifier
Model: Gemma (local via ROCm) 
Output: plain-English portability report + hours-saved estimate
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class ReportGenerator:
    """Generates portability reports using Gemma on local ROCm."""

    def __init__(self, use_gemma: bool = True):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        # Try latest Gemma 4 first, fall back to Gemma 3 if 404
        self.gemma_model = os.environ.get(
            "GEMMA_MODEL",
            "accounts/fireworks/models/gemma-4-31b-it"
        )
        self.use_gemma = use_gemma and bool(self.api_key)

    def generate(self, scan_results: List[Dict], classifier_results: List[Dict],
                 verification_results: List[Dict], memory_stats: Dict,
                 hours_per_fix: float = 4.0) -> Dict:
        """Generate a complete portability report."""
        
        # Store hours_per_fix for template access
        self.hours_per_fix = hours_per_fix

        # Calculate statistics
        total_files = len(scan_results)
        red_count = sum(1 for r in classifier_results if r.get("risk_level") == "red")
        yellow_count = sum(1 for r in classifier_results if r.get("risk_level") == "yellow")
        green_count = sum(1 for r in classifier_results if r.get("risk_level") == "green")
        
        total_findings = sum(len(r.get("findings", [])) for r in classifier_results)
        verified_count = sum(1 for v in verification_results if v.get("passed"))
        
        # Hours saved estimate
        hours_saved = verified_count * hours_per_fix
        
        # Average confidence from porting
        confidences = [v.get("confidence", 0) for v in verification_results if "confidence" in v]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Generate report sections
        if self.use_gemma:
            summary = self._gemma_summary(scan_results, classifier_results, verification_results)
        else:
            summary = self._template_summary(red_count, yellow_count, green_count, 
                                            verified_count, hours_saved, memory_stats)

        report = {
            "report_id": f"KO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "statistics": {
                "files_scanned": total_files,
                "risk_breakdown": {
                    "green": green_count,
                    "yellow": yellow_count,
                    "red": red_count
                },
                "total_danger_patterns_found": total_findings,
                "verifications_passed": verified_count,
                "verifications_total": len(verification_results),
                "avg_porting_confidence": round(avg_confidence, 1),
                "pattern_memory": memory_stats
            },
            "engineer_hours_saved": round(hours_saved, 1),
            "hours_per_fix_assumption": hours_per_fix,
            "sections": {
                "risk_classification": classifier_results,
                "verification": [
                    {k: v for k, v in vr.items() if k != "compile_output" and k != "run_output"}
                    for vr in verification_results
                ]
            }
        }

        return report

    def _template_summary(self, red: int, yellow: int, green: int,
                          verified: int, hours_saved: float, memory: Dict) -> str:
        """Generate a plain-English summary from template."""
        parts = []
        parts.append(f"Portability Analysis Complete")
        parts.append(f"")
        parts.append(f"Risk Overview:")
        parts.append(f"  • {red} file(s) flagged RED — high-risk patterns detected (silent wrong output if naively ported)")
        parts.append(f"  • {yellow} file(s) flagged YELLOW — ported but needs review")
        parts.append(f"  • {green} file(s) flagged GREEN — safe for auto-port")
        parts.append(f"")
        
        if verified > 0:
            parts.append(f"Auto-Fix Results:")
            parts.append(f"  • {verified} kernel(s) successfully ported and verified")
            parts.append(f"  • Estimated engineer-hours saved: {hours_saved:.0f}h")
            parts.append(f"    (based on {self.hours_per_fix}h per manually-fixed red-flag kernel)")
            parts.append(f"")
        
        if memory:
            parts.append(f"Pattern Memory:")
            parts.append(f"  • {memory.get('total_patterns', 0)} verified patterns stored")
            parts.append(f"  • Average confidence: {memory.get('avg_confidence', 0)}%")
            parts.append(f"  • Total retrievals: {memory.get('total_retrievals', 0)}")
            parts.append(f"  → System gets smarter with every verified fix")
        
        return "\n".join(parts)

    def _gemma_summary(self, scan_results, classifier_results, verification_results) -> str:
        """Use Gemma 3 via Fireworks API to generate a narrative summary."""
        import urllib.request
        import json as _json
        
        # Build prompt from pipeline results
        red = [r for r in classifier_results if r.get("risk_level") == "red"]
        findings = []
        for r in classifier_results:
            findings.extend(r.get("findings", []))
        
        prompt = (
            f"You are a GPU kernel porting expert. "
            f"Write a 3-paragraph portability report summary:\n\n"
            f"Files analyzed: {len(scan_results)}\n"
            f"High-risk kernels: {len(red)}\n"
            f"Danger patterns found: {len(findings)}\n"
            f"Verifications passed: {sum(1 for v in verification_results if v.get('passed'))}\n\n"
            f"Key patterns detected:\n"
        )
        for f in findings[:10]:
            prompt += f"- [{f.get('severity','info')}] Line {f.get('line','?')}: {f.get('pattern','?')}\n"
        prompt += "\nWrite a concise, professional portability report summary. Keep it under 200 words."
        
        try:
            data = _json.dumps({
                "model": self.gemma_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3
            }).encode()
            req = urllib.request.Request(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = _json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except Exception:
            pass  # fallback to template
        
        return self._template_summary(
            len(red),
            len([r for r in classifier_results if r.get("risk_level") == "yellow"]),
            len([r for r in classifier_results if r.get("risk_level") == "green"]),
            sum(1 for v in verification_results if v.get("passed")),
            0,
            {}
        )

    def _check_gemma_available(self) -> bool:
        """Check if Gemma is available via Fireworks API key."""
        return bool(self.api_key)


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from utf8_console import enable_utf8_console
    enable_utf8_console()

    # Quick test
    rg = ReportGenerator()
    report = rg.generate([], [], [], {"total_patterns": 12, "avg_confidence": 87, "total_retrievals": 5}, 4.0)
    print(report["summary"])
    print(f"\nHours saved: {report['engineer_hours_saved']}")
