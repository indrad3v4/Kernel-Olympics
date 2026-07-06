"""Unit tests for the report generator module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from report_generator.reporter import ReportGenerator


def test_report_basic():
    """Report should include all required fields."""
    rg = ReportGenerator()
    report = rg.generate(
        scan_results=[
            {"file": "test.cu", "hipify_coverage_pct": 40, "total_lines": 30}
        ],
        classifier_results=[
            {"file": "test.cu", "risk_level": "red", "findings": [
                {"pattern": "warp_size", "line": 7, "severity": "high"}
            ]}
        ],
        verification_results=[
            {"passed": True, "kernel": "test", "confidence": 85}
        ],
        memory_stats={"total_patterns": 3, "avg_confidence": 87, "total_retrievals": 2},
        hours_per_fix=4.0
    )
    assert "report_id" in report
    assert "summary" in report
    assert "statistics" in report
    assert "engineer_hours_saved" in report
    assert "sections" in report
    assert report["statistics"]["files_scanned"] == 1
    assert report["statistics"]["verifications_passed"] == 1


def test_report_hours_saved():
    """Hours saved should be verified_count * hours_per_fix."""
    rg = ReportGenerator()
    report = rg.generate(
        scan_results=[{"file": "a.cu", "hipify_coverage_pct": 50, "total_lines": 10}],
        classifier_results=[{"file": "a.cu", "risk_level": "red", "findings": [{"pattern": "test", "line": 1, "severity": "high"}]}],
        verification_results=[
            {"passed": True, "confidence": 90},
            {"passed": True, "confidence": 85}
        ],
        memory_stats={"total_patterns": 0, "avg_confidence": 0, "total_retrievals": 0},
        hours_per_fix=4.0
    )
    assert report["engineer_hours_saved"] == 8.0  # 2 verified * 4h


def test_report_summary_includes_pattern_memory():
    """Template summary should include pattern memory stats when available."""
    rg = ReportGenerator()
    report = rg.generate(
        scan_results=[{"file": "a.cu", "hipify_coverage_pct": 100, "total_lines": 5}],
        classifier_results=[{"file": "a.cu", "risk_level": "green", "findings": []}],
        verification_results=[],
        memory_stats={"total_patterns": 5, "avg_confidence": 92, "total_retrievals": 10},
        hours_per_fix=4.0
    )
    summary = report["summary"]
    assert "5 verified patterns stored" in summary
    assert "92" in summary  # confidence
    assert "10" in summary  # retrievals


def test_report_risk_breakdown():
    """Risk breakdown should correctly count green/yellow/red."""
    rg = ReportGenerator()
    report = rg.generate(
        scan_results=[
            {"file": "a.cu", "hipify_coverage_pct": 50, "total_lines": 10},
            {"file": "b.cu", "hipify_coverage_pct": 50, "total_lines": 10},
            {"file": "c.cu", "hipify_coverage_pct": 50, "total_lines": 10},
        ],
        classifier_results=[
            {"file": "a.cu", "risk_level": "red", "findings": [{"pattern": "p1", "line": 1, "severity": "high"}]},
            {"file": "b.cu", "risk_level": "yellow", "findings": [{"pattern": "p2", "line": 2, "severity": "medium"}]},
            {"file": "c.cu", "risk_level": "green", "findings": []},
        ],
        verification_results=[],
        memory_stats={"total_patterns": 0, "avg_confidence": 0, "total_retrievals": 0},
        hours_per_fix=4.0
    )
    stats = report["statistics"]["risk_breakdown"]
    assert stats["red"] == 1
    assert stats["yellow"] == 1
    assert stats["green"] == 1


def test_template_summary_avoids_attribute_error():
    """Template summary should not crash with AttributeError (regression test)."""
    rg = ReportGenerator()
    try:
        report = rg.generate(
            scan_results=[{"file": "a.cu", "hipify_coverage_pct": 100, "total_lines": 5}],
            classifier_results=[{"file": "a.cu", "risk_level": "green", "findings": []}],
            verification_results=[{"passed": True, "kernel": "a", "confidence": 85}],
            memory_stats={"total_patterns": 1, "avg_confidence": 85, "total_retrievals": 0},
            hours_per_fix=4.0
        )
        assert "hours_saved" in report or "engineer_hours_saved" in report
    except AttributeError as e:
        assert False, f"AttributeError in report generation: {e}"
