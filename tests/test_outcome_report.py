"""Tests for OutcomeTestingReport."""
import json
import tempfile
from pathlib import Path

from insurance_governance.mrm.model_card import ModelCard as MRMModelCard
from insurance_governance.outcome.report import OutcomeTestingReport
from insurance_governance.outcome.results import OutcomeResult
from insurance_governance.validation.results import RAGStatus, Severity


def make_card():
    return MRMModelCard(
        model_id="motor-v3",
        model_name="Motor Frequency v3",
        version="3.0.0",
    )


def make_results(n_pass=3, n_fail=1):
    results = []
    for i in range(n_pass):
        results.append(OutcomeResult(
            outcome="price_value",
            test_name=f"test_pass_{i}",
            passed=True,
            metric_value=0.75,
            threshold=0.70,
            period="2025-Q4",
            details="All good.",
        ))
    for i in range(n_fail):
        results.append(OutcomeResult(
            outcome="claims",
            test_name=f"test_fail_{i}",
            passed=False,
            metric_value=0.60,
            threshold=0.70,
            period="2025-Q4",
            severity=Severity.WARNING,
            details="Issue found.",
            corrective_actions=["Action 1", "Action 2"],
        ))
    return results


# --- Construction ---

def test_report_rag_computed_from_results():
    results = make_results(n_pass=2, n_fail=1)
    report = OutcomeTestingReport(make_card(), results, period="2025-Q4")
    assert report.rag_status == RAGStatus.AMBER


def test_report_rag_can_be_overridden():
    results = make_results(n_pass=3, n_fail=0)
    report = OutcomeTestingReport(
        make_card(), results, period="2025-Q4", rag_status=RAGStatus.RED
    )
    assert report.rag_status == RAGStatus.RED


def test_report_run_id_auto_generated():
    report = OutcomeTestingReport(make_card(), [], period="2025-Q4")
    assert len(report.run_id) == 36  # UUID format


def test_report_run_id_can_be_set():
    report = OutcomeTestingReport(make_card(), [], period="2025-Q4", run_id="test-id")
    assert report.run_id == "test-id"


# --- render_html ---

def test_render_html_returns_string():
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    html = report.render_html()
    assert isinstance(html, str)


def test_render_html_contains_model_name():
    report = OutcomeTestingReport(make_card(), [], period="2025-Q4")
    html = report.render_html()
    assert "Motor Frequency v3" in html


def test_render_html_contains_period():
    report = OutcomeTestingReport(make_card(), [], period="2025-Q4")
    html = report.render_html()
    assert "2025-Q4" in html


def test_render_html_contains_rag_status():
    results = [OutcomeResult(
        outcome="price_value", test_name="test", passed=False,
        severity=Severity.CRITICAL, period="2025-Q4",
    )]
    report = OutcomeTestingReport(make_card(), results, period="2025-Q4")
    html = report.render_html()
    assert "RED" in html.upper() or "red" in html


def test_render_html_shows_corrective_actions():
    results = [OutcomeResult(
        outcome="claims", test_name="fail_test", passed=False,
        severity=Severity.WARNING, period="2025-Q4",
        corrective_actions=["Do something specific now"],
    )]
    report = OutcomeTestingReport(make_card(), results, period="2025-Q4")
    html = report.render_html()
    assert "Do something specific now" in html


def test_render_html_segment_results():
    results = [OutcomeResult(
        outcome="price_value", test_name="fair_value", passed=True,
        segment="Renewal", period="2025-Q4",
    )]
    report = OutcomeTestingReport(make_card(), results, period="2025-Q4")
    html = report.render_html()
    assert "Renewal" in html


# --- write_html ---

def test_write_html_creates_file(tmp_path):
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    out = report.write_html(tmp_path / "report.html")
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_html_returns_path(tmp_path):
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    out = report.write_html(tmp_path / "report.html")
    assert isinstance(out, Path)


# --- to_dict / write_json ---

def test_to_dict_keys():
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    d = report.to_dict()
    for key in ("run_id", "period", "generated_date", "rag_status", "model_card", "results", "summary"):
        assert key in d


def test_to_dict_summary_counts():
    results = make_results(n_pass=3, n_fail=1)
    report = OutcomeTestingReport(make_card(), results, period="2025-Q4")
    d = report.to_dict()
    assert d["summary"]["total_tests"] == 4
    assert d["summary"]["passed"] == 3
    assert d["summary"]["failed"] == 1


def test_write_json_valid_json(tmp_path):
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    out = report.write_json(tmp_path / "report.json")
    data = json.loads(out.read_text())
    assert "results" in data


def test_write_json_returns_path(tmp_path):
    report = OutcomeTestingReport(make_card(), make_results(), period="2025-Q4")
    out = report.write_json(tmp_path / "report.json")
    assert isinstance(out, Path)


# --- top-level import ---

def test_top_level_import():
    from insurance_governance import (
        OutcomeTestingFramework,
        PriceValueMetrics,
        ClaimsMetrics,
        OutcomeResult,
        OutcomeSuite,
        CustomerSegment,
        SegmentComparison,
        OutcomeTestingReport,
    )
    assert OutcomeTestingFramework is not None
    assert PriceValueMetrics is not None
    assert ClaimsMetrics is not None
