"""Tests for OutcomeResult and OutcomeSuite."""
import pytest
from insurance_governance.outcome.results import OutcomeResult, OutcomeSuite, _compute_outcome_rag
from insurance_governance.validation.results import RAGStatus, Severity


def make_result(
    outcome="price_value",
    test_name="test",
    passed=True,
    severity=Severity.INFO,
    metric_value=None,
    threshold=None,
    segment=None,
    period="2025-Q4",
):
    return OutcomeResult(
        outcome=outcome,
        test_name=test_name,
        passed=passed,
        severity=severity,
        metric_value=metric_value,
        threshold=threshold,
        segment=segment,
        period=period,
        details="test details",
    )


# --- OutcomeResult construction ---

def test_outcome_result_valid_outcomes():
    for outcome in ("price_value", "claims", "support"):
        r = make_result(outcome=outcome)
        assert r.outcome == outcome


def test_outcome_result_invalid_outcome():
    with pytest.raises(ValueError, match="outcome must be one of"):
        OutcomeResult(outcome="invalid", test_name="x", passed=True)


def test_outcome_result_defaults():
    r = OutcomeResult(outcome="claims", test_name="timeliness", passed=True)
    assert r.period == ""
    assert r.segment is None
    assert r.details == ""
    assert r.severity == Severity.INFO
    assert r.corrective_actions == []
    assert r.extra == {}
    assert r.metric_value is None
    assert r.threshold is None


def test_outcome_result_to_dict_structure():
    r = make_result(
        outcome="price_value",
        test_name="fair_value_ratio",
        passed=False,
        severity=Severity.CRITICAL,
        metric_value=0.65,
        threshold=0.70,
        segment="Renewal",
    )
    d = r.to_dict()
    assert d["outcome"] == "price_value"
    assert d["test_name"] == "fair_value_ratio"
    assert d["passed"] is False
    assert d["metric_value"] == 0.65
    assert d["threshold"] == 0.70
    assert d["segment"] == "Renewal"
    assert d["severity"] == "critical"


def test_outcome_result_to_dict_severity_is_string():
    r = make_result(severity=Severity.WARNING)
    assert r.to_dict()["severity"] == "warning"


def test_outcome_result_corrective_actions_in_dict():
    r = OutcomeResult(
        outcome="claims",
        test_name="sla",
        passed=False,
        corrective_actions=["Fix process A", "Escalate to B"],
    )
    d = r.to_dict()
    assert d["corrective_actions"] == ["Fix process A", "Escalate to B"]


def test_outcome_result_extra_in_dict():
    r = OutcomeResult(
        outcome="support",
        test_name="complaints",
        passed=True,
        extra={"count": 42},
    )
    assert r.to_dict()["extra"]["count"] == 42


# --- _compute_outcome_rag ---

def test_rag_green_all_pass():
    results = [make_result(passed=True), make_result(passed=True)]
    assert _compute_outcome_rag(results) == RAGStatus.GREEN


def test_rag_green_info_failure():
    results = [make_result(passed=False, severity=Severity.INFO)]
    assert _compute_outcome_rag(results) == RAGStatus.GREEN


def test_rag_amber_warning_failure():
    results = [
        make_result(passed=True),
        make_result(passed=False, severity=Severity.WARNING),
    ]
    assert _compute_outcome_rag(results) == RAGStatus.AMBER


def test_rag_red_critical_failure():
    results = [make_result(passed=False, severity=Severity.CRITICAL)]
    assert _compute_outcome_rag(results) == RAGStatus.RED


def test_rag_red_overrides_amber():
    results = [
        make_result(passed=False, severity=Severity.WARNING),
        make_result(passed=False, severity=Severity.CRITICAL),
    ]
    assert _compute_outcome_rag(results) == RAGStatus.RED


def test_rag_empty_is_green():
    assert _compute_outcome_rag([]) == RAGStatus.GREEN


# --- OutcomeSuite ---

def test_suite_passed_failed():
    results = [
        make_result(passed=True),
        make_result(passed=False, severity=Severity.WARNING),
    ]
    suite = OutcomeSuite(results=results, period="2025-Q4")
    assert len(suite.passed) == 1
    assert len(suite.failed) == 1


def test_suite_critical_failures():
    results = [
        make_result(passed=False, severity=Severity.CRITICAL),
        make_result(passed=False, severity=Severity.WARNING),
    ]
    suite = OutcomeSuite(results=results)
    assert len(suite.critical_failures) == 1


def test_suite_warning_failures():
    results = [
        make_result(passed=False, severity=Severity.CRITICAL),
        make_result(passed=False, severity=Severity.WARNING),
        make_result(passed=False, severity=Severity.WARNING),
    ]
    suite = OutcomeSuite(results=results)
    assert len(suite.warning_failures) == 2


def test_suite_rag_status():
    results = [make_result(passed=False, severity=Severity.CRITICAL)]
    suite = OutcomeSuite(results=results)
    assert suite.rag_status == RAGStatus.RED


def test_suite_by_outcome():
    results = [
        make_result(outcome="price_value"),
        make_result(outcome="claims"),
        make_result(outcome="price_value"),
    ]
    suite = OutcomeSuite(results=results)
    assert len(suite.by_outcome("price_value")) == 2
    assert len(suite.by_outcome("claims")) == 1
    assert len(suite.by_outcome("support")) == 0


def test_suite_vulnerable_segment_results():
    results = [
        make_result(segment="Renewal"),
        make_result(segment=None),
        make_result(segment="Older Customers"),
    ]
    suite = OutcomeSuite(results=results)
    segged = suite.vulnerable_segment_results()
    assert len(segged) == 2


def test_suite_summary_keys():
    suite = OutcomeSuite(results=[make_result()], period="2025-Q4")
    s = suite.summary()
    assert "total" in s
    assert "passed" in s
    assert "failed" in s
    assert "rag_status" in s
    assert s["period"] == "2025-Q4"


def test_suite_to_dict():
    suite = OutcomeSuite(results=[make_result()], period="2025-Q4")
    d = suite.to_dict()
    assert "results" in d
    assert len(d["results"]) == 1
