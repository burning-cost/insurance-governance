"""Tests for PriceValueMetrics and ClaimsMetrics."""
import numpy as np
import pytest
from insurance_governance.outcome.metrics import ClaimsMetrics, PriceValueMetrics
from insurance_governance.validation.results import Severity


# ===================================================================
# PriceValueMetrics.fair_value_ratio
# ===================================================================

def test_fair_value_ratio_pass():
    premiums = [500.0] * 100
    claims = [380.0] * 100
    expenses = [50.0] * 100
    result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, period="2025-Q4")
    assert result.passed is True
    assert abs(result.metric_value - 0.76) < 1e-4


def test_fair_value_ratio_fail():
    premiums = [500.0] * 100
    claims = [300.0] * 100
    expenses = [50.0] * 100
    result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, period="2025-Q4")
    assert result.passed is False
    assert result.metric_value < 0.70
    assert result.severity == Severity.CRITICAL


def test_fair_value_ratio_exactly_at_threshold():
    premiums = [100.0] * 10
    claims = [70.0] * 10
    expenses = [0.0] * 10
    result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, period="2025-Q4", threshold=0.70)
    assert result.passed is True
    assert abs(result.metric_value - 0.70) < 1e-6


def test_fair_value_ratio_custom_threshold():
    premiums = [100.0] * 10
    claims = [60.0] * 10
    expenses = [0.0] * 10
    result = PriceValueMetrics.fair_value_ratio(
        premiums, claims, expenses, period="2025-Q4", threshold=0.55
    )
    assert result.passed is True


def test_fair_value_ratio_zero_premium():
    result = PriceValueMetrics.fair_value_ratio([0.0], [100.0], [10.0], period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.metric_value is None


def test_fair_value_ratio_includes_expense_ratio_in_extra():
    premiums = [200.0] * 5
    claims = [150.0] * 5
    expenses = [40.0] * 5
    result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, period="2025-Q4")
    assert "expense_ratio" in result.extra
    assert abs(result.extra["expense_ratio"] - 0.20) < 1e-6


def test_fair_value_ratio_segment_label():
    result = PriceValueMetrics.fair_value_ratio(
        [100.0] * 5, [75.0] * 5, [0.0] * 5, period="2025-Q4", segment="Renewal"
    )
    assert result.segment == "Renewal"


def test_fair_value_ratio_outcome_field():
    result = PriceValueMetrics.fair_value_ratio([100.0], [80.0], [0.0], period="2025-Q4")
    assert result.outcome == "price_value"


def test_fair_value_ratio_corrective_actions_on_fail():
    result = PriceValueMetrics.fair_value_ratio([100.0], [50.0], [0.0], period="2025-Q4")
    assert len(result.corrective_actions) > 0


def test_fair_value_ratio_no_corrective_actions_on_pass():
    result = PriceValueMetrics.fair_value_ratio([100.0], [80.0], [0.0], period="2025-Q4")
    assert result.corrective_actions == []


# ===================================================================
# PriceValueMetrics.price_dispersion_by_segment
# ===================================================================

def test_price_dispersion_single_segment():
    results = PriceValueMetrics.price_dispersion_by_segment(
        premiums=[100.0, 200.0, 150.0],
        segment_labels=["A", "A", "A"],
        period="2025-Q4",
    )
    assert len(results) == 1
    assert results[0].passed is True


def test_price_dispersion_within_threshold():
    premiums = [100.0, 110.0, 105.0, 140.0, 135.0, 130.0]
    labels = ["A", "A", "A", "B", "B", "B"]
    results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "price_dispersion_summary"][0]
    assert summary.passed is True
    assert summary.metric_value < 1.50


def test_price_dispersion_exceeds_threshold():
    premiums = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    labels = ["Cheap", "Cheap", "Cheap", "Expensive", "Expensive", "Expensive"]
    results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "price_dispersion_summary"][0]
    assert summary.passed is False
    assert abs(summary.metric_value - 2.0) < 1e-6


def test_price_dispersion_per_segment_results():
    premiums = [100.0, 150.0, 200.0]
    labels = ["A", "B", "C"]
    results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
    per_seg = [r for r in results if r.segment is not None]
    assert len(per_seg) == 3


def test_price_dispersion_zero_min_median():
    premiums = [0.0, 0.0, 200.0, 200.0]
    labels = ["A", "A", "B", "B"]
    results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
    # Should return an error result when min median is zero
    assert any(not r.passed for r in results)


def test_price_dispersion_summary_has_segment_medians_in_extra():
    premiums = [100.0, 200.0]
    labels = ["A", "B"]
    results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "price_dispersion_summary"][0]
    assert "segment_medians" in summary.extra


# ===================================================================
# PriceValueMetrics.renewal_vs_new_business_gap
# ===================================================================

def test_renewal_gap_pass():
    renewal = [300.0] * 50
    nb = [295.0] * 50
    exposure = [1.0] * 50
    result = PriceValueMetrics.renewal_vs_new_business_gap(renewal, nb, exposure, period="2025-Q4")
    assert result.passed is True


def test_renewal_gap_fail():
    renewal = [350.0] * 50
    nb = [250.0] * 50
    exposure = [1.0] * 50
    result = PriceValueMetrics.renewal_vs_new_business_gap(
        renewal, nb, exposure, period="2025-Q4", threshold_pct=5.0
    )
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_renewal_gap_metric_is_percentage():
    renewal = [110.0]
    nb = [100.0]
    exposure = [1.0]
    result = PriceValueMetrics.renewal_vs_new_business_gap(renewal, nb, exposure, period="2025-Q4")
    assert abs(result.metric_value - 10.0) < 1e-4


def test_renewal_gap_negative_gap_is_fine():
    """Renewals cheaper than NB is fine."""
    renewal = [90.0] * 10
    nb = [100.0] * 10
    exposure = [1.0] * 10
    result = PriceValueMetrics.renewal_vs_new_business_gap(renewal, nb, exposure, period="2025-Q4")
    assert result.passed is True
    assert result.metric_value < 0


def test_renewal_gap_empty_arrays():
    result = PriceValueMetrics.renewal_vs_new_business_gap([], [], [], period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_renewal_gap_outcome_field():
    result = PriceValueMetrics.renewal_vs_new_business_gap(
        [300.0], [300.0], [1.0], period="2025-Q4"
    )
    assert result.outcome == "price_value"


# ===================================================================
# ClaimsMetrics.settlement_value_adequacy
# ===================================================================

def test_settlement_adequacy_pass():
    agreed = [4800.0] * 20
    reference = [5000.0] * 20
    result = ClaimsMetrics.settlement_value_adequacy(agreed, reference, period="2025-Q4")
    assert result.passed is True
    assert abs(result.metric_value - 0.96) < 1e-6


def test_settlement_adequacy_fail():
    agreed = [4000.0] * 20
    reference = [5000.0] * 20
    result = ClaimsMetrics.settlement_value_adequacy(agreed, reference, period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_settlement_adequacy_empty():
    result = ClaimsMetrics.settlement_value_adequacy([], [], period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_settlement_adequacy_mismatched_lengths():
    result = ClaimsMetrics.settlement_value_adequacy(
        [1000.0, 2000.0], [1000.0], period="2025-Q4"
    )
    assert result.passed is False
    assert "mismatch" in result.details


def test_settlement_adequacy_zero_reference():
    result = ClaimsMetrics.settlement_value_adequacy([1000.0], [0.0], period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_settlement_adequacy_extra_fields():
    agreed = [4750.0] * 10
    reference = [5000.0] * 10
    result = ClaimsMetrics.settlement_value_adequacy(agreed, reference, period="2025-Q4")
    assert "mean_agreed_settlement" in result.extra
    assert "mean_reference_valuation" in result.extra
    assert result.extra["claim_count"] == 10


def test_settlement_adequacy_outcome_field():
    result = ClaimsMetrics.settlement_value_adequacy([1000.0], [1000.0], period="2025-Q4")
    assert result.outcome == "claims"


# ===================================================================
# ClaimsMetrics.decline_rate_by_segment
# ===================================================================

def test_decline_rate_single_segment():
    results = ClaimsMetrics.decline_rate_by_segment(
        outcomes=[0, 1, 0, 1],
        segment_labels=["A", "A", "A", "A"],
        period="2025-Q4",
    )
    assert len(results) == 1
    assert results[0].passed is True


def test_decline_rate_within_threshold():
    outcomes = [0, 0, 1, 0, 0, 0, 1, 0]
    labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
    results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "decline_rate_disparity_summary"][0]
    assert summary.passed is True


def test_decline_rate_exceeds_threshold():
    # Seg A: 50% decline, Seg B: 10% decline — ratio = 5.0
    outcomes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    labels = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "decline_rate_disparity_summary"][0]
    assert summary.passed is False


def test_decline_rate_per_segment_results():
    outcomes = [0, 1, 0, 1, 0]
    labels = ["A", "A", "B", "B", "C"]
    results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
    per_seg = [r for r in results if r.segment is not None]
    assert len(per_seg) == 3


def test_decline_rate_all_zero():
    """All segments have zero decline rate — no disparity."""
    outcomes = [0, 0, 0, 0]
    labels = ["A", "A", "B", "B"]
    results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
    summary = [r for r in results if r.test_name == "decline_rate_disparity_summary"][0]
    assert summary.passed is True


def test_decline_rate_outcome_field():
    results = ClaimsMetrics.decline_rate_by_segment(
        [0, 1, 0, 1], ["A", "A", "B", "B"], "2025-Q4"
    )
    for r in results:
        assert r.outcome == "claims"


# ===================================================================
# ClaimsMetrics.timeliness_sla
# ===================================================================

def test_timeliness_sla_all_pass():
    days = [2.0, 3.0, 1.0, 4.0, 5.0] * 20
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert result.passed is True
    assert result.metric_value == 1.0


def test_timeliness_sla_fail():
    # Only 50% within SLA
    days = [1.0, 1.0, 10.0, 10.0] * 20
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert result.passed is False


def test_timeliness_sla_empty():
    result = ClaimsMetrics.timeliness_sla([], period="2025-Q4")
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_timeliness_sla_metric_is_proportion():
    days = [3.0, 3.0, 3.0, 10.0]
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert abs(result.metric_value - 0.75) < 1e-6


def test_timeliness_sla_extra_stats():
    days = [2.0, 4.0, 6.0, 8.0]
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert "mean_days" in result.extra
    assert "median_days" in result.extra
    assert "p90_days" in result.extra
    assert result.extra["sla_days"] == 5
    assert result.extra["claim_count"] == 4


def test_timeliness_sla_outcome_field():
    result = ClaimsMetrics.timeliness_sla([3.0, 4.0], period="2025-Q4")
    assert result.outcome == "claims"


def test_timeliness_sla_warning_severity_on_fail():
    days = [20.0] * 10
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert result.severity == Severity.WARNING


def test_timeliness_sla_corrective_actions_on_fail():
    days = [20.0] * 10
    result = ClaimsMetrics.timeliness_sla(days, period="2025-Q4", sla_days=5)
    assert len(result.corrective_actions) > 0
