"""Expanded test coverage for v0.2.0 and v0.3.0 additions.

Targets:
- outcome/metrics.py   — PriceValueMetrics and ClaimsMetrics edge cases
- outcome/framework.py — column combinations not previously covered
- outcome/results.py   — OutcomeSuite to_dict and summary detail
- outcome/segments.py  — SegmentComparison boundary and edge cases
- outcome/report.py    — OutcomeTestingReport edge cases
- audit/log.py         — ExplainabilityAuditLog edge cases
- audit/customer_explanation.py — PlainLanguageExplainer edge cases
- audit/report.py      — AuditSummaryReport edge cases and empty log

These are deliberately standalone from the existing test files so they can
be reviewed and run independently.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_mrm_card(**kwargs):
    from insurance_governance.mrm.model_card import ModelCard as MRMModelCard
    defaults = dict(
        model_id="motor-freq-v3",
        model_name="Motor Frequency v3",
        version="3.0.0",
        model_class="pricing",
    )
    defaults.update(kwargs)
    return MRMModelCard(**defaults)


def _make_audit_entry(**kwargs):
    from insurance_governance.audit.entry import ExplainabilityAuditEntry
    defaults = dict(
        model_id="motor-freq-v3",
        model_version="3.1.0",
        input_features={"driver_age": 32, "ncb_years": 5, "region": "SE"},
        feature_importances={"driver_age": -0.12, "ncb_years": -0.31, "region": 0.08},
        prediction=412.50,
        final_premium=412.50,
        decision_basis="model_output",
    )
    defaults.update(kwargs)
    return ExplainabilityAuditEntry(**defaults)


FEATURE_LABELS = {
    "driver_age": "your age",
    "ncb_years": "your no-claims discount",
    "region": "your postcode area",
}


# ===========================================================================
# outcome/metrics.py — PriceValueMetrics edge cases
# ===========================================================================

class TestFairValueRatioEdgeCases:
    """Additional edge cases not covered by test_outcome_metrics.py."""

    def test_negative_premium_treated_as_zero(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        result = PriceValueMetrics.fair_value_ratio(
            [-100.0], [50.0], [0.0], period="2025-Q4"
        )
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.metric_value is None

    def test_very_high_claims_ratio(self):
        """Claims exceed premium — loss ratio > 1 is valid data."""
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [100.0] * 10
        claims = [200.0] * 10  # 200% loss ratio
        expenses = [0.0] * 10
        result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, "2025-Q4")
        assert result.passed is True  # 2.0 > 0.70 threshold
        assert result.metric_value == pytest.approx(2.0, abs=1e-6)

    def test_threshold_just_below_zero_raises_no_error(self):
        """Custom threshold of 0.0 — ratio >= 0.0 should always pass."""
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.fair_value_ratio(
            [100.0], [0.0], [0.0], period="2025-Q4", threshold=0.0
        )
        assert result.passed is True

    def test_extra_contains_total_premium_and_claims(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.fair_value_ratio(
            [500.0] * 4, [350.0] * 4, [20.0] * 4, "2025-Q4"
        )
        assert result.extra["total_premium"] == pytest.approx(2000.0)
        assert result.extra["total_claims"] == pytest.approx(1400.0)
        assert result.extra["total_expenses"] == pytest.approx(80.0)

    def test_segment_label_in_extra_not_added(self):
        """Segment should be on the result, not in extra."""
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.fair_value_ratio(
            [100.0], [75.0], [5.0], "2025-Q4", segment="DirectSales"
        )
        assert result.segment == "DirectSales"
        assert "segment" not in result.extra

    def test_single_policy_pass(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.fair_value_ratio([800.0], [640.0], [0.0], "2025-H1")
        assert result.passed is True
        assert result.metric_value == pytest.approx(0.80, abs=1e-6)

    def test_details_mentions_expense_ratio(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.fair_value_ratio(
            [100.0] * 5, [80.0] * 5, [10.0] * 5, "2025-Q4"
        )
        assert "Expense ratio" in result.details


class TestPriceDispersionEdgeCases:
    def test_three_segments_ratio_computed_correctly(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        # Medians: A=100, B=150, C=300 → ratio = 3.0
        premiums = [100.0, 100.0, 150.0, 150.0, 300.0, 300.0]
        labels = ["A", "A", "B", "B", "C", "C"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
        summary = next(r for r in results if r.test_name == "price_dispersion_summary")
        assert summary.passed is False
        assert summary.metric_value == pytest.approx(3.0, abs=1e-4)

    def test_custom_max_ratio_respected(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [100.0, 100.0, 200.0, 200.0]
        labels = ["A", "A", "B", "B"]
        # Default 1.5 fails, but max_ratio=2.5 should pass
        results = PriceValueMetrics.price_dispersion_by_segment(
            premiums, labels, "2025-Q4", max_ratio=2.5
        )
        summary = next(r for r in results if r.test_name == "price_dispersion_summary")
        assert summary.passed is True

    def test_per_segment_results_are_informational(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        premiums = [100.0, 200.0]
        labels = ["A", "B"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
        per_seg = [r for r in results if r.segment is not None]
        for r in per_seg:
            assert r.severity == Severity.INFO
            assert r.passed is True  # per-segment are always informational

    def test_segment_count_in_extra(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [100.0, 110.0, 120.0, 200.0]
        labels = ["A", "A", "A", "B"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
        per_seg = [r for r in results if r.segment == "A"]
        assert per_seg[0].extra["segment_count"] == 3

    def test_corrective_actions_populated_on_failure(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [100.0, 100.0, 500.0, 500.0]
        labels = ["Cheap", "Cheap", "Expensive", "Expensive"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, labels, "2025-Q4")
        summary = next(r for r in results if r.test_name == "price_dispersion_summary")
        assert len(summary.corrective_actions) > 0


class TestRenewalGapEdgeCases:
    def test_exact_threshold_is_pass(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        # NB = 100, renewal = 105 → gap = 5.0%, threshold = 5.0 → pass
        renewal = [105.0]
        nb = [100.0]
        exposure = [1.0]
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            renewal, nb, exposure, "2025-Q4", threshold_pct=5.0
        )
        assert result.passed is True
        assert result.metric_value == pytest.approx(5.0, abs=1e-4)

    def test_weighted_exposure_affects_result(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        # High-exposure policies are £300 renewal; low-exposure are £500
        # With weights: mean renewal should be pulled towards £300
        renewal = [300.0, 500.0]
        nb = [300.0, 300.0]
        exposure = [10.0, 1.0]  # heavily weighted to £300
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            renewal, nb, exposure, "2025-Q4"
        )
        # Weighted renewal ~ (300*10 + 500*1)/11 ≈ 318.2, NB mean = 300
        # Gap ≈ 6.1% → fails default 5%
        assert result.metric_value == pytest.approx((318.18 - 300) / 300 * 100, abs=0.5)

    def test_zero_exposure_returns_error(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [300.0], [300.0], [0.0], "2025-Q4"
        )
        assert result.passed is False
        assert result.severity == Severity.CRITICAL

    def test_zero_nb_mean_returns_error(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [300.0], [0.0], [1.0], "2025-Q4"
        )
        assert result.passed is False
        assert result.severity == Severity.CRITICAL

    def test_extra_has_weighted_renewal_and_nb_mean(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [300.0, 320.0], [290.0, 310.0], [1.0, 1.0], "2025-Q4"
        )
        assert "weighted_renewal_premium" in result.extra
        assert "mean_new_business_premium" in result.extra

    def test_corrective_actions_on_failure(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [400.0], [200.0], [1.0], "2025-Q4", threshold_pct=5.0
        )
        assert result.passed is False
        assert len(result.corrective_actions) > 0
        assert any("PS21/5" in a or "GIPP" in a for a in result.corrective_actions)


class TestClaimsMetricsEdgeCases:
    def test_settlement_adequacy_custom_threshold(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        agreed = [900.0] * 10
        reference = [1000.0] * 10
        # Ratio = 0.90; default threshold is 0.95 → fail, but custom 0.85 → pass
        result = ClaimsMetrics.settlement_value_adequacy(
            agreed, reference, "2025-Q4", threshold_ratio=0.85
        )
        assert result.passed is True

    def test_settlement_adequacy_corrective_actions_on_failure(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        result = ClaimsMetrics.settlement_value_adequacy(
            [500.0], [1000.0], "2025-Q4"
        )
        assert result.passed is False
        assert len(result.corrective_actions) > 0

    def test_decline_rate_corrective_actions_on_failure(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        # Seg A 100% decline, Seg B 10% decline → ratio = 10 → fail
        outcomes = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        labels = ["A"] * 5 + ["B"] * 5
        results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
        summary = next(r for r in results if r.test_name == "decline_rate_disparity_summary")
        assert summary.passed is False
        assert len(summary.corrective_actions) > 0

    def test_decline_rate_extra_has_segment_rates(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        outcomes = [1, 0, 1, 0]
        labels = ["A", "A", "B", "B"]
        results = ClaimsMetrics.decline_rate_by_segment(outcomes, labels, "2025-Q4")
        summary = next(r for r in results if r.test_name == "decline_rate_disparity_summary")
        assert "segment_decline_rates" in summary.extra
        assert "A" in summary.extra["segment_decline_rates"]
        assert "B" in summary.extra["segment_decline_rates"]

    def test_timeliness_sla_custom_sla_days(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        # All claims settled in 20 days — fails SLA of 5, but passes SLA of 25
        days = [20.0] * 10
        result_strict = ClaimsMetrics.timeliness_sla(days, "2025-Q4", sla_days=5)
        result_loose = ClaimsMetrics.timeliness_sla(days, "2025-Q4", sla_days=25)
        assert result_strict.passed is False
        assert result_loose.passed is True

    def test_timeliness_sla_exactly_80_pct_compliance(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        # 8 within SLA, 2 outside → 80% compliance = exactly at floor
        days = [3.0] * 8 + [10.0] * 2
        result = ClaimsMetrics.timeliness_sla(days, "2025-Q4", sla_days=5)
        assert result.passed is True
        assert result.metric_value == pytest.approx(0.80, abs=1e-6)

    def test_timeliness_sla_within_sla_count_in_extra(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        days = [2.0, 4.0, 6.0, 8.0]
        result = ClaimsMetrics.timeliness_sla(days, "2025-Q4", sla_days=5)
        assert result.extra["within_sla_count"] == 2


# ===========================================================================
# outcome/framework.py — additional column combinations
# ===========================================================================

class TestOutcomeFrameworkCoverage:
    def _make_df(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        return pl.DataFrame({
            "gross_premium": rng.uniform(200, 500, n).tolist(),
            "claims_paid": rng.uniform(100, 400, n).tolist(),
            "reference_val": rng.uniform(110, 450, n).tolist(),
            "is_renewal": (rng.random(n) > 0.5).astype(int).tolist(),
            "days_to_settlement": rng.uniform(1, 15, n).tolist(),
            "claim_outcome": (rng.random(n) > 0.85).astype(int).tolist(),
            "expenses": rng.uniform(10, 50, n).tolist(),
            "exposure": rng.uniform(0.5, 1.0, n).tolist(),
            "age": rng.integers(20, 80, n).tolist(),
            "policy_type": ["renewal" if i % 2 == 0 else "new" for i in range(n)],
        })

    def test_reference_valuation_col_triggers_settlement_test(self):
        from insurance_governance.outcome import OutcomeTestingFramework
        card = _make_mrm_card()
        df = self._make_df()
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            claim_amount_col="claims_paid",
            reference_valuation_col="reference_val",
        )
        results = fw.run()
        test_names = [r.test_name for r in results]
        assert "settlement_value_adequacy" in test_names

    def test_exposure_col_used_for_renewal_gap(self):
        from insurance_governance.outcome import OutcomeTestingFramework
        card = _make_mrm_card()
        df = self._make_df()
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            renewal_indicator_col="is_renewal",
            exposure_col="exposure",
        )
        results = fw.run()
        test_names = [r.test_name for r in results]
        assert "renewal_vs_new_business_gap" in test_names

    def test_decline_rate_test_runs_with_segment_and_claim_outcome(self):
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        card = _make_mrm_card()
        df = self._make_df()
        seg = CustomerSegment(
            name="Renewal",
            filter_fn=lambda df: df["policy_type"] == "renewal",
        )
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            claim_outcome_col="claim_outcome",
            customer_segments=[seg],
        )
        results = fw.run()
        test_names = [r.test_name for r in results]
        assert "decline_rate_by_segment" in test_names or \
               "decline_rate_disparity_summary" in test_names

    def test_missing_price_col_returns_no_price_tests(self):
        from insurance_governance.outcome import OutcomeTestingFramework
        card = _make_mrm_card()
        df = self._make_df()
        # Use a col name that doesn't exist
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="nonexistent_col",
        )
        results = fw.run()
        price_tests = [r for r in results if r.outcome == "price_value"]
        assert len(price_tests) == 0

    def test_segment_with_no_matching_rows_excluded(self):
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        card = _make_mrm_card()
        df = self._make_df()
        empty_seg = CustomerSegment(
            name="Phantom",
            filter_fn=lambda df: pl.Series([False] * df.height),
        )
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            customer_segments=[empty_seg],
        )
        results = fw.run()
        assert not any(r.segment == "Phantom" for r in results)

    def test_timeliness_test_runs_per_segment(self):
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        card = _make_mrm_card()
        df = self._make_df()
        renewal_seg = CustomerSegment(
            name="Renewal",
            filter_fn=lambda df: df["policy_type"] == "renewal",
        )
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            days_to_settlement_col="days_to_settlement",
            customer_segments=[renewal_seg],
        )
        results = fw.run()
        renewal_timeliness = [
            r for r in results
            if r.test_name == "timeliness_sla" and r.segment == "Renewal"
        ]
        assert len(renewal_timeliness) == 1

    def test_get_rag_status_green_on_all_pass(self):
        from insurance_governance.outcome import OutcomeTestingFramework
        from insurance_governance.validation.results import RAGStatus
        card = _make_mrm_card()
        # Build a df where the fair value ratio will clearly pass
        df = pl.DataFrame({
            "gross_premium": [500.0] * 50,
            "claims_paid": [400.0] * 50,
        })
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            claim_amount_col="claims_paid",
        )
        rag = fw.get_rag_status()
        assert rag == RAGStatus.GREEN

    def test_to_dict_summary_total_tests_positive(self):
        from insurance_governance.outcome import OutcomeTestingFramework
        card = _make_mrm_card()
        df = self._make_df()
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            claim_amount_col="claims_paid",
            days_to_settlement_col="days_to_settlement",
        )
        d = fw.to_dict()
        assert d["summary"]["total_tests"] > 0

    def test_dispersion_check_runs_with_two_segments(self):
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        card = _make_mrm_card()
        df = self._make_df()
        seg_a = CustomerSegment(
            name="Young",
            filter_fn=lambda df: df["age"] < 40,
        )
        seg_b = CustomerSegment(
            name="Older",
            filter_fn=lambda df: df["age"] >= 40,
        )
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            customer_segments=[seg_a, seg_b],
        )
        results = fw.run()
        dispersion = [r for r in results if r.test_name == "price_dispersion_summary"]
        assert len(dispersion) == 1

    def test_other_label_assigned_to_unmatched_rows(self):
        """Rows not matched by any segment are labelled 'Other' in segment labels."""
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        card = _make_mrm_card()
        df = pl.DataFrame({
            "gross_premium": [300.0] * 10,
            "policy_type": ["new"] * 5 + ["renewal"] * 5,
        })
        renewal_seg = CustomerSegment(
            name="Renewal",
            filter_fn=lambda df: df["policy_type"] == "renewal",
        )
        fw = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            customer_segments=[renewal_seg],
        )
        # Calling _build_segment_labels should produce "Renewal" and "Other"
        labels = fw._build_segment_labels(df)
        assert labels is not None
        assert "Other" in labels
        assert "Renewal" in labels


# ===========================================================================
# outcome/results.py — OutcomeSuite edge cases
# ===========================================================================

class TestOutcomeSuiteEdgeCases:
    def _make_result(self, **kwargs):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import Severity
        defaults = dict(
            outcome="price_value",
            test_name="test",
            passed=True,
            severity=Severity.INFO,
            period="2025-Q4",
        )
        defaults.update(kwargs)
        return OutcomeResult(**defaults)

    def test_empty_suite_summary(self):
        from insurance_governance.outcome.results import OutcomeSuite
        from insurance_governance.validation.results import RAGStatus
        suite = OutcomeSuite(results=[], period="2025-Q4")
        s = suite.summary()
        assert s["total"] == 0
        assert s["passed"] == 0
        assert s["failed"] == 0
        assert s["critical"] == 0
        assert s["warnings"] == 0
        assert s["rag_status"] == RAGStatus.GREEN.value

    def test_suite_summary_critical_count(self):
        from insurance_governance.outcome.results import OutcomeSuite
        from insurance_governance.validation.results import Severity
        results = [
            self._make_result(passed=False, severity=Severity.CRITICAL),
            self._make_result(passed=False, severity=Severity.CRITICAL),
            self._make_result(passed=True),
        ]
        suite = OutcomeSuite(results=results)
        s = suite.summary()
        assert s["critical"] == 2
        assert s["warnings"] == 0

    def test_suite_summary_warning_count(self):
        from insurance_governance.outcome.results import OutcomeSuite
        from insurance_governance.validation.results import Severity
        results = [
            self._make_result(passed=False, severity=Severity.WARNING),
            self._make_result(passed=True),
        ]
        suite = OutcomeSuite(results=results)
        s = suite.summary()
        assert s["warnings"] == 1
        assert s["critical"] == 0

    def test_suite_to_dict_has_summary_and_results(self):
        from insurance_governance.outcome.results import OutcomeSuite
        suite = OutcomeSuite(results=[self._make_result()], period="2025-Q4")
        d = suite.to_dict()
        assert "summary" in d
        assert "results" in d
        assert "period" in d
        assert d["period"] == "2025-Q4"
        assert len(d["results"]) == 1

    def test_by_outcome_unknown_returns_empty(self):
        from insurance_governance.outcome.results import OutcomeSuite
        suite = OutcomeSuite(results=[self._make_result()])
        assert suite.by_outcome("support") == []

    def test_vulnerable_segment_results_filters_correctly(self):
        from insurance_governance.outcome.results import OutcomeSuite
        results = [
            self._make_result(segment="Renewal"),
            self._make_result(segment="Vulnerable"),
            self._make_result(segment=None),
        ]
        suite = OutcomeSuite(results=results)
        seg_results = suite.vulnerable_segment_results()
        assert len(seg_results) == 2
        assert all(r.segment is not None for r in seg_results)


# ===========================================================================
# outcome/segments.py — CustomerSegment and SegmentComparison edge cases
# ===========================================================================

class TestCustomerSegmentEdgeCases:
    def test_segment_name_preserved(self):
        from insurance_governance.outcome.segments import CustomerSegment
        seg = CustomerSegment(name="Over 65s", filter_fn=lambda df: df["age"] >= 65)
        assert seg.name == "Over 65s"

    def test_filter_fn_receives_full_dataframe(self):
        from insurance_governance.outcome.segments import CustomerSegment
        received = []
        def capturing_filter(df):
            received.append(df)
            return pl.Series([True] * df.height)
        seg = CustomerSegment(name="All", filter_fn=capturing_filter)
        df = pl.DataFrame({"age": [30, 40]})
        seg.apply(df)
        assert len(received) == 1
        assert received[0].height == 2

    def test_count_returns_integer(self):
        from insurance_governance.outcome.segments import CustomerSegment
        df = pl.DataFrame({"age": [20, 30, 40, 50, 60]})
        seg = CustomerSegment(name="Under40", filter_fn=lambda df: df["age"] < 40)
        assert isinstance(seg.count(df), int)
        assert seg.count(df) == 2

    def test_count_zero_for_empty_result(self):
        from insurance_governance.outcome.segments import CustomerSegment
        df = pl.DataFrame({"age": [20, 30]})
        seg = CustomerSegment(name="Senior", filter_fn=lambda df: df["age"] > 100)
        assert seg.count(df) == 0


class TestSegmentComparisonEdgeCases:
    def test_ratio_exactly_at_threshold_is_pass(self):
        from insurance_governance.outcome.segments import SegmentComparison
        comp = SegmentComparison(
            segment_a="A", segment_b="B",
            metric_name="median_premium",
            value_a=150.0, value_b=100.0,
            ratio=1.50, threshold=1.50,
            passed=True,
        )
        assert comp.passed is True
        assert comp.ratio == 1.50

    def test_to_dict_all_fields_present(self):
        from insurance_governance.outcome.segments import SegmentComparison
        comp = SegmentComparison(
            segment_a="Renewal", segment_b="New",
            metric_name="claims_ratio",
            value_a=0.75, value_b=0.70,
            ratio=1.07, threshold=1.50, passed=True,
        )
        d = comp.to_dict()
        required = {"segment_a", "segment_b", "metric_name", "value_a", "value_b",
                    "ratio", "threshold", "passed"}
        assert required == set(d.keys())

    def test_to_dict_values_correct(self):
        from insurance_governance.outcome.segments import SegmentComparison
        comp = SegmentComparison(
            segment_a="Direct", segment_b="Aggregator",
            metric_name="decline_rate",
            value_a=0.20, value_b=0.10,
            ratio=2.0, threshold=1.50, passed=False,
        )
        d = comp.to_dict()
        assert d["passed"] is False
        assert d["ratio"] == 2.0
        assert d["value_a"] == 0.20


# ===========================================================================
# outcome/report.py — OutcomeTestingReport edge cases
# ===========================================================================

class TestOutcomeTestingReportEdgeCases:
    def _make_report(self, results=None, period="2025-Q4", **kwargs):
        from insurance_governance.outcome.report import OutcomeTestingReport
        if results is None:
            results = []
        return OutcomeTestingReport(_make_mrm_card(), results, period=period, **kwargs)

    def test_empty_results_renders_html_without_error(self):
        report = self._make_report(results=[])
        html = report.render_html()
        assert isinstance(html, str)
        assert len(html) > 100

    def test_custom_generated_date_appears_in_dict(self):
        d = date(2025, 6, 1)
        report = self._make_report(generated_date=d)
        data = report.to_dict()
        assert "2025-06-01" in data["generated_date"]

    def test_run_id_preserved_in_dict(self):
        report = self._make_report(run_id="my-test-run")
        assert report.to_dict()["run_id"] == "my-test-run"

    def test_rag_status_green_when_all_pass(self):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import RAGStatus
        results = [
            OutcomeResult(outcome="price_value", test_name="fvr", passed=True, period="2025-Q4"),
        ]
        report = self._make_report(results=results)
        assert report.rag_status == RAGStatus.GREEN

    def test_rag_status_red_in_dict(self):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import RAGStatus, Severity
        results = [
            OutcomeResult(
                outcome="claims", test_name="fail", passed=False,
                severity=Severity.CRITICAL, period="2025-Q4",
            )
        ]
        report = self._make_report(results=results)
        d = report.to_dict()
        assert d["rag_status"] == RAGStatus.RED.value

    def test_write_html_html_is_valid_structure(self, tmp_path):
        from insurance_governance.outcome.results import OutcomeResult
        results = [
            OutcomeResult(outcome="price_value", test_name="t1", passed=True, period="2025-Q4"),
            OutcomeResult(outcome="claims", test_name="t2", passed=False,
                         period="2025-Q4", details="Something failed."),
        ]
        report = self._make_report(results=results)
        out = report.write_html(tmp_path / "test.html")
        content = out.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_to_dict_summary_warning_count(self):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import Severity
        results = [
            OutcomeResult(
                outcome="claims", test_name="warn", passed=False,
                severity=Severity.WARNING, period="2025-Q4",
            )
        ]
        report = self._make_report(results=results)
        d = report.to_dict()
        assert d["summary"]["warnings"] == 1
        assert d["summary"]["failed"] == 1

    def test_to_dict_model_card_has_model_id(self):
        report = self._make_report()
        d = report.to_dict()
        assert "model_id" in d["model_card"]
        assert d["model_card"]["model_id"] == "motor-freq-v3"


# ===========================================================================
# audit/log.py — ExplainabilityAuditLog edge cases
# ===========================================================================

class TestAuditLogEdgeCases:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_log(self, filename="audit.jsonl"):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        return ExplainabilityAuditLog(
            path=os.path.join(self._tmpdir, filename),
            model_id="motor-v3",
            model_version="3.0.0",
        )

    def test_log_opens_existing_file_without_truncating(self):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = os.path.join(self._tmpdir, "existing.jsonl")
        # Pre-populate
        log1 = ExplainabilityAuditLog(path, "motor-v3", "3.0.0")
        log1.append(_make_audit_entry())

        # Re-open — should not truncate
        log2 = ExplainabilityAuditLog(path, "motor-v3", "3.0.0")
        entries = log2.read_all()
        assert len(entries) == 1

    def test_read_since_returns_empty_when_all_before_cutoff(self):
        log = self._make_log()
        log.append(_make_audit_entry(timestamp_utc="2024-01-01T00:00:00+00:00"))
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = log.read_since(cutoff)
        assert result == []

    def test_read_since_includes_entry_at_exact_cutoff(self):
        log = self._make_log()
        ts = "2025-06-01T12:00:00+00:00"
        log.append(_make_audit_entry(timestamp_utc=ts))
        cutoff = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = log.read_since(cutoff)
        assert len(result) == 1

    def test_export_period_empty_window_writes_header_only(self):
        log = self._make_log()
        log.append(_make_audit_entry(timestamp_utc="2025-03-01T00:00:00+00:00"))
        out_path = os.path.join(self._tmpdir, "empty_export.jsonl")
        log.export_period(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 12, 31, tzinfo=timezone.utc),
            path=out_path,
        )
        lines = Path(out_path).read_text().splitlines()
        # Only the metadata header line
        assert len(lines) == 1
        assert lines[0].startswith("#")
        meta = json.loads(lines[0][2:])
        assert meta["entry_count"] == 0

    def test_export_period_naive_datetimes_treated_as_utc(self):
        log = self._make_log()
        log.append(_make_audit_entry(timestamp_utc="2025-06-01T12:00:00+00:00"))
        out_path = os.path.join(self._tmpdir, "naive_export.jsonl")
        # Naive datetimes
        result = log.export_period(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 12, 31),
            path=out_path,
        )
        lines = [l for l in result.read_text().splitlines() if not l.startswith("#")]
        assert len(lines) == 1

    def test_verify_chain_empty_log_returns_empty(self):
        log = self._make_log()
        assert log.verify_chain() == []

    def test_verify_chain_detects_deserialisation_error(self):
        """Lines with missing required fields should show in verify_chain failures."""
        log = self._make_log()
        # Append a line with a missing required field
        bad_line = json.dumps({"entry_id": "bad", "not_valid": True})
        with open(log.path, "a") as f:
            f.write(bad_line + "\n")
        failures = log.verify_chain()
        assert len(failures) >= 1

    def test_append_preserves_session_id(self):
        log = self._make_log()
        entry = _make_audit_entry(session_id="batch-2025-q4")
        log.append(entry)
        loaded = log.read_all()[0]
        assert loaded.session_id == "batch-2025-q4"

    def test_multiple_appends_maintain_order(self):
        log = self._make_log()
        for i in range(5):
            log.append(_make_audit_entry(prediction=float(100 + i * 10)))
        entries = log.read_all()
        predictions = [e.prediction for e in entries]
        assert predictions == [100.0, 110.0, 120.0, 130.0, 140.0]

    def test_corrupt_json_line_raises_on_read_all(self):
        log = self._make_log()
        with open(log.path, "w") as f:
            f.write("this is not json\n")
        with pytest.raises(ValueError, match="Corrupt JSONL"):
            log.read_all()


# ===========================================================================
# audit/customer_explanation.py — PlainLanguageExplainer edge cases
# ===========================================================================

class TestPlainLanguageExplainerEdgeCases:
    def _make_explainer(self, **kwargs):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        defaults = dict(feature_labels=FEATURE_LABELS)
        defaults.update(kwargs)
        return PlainLanguageExplainer(**defaults)

    def test_zero_base_premium_raises(self):
        explainer = self._make_explainer()
        entry = _make_audit_entry()
        with pytest.raises(ValueError, match="base_premium must be positive"):
            explainer.generate(entry, base_premium=0.0)

    def test_all_features_unlabelled_returns_fallback(self):
        """When no features map to labels, produce the fallback message."""
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(feature_labels={})  # no labels
        entry = _make_audit_entry()
        text = explainer.generate(entry, base_premium=350.0)
        assert "contact us" in text.lower() or "overall risk" in text.lower()

    def test_min_impact_pct_filters_small_contributions(self):
        """Factors with tiny impact should be filtered out."""
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels={"driver_age": "your age"},
            min_impact_pct=50.0,  # very high threshold — 50% of base
        )
        # SHAP value of 0.01 on a scale where total_shap << premium range
        # means impact will be tiny
        entry = _make_audit_entry(
            feature_importances={"driver_age": 0.001},
            prediction=350.5,
            final_premium=350.5,
        )
        text = explainer.generate(entry, base_premium=350.0)
        # Should hit fallback because impact < 50% of base
        assert "contact us" in text.lower() or "overall risk" in text.lower()

    def test_usd_currency_symbol(self):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels=FEATURE_LABELS, currency="USD"
        )
        entry = _make_audit_entry(final_premium=412.50)
        text = explainer.generate(entry, base_premium=350.0)
        assert "$" in text

    def test_unknown_currency_uses_currency_code(self):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels=FEATURE_LABELS, currency="CHF"
        )
        entry = _make_audit_entry(final_premium=412.50)
        text = explainer.generate(entry, base_premium=350.0)
        assert "CHF" in text

    def test_generate_bullet_list_empty_importances(self):
        explainer = self._make_explainer()
        entry = _make_audit_entry(feature_importances={})
        bullets = explainer.generate_bullet_list(entry, base_premium=350.0)
        # Should return at least the summary bullet
        assert isinstance(bullets, list)
        assert len(bullets) >= 1

    def test_generate_uses_final_premium_over_prediction(self):
        """When final_premium differs from prediction, final_premium takes precedence."""
        explainer = self._make_explainer()
        entry = _make_audit_entry(prediction=400.0, final_premium=450.0)
        text = explainer.generate(entry, base_premium=350.0)
        assert "450.00" in text
        assert "400.00" not in text

    def test_generate_bullet_list_uses_final_premium(self):
        explainer = self._make_explainer()
        entry = _make_audit_entry(prediction=380.0, final_premium=420.0)
        bullets = explainer.generate_bullet_list(entry, base_premium=350.0)
        assert "420.00" in bullets[0]

    def test_max_factors_zero_returns_fallback(self):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels=FEATURE_LABELS, max_factors=0
        )
        entry = _make_audit_entry()
        text = explainer.generate(entry, base_premium=350.0)
        # With max_factors=0, contributions list is empty → fallback
        assert "contact us" in text.lower() or "overall risk" in text.lower()

    def test_total_shap_zero_returns_fallback_not_crash(self):
        """All SHAP values sum to zero — should not divide by zero."""
        explainer = self._make_explainer()
        entry = _make_audit_entry(
            feature_importances={"driver_age": 0.5, "ncb_years": -0.5},
            prediction=350.0,
            final_premium=350.0,
        )
        # If total_shap = 0, scale = 0, so all impacts are 0
        # Everything will be filtered by min_impact_pct → fallback
        text = explainer.generate(entry, base_premium=350.0)
        assert isinstance(text, str)

    def test_feature_labels_property_is_copy(self):
        """Modifying the returned dict should not affect the explainer."""
        explainer = self._make_explainer()
        labels = explainer.feature_labels
        labels["injected"] = "hack"
        assert "injected" not in explainer.feature_labels


# ===========================================================================
# audit/report.py — AuditSummaryReport edge cases
# ===========================================================================

class TestAuditSummaryReportEdgeCases:
    def setup_method(self):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        self._tmpdir = tempfile.mkdtemp()
        self._log_path = os.path.join(self._tmpdir, "audit.jsonl")
        self._log = ExplainabilityAuditLog(self._log_path, "motor-v3", "3.0.0")

    def test_empty_log_build_returns_zero_entries(self):
        from insurance_governance.audit.report import AuditSummaryReport
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert data["metadata"]["entry_count"] == 0
        assert data["decision_volume"]["total"] == 0
        assert data["feature_importance"] == []

    def test_empty_log_integrity_passes(self):
        from insurance_governance.audit.report import AuditSummaryReport
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert data["integrity"]["pass"] is True
        assert data["integrity"]["total_checked"] == 0

    def test_entries_cached_after_first_build(self):
        """Second call to build() should return same entries without re-reading."""
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(_make_audit_entry(timestamp_utc="2025-03-01T00:00:00+00:00"))
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data1 = report.build()
        # Append after first build — second build should still see 1 entry (cached)
        self._log.append(_make_audit_entry(timestamp_utc="2025-04-01T00:00:00+00:00"))
        data2 = report.build()
        assert data1["metadata"]["entry_count"] == data2["metadata"]["entry_count"]

    def test_date_range_excludes_entries_outside_window(self):
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(_make_audit_entry(timestamp_utc="2025-01-01T00:00:00+00:00"))
        self._log.append(_make_audit_entry(timestamp_utc="2025-06-01T00:00:00+00:00"))
        self._log.append(_make_audit_entry(timestamp_utc="2025-12-01T00:00:00+00:00"))
        report = AuditSummaryReport(
            self._log,
            period="2025-H1",
            start=datetime(2025, 4, 1, tzinfo=timezone.utc),
            end=datetime(2025, 8, 31, tzinfo=timezone.utc),
        )
        data = report.build()
        assert data["metadata"]["entry_count"] == 1  # only June

    def test_segment_analysis_missing_feature_uses_missing_label(self):
        """Entries without the segment_feature in input_features get '__missing__'."""
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(
            _make_audit_entry(
                input_features={"driver_age": 30},  # no "region"
                timestamp_utc="2025-03-01T00:00:00+00:00",
            )
        )
        report = AuditSummaryReport(
            self._log, period="2025-Q1", segment_feature="region"
        )
        data = report.build()
        seg_vals = {row["segment"] for row in data["segment_analysis"]["rows"]}
        assert "__missing__" in seg_vals

    def test_feature_importance_std_zero_for_single_entry(self):
        """With a single entry, std should be 0.0."""
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(
            _make_audit_entry(
                feature_importances={"driver_age": 0.25},
                timestamp_utc="2025-03-01T00:00:00+00:00",
            )
        )
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        fi_row = next(r for r in data["feature_importance"] if r["feature"] == "driver_age")
        assert fi_row["std_abs_shap"] == 0.0
        assert fi_row["n_observations"] == 1

    def test_save_html_fail_integrity_shows_fail_badge(self):
        """Tampered log should render FAIL in the HTML."""
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(_make_audit_entry(timestamp_utc="2025-03-01T00:00:00+00:00"))
        # Tamper the file
        with open(self._log_path, "r") as f:
            content = f.read()
        d = json.loads(content.strip())
        d["prediction"] = 999.99  # change value, keep old hash
        with open(self._log_path, "w") as f:
            f.write(json.dumps(d) + "\n")

        report = AuditSummaryReport(self._log, period="2025-Q1")
        out = os.path.join(self._tmpdir, "tampered.html")
        result = report.save_html(out)
        html = result.read_text()
        assert "FAIL" in html

    def test_metadata_has_start_end_when_filtered(self):
        from insurance_governance.audit.report import AuditSummaryReport
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)
        report = AuditSummaryReport(
            self._log, period="2025", start=start, end=end
        )
        data = report.build()
        assert data["metadata"]["start"] is not None
        assert data["metadata"]["end"] is not None

    def test_override_rate_pct_zero_when_no_overrides(self):
        from insurance_governance.audit.report import AuditSummaryReport
        for _ in range(5):
            self._log.append(
                _make_audit_entry(
                    override_applied=False,
                    timestamp_utc="2025-03-01T00:00:00+00:00",
                )
            )
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert data["decision_volume"]["override_rate_pct"] == 0.0
        assert data["decision_volume"]["overridden"] == 0

    def test_human_reviewed_pct_computed_correctly(self):
        from insurance_governance.audit.report import AuditSummaryReport
        self._log.append(
            _make_audit_entry(
                human_reviewed=True,
                timestamp_utc="2025-03-01T00:00:00+00:00",
            )
        )
        self._log.append(
            _make_audit_entry(
                human_reviewed=True,
                timestamp_utc="2025-03-02T00:00:00+00:00",
            )
        )
        self._log.append(
            _make_audit_entry(
                human_reviewed=False,
                timestamp_utc="2025-03-03T00:00:00+00:00",
            )
        )
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        vol = data["decision_volume"]
        assert vol["human_reviewed"] == 2
        assert vol["human_reviewed_pct"] == pytest.approx(66.67, abs=0.1)


# ===========================================================================
# Top-level imports from insurance_governance
# ===========================================================================

class TestTopLevelImports:
    def test_outcome_imports(self):
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
        assert all(obj is not None for obj in [
            OutcomeTestingFramework, PriceValueMetrics, ClaimsMetrics,
            OutcomeResult, OutcomeSuite, CustomerSegment, SegmentComparison,
            OutcomeTestingReport,
        ])

    def test_audit_imports(self):
        from insurance_governance import (
            ExplainabilityAuditEntry,
            ExplainabilityAuditLog,
            PlainLanguageExplainer,
            AuditSummaryReport,
            SHAPExplainer,
        )
        assert all(obj is not None for obj in [
            ExplainabilityAuditEntry, ExplainabilityAuditLog,
            PlainLanguageExplainer, AuditSummaryReport, SHAPExplainer,
        ])

    def test_euaia_imports(self):
        from insurance_governance.euaia import (
            Article13Document,
            ConformityAssessment,
            AIActClassifier,
        )
        assert all(obj is not None for obj in [
            Article13Document, ConformityAssessment, AIActClassifier,
        ])
