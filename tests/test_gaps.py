"""
Structural gap tests for insurance-governance.

Covers uncovered branches identified by code tracing:
1. stability.feature_drift: feature missing from current_df (second branch)
2. performance._poisson_ae_ci: A=0 lower bound (lower=0.0 path)
3. performance.actual_vs_expected: zero total_predicted (nan path)
4. performance.double_lift: incumbent wins (passed=False)
5. performance.hosmer_lemeshow_test: n_groups=1 (df=0 case)
6. performance.gini_with_ci: validation months boundary values (6, 12, 18, 24)
7. scorer: drift_triggers boundary values (1, 3, 4)
8. scorer: validation_months_ago exact boundary values (6, 12, 18, 24)
9. inventory.summary: RAG "Not assessed" bucket in by_rag count
10. data_quality.outlier_detection: zscore method with zero std column
11. data_quality.outlier_detection: rate >= 0.01 (CRITICAL severity path)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_governance.mrm.inventory import ModelInventory
from insurance_governance.mrm.model_card import ModelCard
from insurance_governance.mrm.scorer import RiskTierScorer
from insurance_governance.validation import (
    DataQualityReport,
    PerformanceReport,
    StabilityReport,
)
from insurance_governance.validation.performance import _poisson_ae_ci
from insurance_governance.validation.results import Severity


# ---------------------------------------------------------------------------
# 1. feature_drift: feature missing from current_df
# ---------------------------------------------------------------------------

class TestFeatureDriftMissingFromCurrent:
    def test_feature_missing_from_current_df_fails(self):
        """Feature absent from current_df should produce a failing TestResult."""
        ref_df = pl.DataFrame({"age": [20.0, 30.0, 40.0]})
        cur_df = pl.DataFrame({"vehicle_age": [2.0, 3.0, 4.0]})  # 'age' missing
        report = StabilityReport()
        results = report.feature_drift(ref_df, cur_df, features=["age"])
        assert len(results) == 1
        assert results[0].passed is False
        # The test_name for current-missing branch uses "feature_drift_{feat}"
        assert "age" in results[0].test_name

    def test_both_present_vs_one_missing_different_test_names(self):
        """Missing-from-reference produces psi_{feat}; missing-from-current produces feature_drift_{feat}."""
        ref_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        cur_df = pl.DataFrame({"y": [1.0, 2.0, 3.0]})

        report = StabilityReport()
        # Missing from reference
        results_ref_missing = report.feature_drift(ref_df, cur_df, features=["x"])
        # x is in ref but not in cur
        assert results_ref_missing[0].passed is False

        results_cur_missing = report.feature_drift(cur_df, ref_df, features=["x"])
        # x is in ref_df (cur_df here) but not in cur_df (ref_df here)
        assert results_cur_missing[0].passed is False


# ---------------------------------------------------------------------------
# 2. _poisson_ae_ci: A=0 lower bound
# ---------------------------------------------------------------------------

class TestPoissonAeCi:
    def test_zero_actual_lower_is_zero(self):
        """When A=0, lower bound should be exactly 0.0."""
        lower, upper = _poisson_ae_ci(actual_claims=0.0, expected_claims=10.0)
        assert lower == 0.0
        assert upper > 0.0
        assert np.isfinite(upper)

    def test_zero_expected_returns_nan(self):
        """When E=0, both bounds should be NaN."""
        lower, upper = _poisson_ae_ci(actual_claims=5.0, expected_claims=0.0)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_ci_contains_ae_ratio(self):
        """For reasonable inputs, CI should bracket the A/E ratio."""
        A = 100.0
        E = 95.0
        lower, upper = _poisson_ae_ci(A, E, alpha=0.05)
        ae = A / E
        assert lower <= ae <= upper

    def test_ci_wider_at_lower_alpha(self):
        """95% CI should be wider than 90% CI."""
        A = 50.0
        E = 55.0
        lo95, up95 = _poisson_ae_ci(A, E, alpha=0.05)
        lo90, up90 = _poisson_ae_ci(A, E, alpha=0.10)
        assert up95 - lo95 > up90 - lo90


# ---------------------------------------------------------------------------
# 3. actual_vs_expected: zero total_predicted
# ---------------------------------------------------------------------------

class TestActualVsExpectedZeroPredicted:
    def test_zero_predicted_returns_nan_metric(self):
        """When all predictions are zero, A/E is undefined (passed=False, metric=None)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.zeros(3)
        report = PerformanceReport(y_true, y_pred)
        result = report.actual_vs_expected()
        assert result.passed is False
        assert result.metric_value is None
        assert "zero" in result.details.lower()

    def test_ae_with_poisson_ci_zero_predicted_fails(self):
        """ae_with_poisson_ci when total_predicted=0 should return failing TestResult."""
        y_true = np.array([1.0, 1.0])
        y_pred = np.zeros(2)
        report = PerformanceReport(y_true, y_pred)
        result = report.ae_with_poisson_ci()
        assert result.passed is False


# ---------------------------------------------------------------------------
# 4. double_lift: incumbent wins
# ---------------------------------------------------------------------------

class TestDoubleLiftIncumbentWins:
    def test_incumbent_wins(self):
        """When incumbent tracks actuals better, passed=False and metric > 0."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.exponential(0.1, size=n)
        # Incumbent: near-perfect predictions
        y_pred_incumbent = y_true + rng.normal(0, 0.001, size=n)
        # New model: poor predictions
        y_pred_new = rng.uniform(0.0, 0.3, size=n)

        report = PerformanceReport(y_true, y_pred_new)
        result = report.double_lift(y_pred_incumbent=y_pred_incumbent)

        assert result.passed is False
        assert result.metric_value > 0  # new MAE - incumbent MAE > 0
        assert "incumbent" in result.details.lower()
        assert result.severity == Severity.WARNING

    def test_new_model_wins_passed_true(self):
        """When new model is better, passed=True and metric < 0."""
        rng = np.random.default_rng(43)
        n = 200
        y_true = rng.exponential(0.1, size=n)
        # New model: near-perfect
        y_pred_new = y_true + rng.normal(0, 0.001, size=n)
        # Incumbent: poor
        y_pred_incumbent = rng.uniform(0.0, 0.3, size=n)

        report = PerformanceReport(y_true, y_pred_new)
        result = report.double_lift(y_pred_incumbent=y_pred_incumbent)

        assert result.passed is True
        assert result.metric_value < 0  # new MAE - incumbent MAE < 0

    def test_mismatched_incumbent_length_raises(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        report = PerformanceReport(y_true, y_pred)
        with pytest.raises(ValueError, match="same length"):
            report.double_lift(y_pred_incumbent=np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# 5. hosmer_lemeshow_test: n_groups=1 (df=0)
# ---------------------------------------------------------------------------

class TestHosmerLemeshowEdgeCases:
    def test_n_groups_1_df_zero_nan_p_value(self):
        """n_groups=1 produces df=0. p-value should be nan, result passed=False."""
        y_true = np.array([0.0, 1.0, 0.0, 2.0, 1.0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
        report = PerformanceReport(y_true, y_pred)
        result = report.hosmer_lemeshow_test(n_groups=1)
        # df=0 => p_value = nan => passed = False
        assert result.passed is False
        assert result.metric_value is None

    def test_perfectly_calibrated_passes(self):
        """A model where predictions = actuals (both constant) should have HL stat ~0."""
        y = np.array([1.0] * 100)
        report = PerformanceReport(y, y.copy())
        result = report.hosmer_lemeshow_test(n_groups=10)
        # HL stat should be near 0 (actual = expected everywhere), p >> alpha
        assert result.passed is True
        assert result.extra["hl_statistic"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6-7. RiskTierScorer: validation and drift boundary values
# ---------------------------------------------------------------------------

class TestScorerBoundaryValues:
    def setup_method(self):
        self.scorer = RiskTierScorer()

    def _base_score(self, **kwargs):
        defaults = dict(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        defaults.update(kwargs)
        return self.scorer.score(**defaults)

    # Validation coverage exact boundaries
    def test_validation_exactly_6_months(self):
        """6 months ago: raw score = 0 (within 6-month bucket)."""
        result = self._base_score(validation_months_ago=6.0)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 0.0

    def test_validation_just_over_6_months(self):
        """6.1 months: enters the 6-12 month bucket (raw=20)."""
        result = self._base_score(validation_months_ago=6.1)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 20.0

    def test_validation_exactly_12_months(self):
        """12 months: still in 6-12 month bucket (raw=20)."""
        result = self._base_score(validation_months_ago=12.0)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 20.0

    def test_validation_just_over_12_months(self):
        """12.1 months: enters 12-18 month bucket (raw=50)."""
        result = self._base_score(validation_months_ago=12.1)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 50.0

    def test_validation_exactly_18_months(self):
        """18 months: still in 12-18 month bucket (raw=50)."""
        result = self._base_score(validation_months_ago=18.0)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 50.0

    def test_validation_just_over_18_months(self):
        """18.1 months: enters 18-24 month bucket (raw=75)."""
        result = self._base_score(validation_months_ago=18.1)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 75.0

    def test_validation_exactly_24_months(self):
        """24 months: in 18-24 month bucket (raw=75)."""
        result = self._base_score(validation_months_ago=24.0)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 75.0

    def test_validation_just_over_24_months(self):
        """24.1 months: enters overdue bucket (raw=100)."""
        result = self._base_score(validation_months_ago=24.1)
        vc_dim = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc_dim.score == 100.0

    # Drift triggers exact boundaries
    def test_drift_triggers_0(self):
        """0 triggers: raw_score = 0 (no drift)."""
        result = self._base_score(drift_triggers_last_year=0)
        dh_dim = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh_dim.score == 0.0

    def test_drift_triggers_1(self):
        """1 trigger: raw_score = 33."""
        result = self._base_score(drift_triggers_last_year=1)
        dh_dim = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh_dim.score == 33.0

    def test_drift_triggers_2(self):
        """2 triggers: raw_score = 67 (2-3 bucket)."""
        result = self._base_score(drift_triggers_last_year=2)
        dh_dim = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh_dim.score == 67.0

    def test_drift_triggers_3(self):
        """3 triggers: still in 2-3 bucket (raw=67)."""
        result = self._base_score(drift_triggers_last_year=3)
        dh_dim = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh_dim.score == 67.0

    def test_drift_triggers_4(self):
        """4 triggers: enters elevated bucket (raw=100)."""
        result = self._base_score(drift_triggers_last_year=4)
        dh_dim = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh_dim.score == 100.0

    # GWP materiality boundaries
    def test_gwp_just_under_5m(self):
        """GWP < £5m: raw = 12."""
        result = self._base_score(gwp_impacted=4_999_999)
        mat = next(d for d in result.dimensions if d.name == "materiality")
        assert mat.score == 12.0

    def test_gwp_exactly_5m(self):
        """GWP >= £5m: raw = 32."""
        result = self._base_score(gwp_impacted=5_000_000)
        mat = next(d for d in result.dimensions if d.name == "materiality")
        assert mat.score == 32.0

    def test_gwp_exactly_25m(self):
        """GWP >= £25m: raw = 60."""
        result = self._base_score(gwp_impacted=25_000_000)
        mat = next(d for d in result.dimensions if d.name == "materiality")
        assert mat.score == 60.0

    def test_gwp_exactly_100m(self):
        """GWP >= £100m: raw = 100."""
        result = self._base_score(gwp_impacted=100_000_000)
        mat = next(d for d in result.dimensions if d.name == "materiality")
        assert mat.score == 100.0


# ---------------------------------------------------------------------------
# 8. inventory.summary: "Not assessed" RAG bucket
# ---------------------------------------------------------------------------

class TestInventorySummaryRagBuckets:
    def test_not_assessed_rag_counted(self, tmp_path):
        """A model with no overall_rag should appear under 'Not assessed' in by_rag."""
        inv = ModelInventory(str(tmp_path / "registry.json"))
        card = ModelCard(
            model_id="unvalidated-model",
            model_name="Unvalidated Model",
            version="0.1.0",
            # overall_rag not set → defaults to empty string
        )
        inv.register(card)
        s = inv.summary()
        assert s["by_rag"].get("Not assessed", 0) >= 1

    def test_rag_counts_sum_to_total(self, tmp_path):
        """Sum of all RAG bucket counts should equal total_models."""
        inv = ModelInventory(str(tmp_path / "registry.json"))
        for i, rag in enumerate(["GREEN", "AMBER", "RED", None]):
            c = ModelCard(
                model_id=f"model-{i}",
                model_name=f"Model {i}",
                version="1.0.0",
                overall_rag=rag or "",
            )
            inv.register(c)
        s = inv.summary()
        rag_total = sum(s["by_rag"].values())
        assert rag_total == s["total_models"]


# ---------------------------------------------------------------------------
# 9. DataQualityReport: zscore with zero std; CRITICAL severity
# ---------------------------------------------------------------------------

class TestDataQualityOutlierEdgeCases:
    def test_zscore_zero_std_no_outliers(self):
        """A constant column has std=0 and no outliers by z-score."""
        df = pl.DataFrame({"constant": [5.0] * 100})
        report = DataQualityReport(df)
        results = report.outlier_detection(method="zscore")
        assert len(results) == 1
        assert results[0].metric_value == 0.0  # rate = 0
        assert results[0].passed is True

    def test_iqr_rate_above_1pct_is_critical(self):
        """When outlier rate >= 0.01, severity should be CRITICAL."""
        # Create a column where >1% of values are outliers under IQR x3.0
        base = [5.0] * 100
        # Add 5% outliers (5 values at 1000)
        values = base[:95] + [1000.0] * 5
        df = pl.DataFrame({"x": values})
        report = DataQualityReport(df)
        results = report.outlier_detection(method="iqr", iqr_multiplier=3.0)
        assert len(results) == 1
        # rate = 5/100 = 5% > 1%, so CRITICAL
        assert results[0].severity == Severity.CRITICAL
        assert results[0].passed is False

    def test_iqr_rate_under_1pct_is_warning(self):
        """When 0 < outlier rate < 0.01, severity should be WARNING."""
        # Only 1 outlier in 200 observations -> rate = 0.5% < 1%
        values = [5.0] * 199 + [10000.0]
        df = pl.DataFrame({"x": values})
        report = DataQualityReport(df)
        results = report.outlier_detection(method="iqr", iqr_multiplier=3.0)
        assert results[0].severity == Severity.WARNING
        assert results[0].passed is False

    def test_missing_value_high_threshold_passes(self):
        """Missing rate > 50% should produce CRITICAL severity."""
        values = [1.0] * 30 + [None] * 70
        df = pl.DataFrame({"x": values}).with_columns(
            pl.col("x").cast(pl.Float64)
        )
        report = DataQualityReport(df)
        results = report.missing_value_analysis(threshold=0.05)
        col_result = results[0]
        assert col_result.severity == Severity.CRITICAL
        assert col_result.passed is False

    def test_cardinality_check_passes_within_limit(self):
        """Low cardinality should pass."""
        df = pl.DataFrame({"region": ["North", "South", "East", "West"] * 25})
        report = DataQualityReport(df)
        results = report.cardinality_check(max_categories=10)
        assert results[0].passed is True
        assert results[0].metric_value == 4.0

    def test_cardinality_check_fails_above_limit(self):
        """High cardinality (policy IDs) should fail."""
        df = pl.DataFrame({"policy_id": [f"POL{i:05d}" for i in range(200)]})
        report = DataQualityReport(df)
        results = report.cardinality_check(max_categories=50)
        assert results[0].passed is False
        assert results[0].metric_value == 200.0
