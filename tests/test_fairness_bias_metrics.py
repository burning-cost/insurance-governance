"""
Tests for insurance_fairness.bias_metrics.

These tests exercise the fairness metric functions from the insurance-fairness
package, which is a dependency of insurance-governance. The tests focus on:
  - Analytically verifiable results
  - Edge cases (multi-group, equal premiums, zero exposure)
  - Bootstrap confidence intervals
  - Non-log-space parity metrics
  - RAG status thresholds
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("insurance_fairness")

from insurance_fairness.bias_metrics import (
    CalibrationResult,
    DemographicParityResult,
    DisparateImpactResult,
    EqualisedOddsResult,
    GiniResult,
    TheilResult,
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_df() -> pl.DataFrame:
    """500+500 policies with a deliberate 30% premium uplift for group 1."""
    rng = np.random.default_rng(2024)
    n = 1000
    group = np.array([0] * 500 + [1] * 500, dtype=np.int32)
    base = rng.lognormal(4.6, 0.3, n)
    pred = base * np.where(group == 1, 1.3, 1.0)
    actual = pred * rng.lognormal(0.0, 0.15, n)
    exposure = rng.uniform(0.5, 1.0, n)
    return pl.DataFrame({
        "gender": group,
        "predicted_premium": pred.tolist(),
        "claim_amount": actual.tolist(),
        "exposure": exposure.tolist(),
    })


@pytest.fixture(scope="module")
def multi_group_df() -> pl.DataFrame:
    """300 policies each for groups A, B, C with distinct mean premiums."""
    rng = np.random.default_rng(99)
    groups = ["A"] * 300 + ["B"] * 300 + ["C"] * 300
    means = {"A": 100.0, "B": 130.0, "C": 80.0}
    preds, actuals = [], []
    for g in groups:
        m = means[g]
        p = float(rng.lognormal(np.log(m), 0.2))
        preds.append(p)
        actuals.append(float(p * rng.lognormal(0.0, 0.15)))
    exposure = rng.uniform(0.5, 1.0, len(groups))
    return pl.DataFrame({
        "region": groups,
        "predicted_premium": preds,
        "claim_amount": actuals,
        "exposure": exposure.tolist(),
    })


@pytest.fixture(scope="module")
def perfect_calibration_df() -> pl.DataFrame:
    """Dataset where A/E is exactly 1.0 for both groups (predictions = actuals)."""
    n = 400
    group = [0] * 200 + [1] * 200
    pred = [100.0] * 200 + [150.0] * 200
    actual = [100.0] * 200 + [150.0] * 200  # Perfect A/E
    exposure = [1.0] * n
    return pl.DataFrame({
        "g": group,
        "pred": pred,
        "actual": actual,
        "exp": exposure,
    })


# ---------------------------------------------------------------------------
# demographic_parity_ratio
# ---------------------------------------------------------------------------


class TestDemographicParityRatio:
    def test_equal_groups_zero_log_ratio(self):
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert abs(result.log_ratio) < 1e-10
        assert abs(result.ratio - 1.0) < 1e-10

    def test_known_log_ratio_log_space(self):
        """Group 1 with exactly double the premium -> log-ratio = log(2)."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 200.0],
        })
        result = demographic_parity_ratio(df, "g", "pred", log_space=True)
        assert abs(result.log_ratio - math.log(2.0)) < 0.01
        assert result.rag == "red"

    def test_level_space_disparity(self):
        """Non-log-space version should give a level difference-based ratio."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 150.0],
        })
        result = demographic_parity_ratio(df, "g", "pred", log_space=False)
        # In level space: diff=50, ref_mean=100, ratio = 1 + 50/100 = 1.5
        assert result.ratio > 1.0

    def test_multi_group_returns_result(self, multi_group_df):
        result = demographic_parity_ratio(
            multi_group_df, "region", "predicted_premium", "exposure"
        )
        assert isinstance(result, DemographicParityResult)
        assert "A" in result.group_means
        assert "B" in result.group_means
        assert "C" in result.group_means

    def test_multi_group_log_ratio_positive(self, multi_group_df):
        """Groups B and C differ from A — log_ratio should be > 0."""
        result = demographic_parity_ratio(
            multi_group_df, "region", "predicted_premium", "exposure"
        )
        assert result.log_ratio > 0.0

    def test_no_exposure_equal_weighting(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 200.0],
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert abs(result.log_ratio - math.log(2.0)) < 0.01

    def test_exposure_weighting_changes_result(self):
        """Heavy exposure on high-premium policy changes the group mean."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 200.0, 100.0, 200.0],
            "exp": [9.0, 1.0, 1.0, 9.0],  # group 0 mostly low, group 1 mostly high
        })
        result_with = demographic_parity_ratio(df, "g", "pred", "exp")
        result_without = demographic_parity_ratio(df, "g", "pred")
        # With exposure weighting, the disparity should be larger
        assert abs(result_with.log_ratio) > abs(result_without.log_ratio)

    def test_rag_green_for_small_disparity(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 101.0],  # < 1% difference
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert result.rag == "green"

    def test_rag_amber_for_moderate_disparity(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 107.0],  # log(1.07) ~ 0.068, between amber=0.05 and red=0.10
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert result.rag == "amber"

    def test_rag_red_for_large_disparity(self, binary_df):
        """30% premium disparity in binary_df should be red."""
        result = demographic_parity_ratio(
            binary_df, "gender", "predicted_premium", "exposure"
        )
        assert result.rag == "red"

    def test_bootstrap_ci_produced_when_requested(self, binary_df):
        result = demographic_parity_ratio(
            binary_df, "gender", "predicted_premium", "exposure",
            n_bootstrap=100,
        )
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        # Bootstrap CI bounds should be a valid range (lower <= upper)
        assert result.ci_lower <= result.ci_upper

    def test_bootstrap_ci_none_when_not_requested(self, binary_df):
        result = demographic_parity_ratio(
            binary_df, "gender", "predicted_premium", "exposure",
            n_bootstrap=0,
        )
        assert result.ci_lower is None
        assert result.ci_upper is None

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            demographic_parity_ratio(df, "gender", "pred")

    def test_group_exposures_sum_correctly(self, binary_df):
        result = demographic_parity_ratio(
            binary_df, "gender", "predicted_premium", "exposure"
        )
        total_from_result = sum(result.group_exposures.values())
        expected_total = float(binary_df["exposure"].sum())
        assert abs(total_from_result - expected_total) < 1e-3


# ---------------------------------------------------------------------------
# calibration_by_group
# ---------------------------------------------------------------------------


class TestCalibrationByGroup:
    def test_perfect_calibration_low_max_disparity(self, perfect_calibration_df):
        result = calibration_by_group(
            perfect_calibration_df, "g", "pred", "actual", "exp", n_deciles=5
        )
        assert isinstance(result, CalibrationResult)
        assert result.max_disparity < 0.02

    def test_known_calibration_disparity(self):
        """Group 0 A/E = 0.5, group 1 A/E = 1.0 — max_disparity >= 0.4."""
        df = pl.DataFrame({
            "g": [0] * 20 + [1] * 20,
            "pred": [100.0] * 40,
            "actual": [50.0] * 20 + [100.0] * 20,
            "exp": [1.0] * 40,
        })
        result = calibration_by_group(df, "g", "pred", "actual", n_deciles=2)
        assert result.max_disparity >= 0.4

    def test_result_has_correct_number_of_deciles(self, binary_df):
        result = calibration_by_group(
            binary_df, "gender", "predicted_premium", "claim_amount",
            "exposure", n_deciles=5
        )
        assert len(result.actual_to_expected) == 5

    def test_group_counts_populated(self, binary_df):
        result = calibration_by_group(
            binary_df, "gender", "predicted_premium", "claim_amount",
            n_deciles=10
        )
        assert "0" in result.group_counts
        assert "1" in result.group_counts

    def test_rag_status_valid(self, binary_df):
        result = calibration_by_group(
            binary_df, "gender", "predicted_premium", "claim_amount"
        )
        assert result.rag in ("green", "amber", "red")

    def test_no_exposure_runs_cleanly(self, binary_df):
        result = calibration_by_group(
            binary_df, "gender", "predicted_premium", "claim_amount"
        )
        assert isinstance(result, CalibrationResult)


# ---------------------------------------------------------------------------
# disparate_impact_ratio
# ---------------------------------------------------------------------------


class TestDisparateImpactRatio:
    def test_equal_groups_ratio_one(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 100.0],
        })
        result = disparate_impact_ratio(df, "g", "pred")
        assert abs(result.ratio - 1.0) < 1e-10

    def test_known_ratio_explicit_reference(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 80.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", reference_group="0")
        assert abs(result.ratio - 0.80) < 0.001

    def test_clearly_red_dir(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 70.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", reference_group="0")
        assert result.rag == "red"

    def test_auto_reference_is_highest_mean(self):
        """Without explicit reference, the group with the highest mean is reference."""
        df = pl.DataFrame({
            "g": ["X", "Y"],
            "pred": [80.0, 120.0],
        })
        result = disparate_impact_ratio(df, "g", "pred")
        # Y has higher mean, so DIR = 80/120 < 1.0
        assert result.ratio < 1.0

    def test_multi_group_min_ratio(self, multi_group_df):
        """For 3 groups, DIR should be the minimum ratio (most disadvantaged)."""
        result = disparate_impact_ratio(
            multi_group_df, "region", "predicted_premium", "exposure"
        )
        assert isinstance(result, DisparateImpactResult)
        assert result.ratio <= 1.0

    def test_exposure_weighting(self, binary_df):
        result_with_exp = disparate_impact_ratio(
            binary_df, "gender", "predicted_premium", "exposure"
        )
        result_no_exp = disparate_impact_ratio(
            binary_df, "gender", "predicted_premium"
        )
        # Both should produce a valid result; with exposure, values differ
        assert isinstance(result_with_exp, DisparateImpactResult)
        assert isinstance(result_no_exp, DisparateImpactResult)

    def test_group_means_both_present(self, binary_df):
        result = disparate_impact_ratio(
            binary_df, "gender", "predicted_premium"
        )
        assert "0" in result.group_means
        assert "1" in result.group_means

    def test_explicit_reference_group_used(self):
        df = pl.DataFrame({
            "g": ["A", "B", "C"],
            "pred": [100.0, 120.0, 90.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", reference_group="B")
        # C/B = 90/120 = 0.75 is minimum; A/B = 100/120 ~ 0.833
        assert result.ratio < 1.0


# ---------------------------------------------------------------------------
# equalised_odds
# ---------------------------------------------------------------------------


class TestEqualisedOdds:
    def test_returns_equalised_odds_result(self, binary_df):
        result = equalised_odds(
            binary_df, "gender", "predicted_premium", "claim_amount", "exposure"
        )
        assert isinstance(result, EqualisedOddsResult)

    def test_two_group_metrics_returned(self, binary_df):
        result = equalised_odds(
            binary_df, "gender", "predicted_premium", "claim_amount", "exposure"
        )
        assert len(result.group_metrics) == 2
        group_vals = {gm.group_value for gm in result.group_metrics}
        assert "0" in group_vals
        assert "1" in group_vals

    def test_max_tpr_disparity_non_negative(self, binary_df):
        result = equalised_odds(
            binary_df, "gender", "predicted_premium", "claim_amount"
        )
        assert result.max_tpr_disparity >= 0.0

    def test_perfect_rank_correlation_within_groups(self):
        """When predictions perfectly rank actuals, both groups have high Spearman r."""
        n = 200
        pred = list(range(1, n + 1))
        actual = list(range(1, n + 1))
        group = [0] * (n // 2) + [1] * (n // 2)
        df = pl.DataFrame({
            "g": group,
            "pred": [float(x) for x in pred],
            "actual": [float(x) for x in actual],
        })
        result = equalised_odds(df, "g", "pred", "actual")
        for gm in result.group_metrics:
            assert gm.metric_value > 0.99

    def test_binary_threshold_mode(self):
        """With binary_threshold set, uses TPR rather than Spearman r."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [0.8, 0.2, 0.9, 0.1],
            "actual": [1, 0, 1, 0],
        })
        result = equalised_odds(df, "g", "pred", "actual", binary_threshold=0.5)
        assert isinstance(result, EqualisedOddsResult)
        # Group 0: threshold 0.5 -> pred [0.8, 0.2] -> pred_bin [1, 0], actual [1, 0]
        # TPR for group 0 = 1 actual positive: pred_bin[actual==1] = [1] -> TPR=1.0
        for gm in result.group_metrics:
            # At least one group should have a valid metric
            assert gm.metric_value is not None

    def test_too_few_samples_returns_nan(self):
        """Groups with fewer than 3 members produce nan metric."""
        df = pl.DataFrame({
            "g": [0, 1],  # Only 1 policy per group
            "pred": [100.0, 200.0],
            "actual": [90.0, 210.0],
        })
        result = equalised_odds(df, "g", "pred", "actual")
        for gm in result.group_metrics:
            assert math.isnan(gm.metric_value)

    def test_multi_group_produces_three_metrics(self, multi_group_df):
        result = equalised_odds(
            multi_group_df, "region", "predicted_premium", "claim_amount"
        )
        assert len(result.group_metrics) == 3


# ---------------------------------------------------------------------------
# gini_by_group
# ---------------------------------------------------------------------------


class TestGiniByGroup:
    def test_equal_premiums_zero_gini(self):
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = gini_by_group(df, "g", "pred")
        assert abs(result.overall_gini) < 1e-10
        assert abs(result.group_ginis["0"]) < 1e-10
        assert abs(result.group_ginis["1"]) < 1e-10

    def test_gini_of_known_distribution(self):
        """Gini of [1, 2, 3, 4] with equal weights is approximately 0.25."""
        df = pl.DataFrame({
            "group": [0, 0, 1, 1],
            "pred": [1.0, 2.0, 3.0, 4.0],
        })
        result = gini_by_group(df, "group", "pred")
        assert 0.20 < result.overall_gini < 0.35

    def test_returns_gini_result(self, binary_df):
        result = gini_by_group(
            binary_df, "gender", "predicted_premium", "exposure"
        )
        assert isinstance(result, GiniResult)

    def test_group_ginis_populated(self, binary_df):
        result = gini_by_group(binary_df, "gender", "predicted_premium")
        assert "0" in result.group_ginis
        assert "1" in result.group_ginis

    def test_overall_gini_non_negative(self, binary_df):
        result = gini_by_group(binary_df, "gender", "predicted_premium")
        assert result.overall_gini >= 0.0

    def test_max_disparity_non_negative(self, binary_df):
        result = gini_by_group(binary_df, "gender", "predicted_premium", "exposure")
        assert result.max_disparity >= 0.0

    def test_multi_group_ginis(self, multi_group_df):
        result = gini_by_group(multi_group_df, "region", "predicted_premium")
        assert "A" in result.group_ginis
        assert "B" in result.group_ginis
        assert "C" in result.group_ginis

    def test_higher_spread_higher_gini(self):
        """A distribution with more spread should have a higher Gini coefficient."""
        df_tight = pl.DataFrame({
            "g": [0, 0, 0, 0],
            "pred": [95.0, 98.0, 102.0, 105.0],
        })
        df_wide = pl.DataFrame({
            "g": [0, 0, 0, 0],
            "pred": [10.0, 50.0, 150.0, 300.0],
        })
        r_tight = gini_by_group(df_tight, "g", "pred")
        r_wide = gini_by_group(df_wide, "g", "pred")
        assert r_wide.overall_gini > r_tight.overall_gini

    def test_missing_column_raises(self):
        df = pl.DataFrame({"g": [0, 1], "a": [1.0, 2.0]})
        with pytest.raises(ValueError):
            gini_by_group(df, "g", "nonexistent")


# ---------------------------------------------------------------------------
# theil_index
# ---------------------------------------------------------------------------


class TestTheilIndex:
    def test_equal_premiums_zero_theil(self):
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = theil_index(df, "g", "pred")
        assert abs(result.theil_total) < 1e-8
        assert abs(result.theil_between) < 1e-8

    def test_decomposition_within_plus_between_equals_total(self, binary_df):
        """T_within + T_between should approximately equal T_total."""
        result = theil_index(
            binary_df, "gender", "predicted_premium", "exposure"
        )
        reconstructed = result.theil_within + result.theil_between
        assert abs(reconstructed - result.theil_total) < 0.05

    def test_nonpositive_prediction_raises(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 0.0],
        })
        with pytest.raises(ValueError, match="strictly positive"):
            theil_index(df, "g", "pred")

    def test_negative_prediction_raises(self):
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, -1.0],
        })
        with pytest.raises(ValueError, match="strictly positive"):
            theil_index(df, "g", "pred")

    def test_between_group_increases_with_larger_disparity(self):
        df_small = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 110.0, 110.0],
        })
        df_large = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 300.0, 300.0],
        })
        r_small = theil_index(df_small, "g", "pred")
        r_large = theil_index(df_large, "g", "pred")
        assert r_large.theil_between > r_small.theil_between

    def test_returns_theil_result(self, binary_df):
        result = theil_index(binary_df, "gender", "predicted_premium", "exposure")
        assert isinstance(result, TheilResult)

    def test_group_contributions_populated(self, binary_df):
        result = theil_index(binary_df, "gender", "predicted_premium", "exposure")
        assert "0" in result.group_contributions
        assert "1" in result.group_contributions

    def test_multi_group_decomposition(self, multi_group_df):
        result = theil_index(multi_group_df, "region", "predicted_premium", "exposure")
        assert "A" in result.group_contributions
        assert "B" in result.group_contributions
        assert "C" in result.group_contributions
        reconstructed = result.theil_within + result.theil_between
        assert abs(reconstructed - result.theil_total) < 0.1

    def test_missing_column_raises(self):
        df = pl.DataFrame({"g": [0, 1], "a": [1.0, 2.0]})
        with pytest.raises(ValueError):
            theil_index(df, "g", "nonexistent")
