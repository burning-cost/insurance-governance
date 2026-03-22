"""
Tests for insurance_fairness._utils helpers.

These tests exercise the internal utilities from the insurance-fairness package,
which is a runtime dependency of insurance-governance. Covering this code here
ensures the governance test suite verifies its dependency chain end-to-end.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("insurance_fairness")

from insurance_fairness._utils import (
    DEFAULT_THRESHOLDS,
    assign_prediction_deciles,
    bootstrap_ci,
    exposure_weighted_mean,
    log_ratio,
    rag_status,
    resolve_exposure,
    to_pandas,
    to_polars,
    validate_binary,
    validate_columns,
    validate_positive,
)


# ---------------------------------------------------------------------------
# to_polars
# ---------------------------------------------------------------------------


class TestToPolars:
    def test_passthrough_polars_frame(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = to_polars(df)
        assert result is df

    def test_converts_pandas_dataframe(self):
        import pandas as pd

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        result = to_polars(pdf)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 2)
        assert result["a"].to_list() == [1, 2, 3]

    def test_raises_for_unsupported_type(self):
        with pytest.raises(TypeError, match="Expected a Polars or pandas DataFrame"):
            to_polars({"a": [1, 2]})

    def test_raises_for_list_input(self):
        with pytest.raises(TypeError):
            to_polars([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_converts_polars_to_pandas(self):
        df = pl.DataFrame({"premium": [100.0, 200.0, 300.0], "group": [0, 1, 0]})
        result = to_pandas(df)
        import pandas as pd

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["premium", "group"]
        assert result.shape == (3, 2)

    def test_values_preserved(self):
        df = pl.DataFrame({"x": [1.5, 2.5, 3.5]})
        result = to_pandas(df)
        assert abs(result["x"].iloc[0] - 1.5) < 1e-10


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------


class TestValidateColumns:
    def test_passes_when_all_columns_present(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_columns(df, "a", "b", "c")  # Should not raise

    def test_raises_with_one_missing_column(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            validate_columns(df, "a", "missing_col")

    def test_raises_with_all_missing_columns(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            validate_columns(df, "x", "y", "z")

    def test_error_message_lists_available_columns(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Available columns"):
            validate_columns(df, "nonexistent")

    def test_no_columns_to_check_passes(self):
        df = pl.DataFrame({"a": [1]})
        validate_columns(df)  # No columns to validate — should not raise


# ---------------------------------------------------------------------------
# validate_positive
# ---------------------------------------------------------------------------


class TestValidatePositive:
    def test_raises_on_zero_value(self):
        df = pl.DataFrame({"v": [0.0, 1.0, 2.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_positive(df, "v")

    def test_raises_on_negative_value(self):
        df = pl.DataFrame({"v": [1.0, -0.01]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_positive(df, "v")

    def test_passes_for_all_positive(self):
        df = pl.DataFrame({"v": [0.001, 1.0, 99.9]})
        validate_positive(df, "v")  # Should not raise

    def test_error_includes_min_value(self):
        df = pl.DataFrame({"v": [1.0, 2.0, -5.0]})
        with pytest.raises(ValueError, match="-5"):
            validate_positive(df, "v")


# ---------------------------------------------------------------------------
# validate_binary
# ---------------------------------------------------------------------------


class TestValidateBinary:
    def test_passes_for_zero_and_one(self):
        df = pl.DataFrame({"b": [0, 0, 1, 1, 0, 1]})
        validate_binary(df, "b")  # Should not raise

    def test_passes_for_only_zeros(self):
        df = pl.DataFrame({"b": [0, 0, 0]})
        validate_binary(df, "b")  # All zero is technically valid

    def test_raises_for_value_two(self):
        df = pl.DataFrame({"b": [0, 1, 2]})
        with pytest.raises(ValueError, match="binary"):
            validate_binary(df, "b")

    def test_raises_for_negative_value(self):
        df = pl.DataFrame({"b": [-1, 0, 1]})
        with pytest.raises(ValueError, match="binary"):
            validate_binary(df, "b")


# ---------------------------------------------------------------------------
# resolve_exposure
# ---------------------------------------------------------------------------


class TestResolveExposure:
    def test_returns_column_values_when_present(self):
        df = pl.DataFrame({"exp": [0.25, 0.5, 0.75, 1.0]})
        result = resolve_exposure(df, "exp")
        assert result.to_list() == [0.25, 0.5, 0.75, 1.0]

    def test_returns_ones_when_col_is_none(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = resolve_exposure(df, None)
        assert all(v == 1.0 for v in result.to_list())
        assert len(result) == 3

    def test_returns_ones_when_col_not_in_df(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = resolve_exposure(df, "does_not_exist")
        assert result.to_list() == [1.0, 1.0]

    def test_exposure_col_none_returns_correct_length(self):
        df = pl.DataFrame({"a": list(range(100))})
        result = resolve_exposure(df, None)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# exposure_weighted_mean
# ---------------------------------------------------------------------------


class TestExposureWeightedMean:
    def test_equal_weights_gives_arithmetic_mean(self):
        vals = pl.Series([10.0, 20.0, 30.0])
        wts = pl.Series([1.0, 1.0, 1.0])
        result = exposure_weighted_mean(vals, wts)
        assert abs(result - 20.0) < 1e-10

    def test_skewed_weights_favour_high_weight(self):
        vals = pl.Series([100.0, 200.0])
        wts = pl.Series([9.0, 1.0])  # 90% weight on 100
        result = exposure_weighted_mean(vals, wts)
        assert abs(result - 110.0) < 1e-10

    def test_zero_exposure_returns_nan(self):
        vals = pl.Series([1.0, 2.0])
        wts = pl.Series([0.0, 0.0])
        result = exposure_weighted_mean(vals, wts)
        assert math.isnan(result)

    def test_single_policy(self):
        vals = pl.Series([150.0])
        wts = pl.Series([0.5])
        result = exposure_weighted_mean(vals, wts)
        assert abs(result - 150.0) < 1e-10


# ---------------------------------------------------------------------------
# log_ratio
# ---------------------------------------------------------------------------


class TestLogRatio:
    def test_equal_values_gives_zero(self):
        assert abs(log_ratio(100.0, 100.0)) < 1e-10

    def test_double_gives_log_two(self):
        result = log_ratio(200.0, 100.0)
        assert abs(result - math.log(2.0)) < 1e-10

    def test_half_gives_negative_log_two(self):
        result = log_ratio(100.0, 200.0)
        assert abs(result - (-math.log(2.0))) < 1e-10

    def test_zero_denominator_returns_nan(self):
        result = log_ratio(100.0, 0.0)
        assert math.isnan(result)

    def test_ratio_of_one_tenth_is_negative(self):
        result = log_ratio(10.0, 100.0)
        assert result < 0


# ---------------------------------------------------------------------------
# assign_prediction_deciles
# ---------------------------------------------------------------------------


class TestAssignPredictionDeciles:
    def test_adds_prediction_decile_column(self):
        df = pl.DataFrame({"pred": np.linspace(1.0, 100.0, 100).tolist()})
        result = assign_prediction_deciles(df, "pred", n_deciles=10)
        assert "prediction_decile" in result.columns

    def test_decile_values_within_expected_range(self):
        df = pl.DataFrame({"pred": np.linspace(1.0, 100.0, 200).tolist()})
        result = assign_prediction_deciles(df, "pred", n_deciles=10)
        deciles = result["prediction_decile"].to_list()
        assert min(deciles) >= 1
        assert max(deciles) <= 10

    def test_all_same_predictions_go_to_decile_one(self):
        """When all predictions are identical, all rows should land in decile 1."""
        df = pl.DataFrame({"pred": [250.0] * 50})
        result = assign_prediction_deciles(df, "pred", n_deciles=5)
        assert result["prediction_decile"].unique().to_list() == [1]

    def test_five_decile_split(self):
        n = 100
        df = pl.DataFrame({"pred": np.linspace(1.0, 100.0, n).tolist()})
        result = assign_prediction_deciles(df, "pred", n_deciles=5)
        unique = sorted(result["prediction_decile"].unique().to_list())
        assert min(unique) >= 1
        assert max(unique) <= 5

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError):
            assign_prediction_deciles(df, "nonexistent_pred")

    def test_low_cardinality_predictions_handled(self):
        """Low-cardinality predictions (only 3 distinct values) should not crash."""
        preds = [100.0] * 30 + [200.0] * 30 + [300.0] * 30
        df = pl.DataFrame({"pred": preds})
        result = assign_prediction_deciles(df, "pred", n_deciles=10)
        assert "prediction_decile" in result.columns
        assert len(result) == 90


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_lower_and_upper_tuple(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(100.0, 10.0, 200)
        wts = np.ones(200)
        lo, hi = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=100)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_less_than_upper(self):
        rng = np.random.default_rng(7)
        vals = rng.lognormal(0, 0.5, 300)
        wts = np.ones(300)
        lo, hi = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=100)
        assert lo <= hi

    def test_ci_covers_true_mean(self):
        """95% CI should contain the true mean the vast majority of the time."""
        rng = np.random.default_rng(42)
        vals = rng.normal(50.0, 5.0, 1000)
        wts = np.ones(1000)
        lo, hi = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=300)
        assert lo < 50.0 < hi

    def test_custom_rng_is_reproducible(self):
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        vals = np.arange(100, dtype=float)
        wts = np.ones(100)
        stat = lambda v, w: float(np.mean(v))
        lo_a, hi_a = bootstrap_ci(vals, wts, stat, n_bootstrap=50, rng=rng_a)
        lo_b, hi_b = bootstrap_ci(vals, wts, stat, n_bootstrap=50, rng=rng_b)
        assert abs(lo_a - lo_b) < 1e-10
        assert abs(hi_a - hi_b) < 1e-10

    def test_weighted_statistic(self):
        """Weighted mean statistic should produce valid CI."""
        rng = np.random.default_rng(12)
        vals = rng.normal(200.0, 20.0, 500)
        wts = rng.uniform(0.5, 1.5, 500)

        def _weighted_mean(v, w):
            return float(np.average(v, weights=w))

        lo, hi = bootstrap_ci(vals, wts, _weighted_mean, n_bootstrap=200)
        assert lo < 200.0 < hi


# ---------------------------------------------------------------------------
# rag_status
# ---------------------------------------------------------------------------


class TestRagStatus:
    # Disparate impact ratio (range-based)
    def test_dir_green_at_parity(self):
        assert rag_status("disparate_impact_ratio", 1.0) == "green"

    def test_dir_green_slightly_below_one(self):
        assert rag_status("disparate_impact_ratio", 0.95) == "green"

    def test_dir_green_slightly_above_one(self):
        assert rag_status("disparate_impact_ratio", 1.05) == "green"

    def test_dir_amber_moderate_disparity(self):
        # 0.85 is between amber_lo=0.80 and green_lo=0.90
        assert rag_status("disparate_impact_ratio", 0.85) == "amber"

    def test_dir_amber_on_upper_side(self):
        # 1.20 is between green_hi=1.11 and amber_hi=1.25
        assert rag_status("disparate_impact_ratio", 1.20) == "amber"

    def test_dir_red_below_amber_lo(self):
        assert rag_status("disparate_impact_ratio", 0.75) == "red"

    def test_dir_red_above_amber_hi(self):
        assert rag_status("disparate_impact_ratio", 1.30) == "red"

    # proxy_r2 (scalar threshold)
    def test_proxy_r2_green(self):
        assert rag_status("proxy_r2", 0.01) == "green"

    def test_proxy_r2_amber_at_threshold(self):
        # amber threshold is 0.05; at exactly 0.05 should be amber
        assert rag_status("proxy_r2", 0.05) == "amber"

    def test_proxy_r2_amber_below_red(self):
        assert rag_status("proxy_r2", 0.07) == "amber"

    def test_proxy_r2_red(self):
        assert rag_status("proxy_r2", 0.15) == "red"

    # calibration_disparity (scalar threshold)
    def test_calibration_disparity_green(self):
        assert rag_status("calibration_disparity", 0.05) == "green"

    def test_calibration_disparity_amber(self):
        assert rag_status("calibration_disparity", 0.12) == "amber"

    def test_calibration_disparity_red(self):
        assert rag_status("calibration_disparity", 0.25) == "red"

    # demographic_parity_log_ratio (scalar threshold)
    def test_demographic_parity_green(self):
        assert rag_status("demographic_parity_log_ratio", 0.02) == "green"

    def test_demographic_parity_amber(self):
        assert rag_status("demographic_parity_log_ratio", 0.07) == "amber"

    def test_demographic_parity_red(self):
        assert rag_status("demographic_parity_log_ratio", 0.15) == "red"

    # Unknown metric name
    def test_unknown_metric_returns_unknown(self):
        assert rag_status("completely_made_up_metric", 0.5) == "unknown"

    def test_custom_thresholds_override_defaults(self):
        """Passing custom thresholds should bypass DEFAULT_THRESHOLDS."""
        custom = {"amber": 0.20, "red": 0.40}
        assert rag_status("proxy_r2", 0.15, thresholds=custom) == "green"
        assert rag_status("proxy_r2", 0.25, thresholds=custom) == "amber"
        assert rag_status("proxy_r2", 0.45, thresholds=custom) == "red"


# ---------------------------------------------------------------------------
# DEFAULT_THRESHOLDS constant
# ---------------------------------------------------------------------------


class TestDefaultThresholds:
    def test_all_expected_keys_present(self):
        expected_keys = {
            "disparate_impact_ratio",
            "proxy_r2",
            "calibration_disparity",
            "demographic_parity_log_ratio",
        }
        assert expected_keys.issubset(set(DEFAULT_THRESHOLDS.keys()))

    def test_disparate_impact_has_green_and_amber(self):
        t = DEFAULT_THRESHOLDS["disparate_impact_ratio"]
        assert "green" in t
        assert "amber" in t

    def test_proxy_r2_has_amber_and_red(self):
        t = DEFAULT_THRESHOLDS["proxy_r2"]
        assert "amber" in t
        assert "red" in t
        assert t["amber"] < t["red"]
