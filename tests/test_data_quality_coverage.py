"""
Supplemental tests for DataQualityReport covering remaining edge cases.

The existing test_data_quality.py covers the main paths. This file targets:
  - The `df` property getter
  - The "within threshold" missing value branch (rate > 0, rate <= threshold)
  - The empty column edge case in outlier_detection (all values null)
  - The z-score method with zero standard deviation (all-constant column)
"""

from __future__ import annotations

import polars as pl
import pytest

from insurance_governance.validation import DataQualityReport
from insurance_governance.validation.results import Severity, TestCategory


# ---------------------------------------------------------------------------
# df property
# ---------------------------------------------------------------------------


def test_df_property_returns_original_dataframe():
    """The `df` property should return the underlying DataFrame."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    dqr = DataQualityReport(df, dataset_name="test")
    result = dqr.df
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 2)
    assert result is df  # Same object, not a copy


# ---------------------------------------------------------------------------
# missing_value_analysis — within-threshold branch
# ---------------------------------------------------------------------------


def test_missing_value_below_threshold_passes_with_info():
    """
    A column with missing rate > 0 but <= threshold should pass with INFO.

    With threshold=0.5 and 1 null in 10 rows (10%), the result should pass.
    """
    data = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 10% missing
    df = pl.DataFrame({"premium": data})
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis(threshold=0.5)  # threshold 50%
    assert len(results) == 1
    r = results[0]
    assert r.passed is True
    assert r.severity == Severity.INFO
    # Rate is 10%, which is > 0 but <= 50% threshold
    assert abs(r.metric_value - 0.1) < 0.001


def test_missing_value_exactly_at_threshold_passes():
    """A column with missing rate equal to the threshold should pass."""
    # 1 null in 10 rows = 10%; threshold = 0.10
    data = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"x": data})
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis(threshold=0.10)
    r = results[0]
    assert r.passed is True


def test_missing_value_just_above_threshold_fails_warning():
    """A column with missing rate just above threshold should fail with WARNING."""
    # 3 nulls in 10 rows = 30%; threshold = 0.20 -> WARNING (rate <= 0.5)
    data = [1.0, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"x": data})
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis(threshold=0.20)
    r = results[0]
    assert r.passed is False
    assert r.severity == Severity.WARNING


# ---------------------------------------------------------------------------
# outlier_detection — empty column edge case
# ---------------------------------------------------------------------------


def test_outlier_detection_all_nulls_column_skipped():
    """
    A column that is entirely null should be skipped (len(series)==0 after drop_nulls).

    The test verifies no crash and no spurious result for the all-null column,
    but a valid result for the non-null numeric column.
    """
    df = pl.DataFrame({
        "all_null": pl.Series([None, None, None, None, None], dtype=pl.Float64),
        "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    dqr = DataQualityReport(df)
    # only 2 null values in all_null (3 rows) — after drop_nulls: 0 rows, should skip
    # The all_null col has 3 rows; after drop_nulls, n_total=0 -> continue
    results = dqr.outlier_detection(method="iqr")
    # Only the 'normal' column should produce a result
    names = [r.test_name for r in results]
    assert any("normal" in n for n in names)
    assert all("all_null" not in n for n in names)


def test_outlier_detection_single_null_column_in_mixed_df():
    """Ensure only numeric columns with data produce results."""
    df = pl.DataFrame({
        "all_null_float": pl.Series([None, None, None], dtype=pl.Float64),
        "fine_col": [10.0, 20.0, 30.0],
    })
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="iqr")
    # all_null_float: 2 rows all null -> 0 after drop_nulls -> skipped
    # fine_col: 3 rows -> processed
    result_names = [r.test_name for r in results]
    assert any("fine_col" in n for n in result_names)
    assert all("all_null_float" not in n for n in result_names)


# ---------------------------------------------------------------------------
# outlier_detection — z-score with zero standard deviation
# ---------------------------------------------------------------------------


def test_outlier_detection_zscore_constant_column():
    """
    A column where all values are identical has std=0.
    The z-score method should return 0 outliers (not crash with divide-by-zero).
    """
    df = pl.DataFrame({"constant": [100.0] * 20})
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="zscore", zscore_threshold=3.0)
    assert len(results) == 1
    r = results[0]
    assert r.passed is True
    assert r.metric_value == 0.0


def test_outlier_detection_zscore_near_constant_column():
    """Only extreme outliers are flagged; a near-constant column is mostly fine."""
    values = [100.0] * 99 + [1000.0]  # one extreme outlier
    df = pl.DataFrame({"x": values})
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="zscore", zscore_threshold=3.0)
    assert len(results) == 1
    # 1 outlier in 100 values = 1% — should fail but not crash
    assert results[0].passed is False


# ---------------------------------------------------------------------------
# dataset_name propagation
# ---------------------------------------------------------------------------


def test_summary_statistics_includes_dataset_name():
    """Dataset name should appear in the summary details."""
    df = pl.DataFrame({"x": [1, 2, 3]})
    dqr = DataQualityReport(df, dataset_name="validation_set_2024")
    result = dqr.summary_statistics()
    assert "validation_set_2024" in result.details
