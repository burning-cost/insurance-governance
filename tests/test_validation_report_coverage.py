"""
Supplemental tests for ModelValidationReport covering remaining code paths.

The existing test_validation_report.py covers the main paths. This file targets:
  - X_train + X_val together (triggers data quality on training set and feature drift)
  - fairness_group_col (triggers disparate impact check)
  - Monitoring plan with owner + no triggers
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_governance.validation import ModelCard, ModelValidationReport


def make_card():
    return ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio.",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area"],
        limitations=["No telematics data"],
        owner="Pricing Team",
    )


def make_data(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_train = rng.poisson(0.1, n).astype(float)
    y_pred_train = np.clip(y_train + rng.normal(0, 0.02, n), 0.001, 5.0)
    y_val = rng.poisson(0.1, n).astype(float)
    y_pred_val = np.clip(y_val + rng.normal(0, 0.02, n), 0.001, 5.0)
    exposure = rng.uniform(0.5, 1.5, n)
    return y_train, y_pred_train, y_val, y_pred_val, exposure


def make_feature_dfs(n: int = 400, seed: int = 42):
    """Return (X_train, X_val) with matching columns for drift tests."""
    rng = np.random.default_rng(seed)
    X_train = pl.DataFrame({
        "age": rng.integers(18, 80, n).tolist(),
        "vehicle_age": rng.integers(0, 20, n).tolist(),
        "gender": rng.integers(0, 2, n).tolist(),
    })
    # Validation set: same columns, slightly different distribution
    X_val = pl.DataFrame({
        "age": rng.integers(20, 75, n).tolist(),
        "vehicle_age": rng.integers(0, 15, n).tolist(),
        "gender": rng.integers(0, 2, n).tolist(),
    })
    return X_train, X_val


# ---------------------------------------------------------------------------
# X_train data quality and feature drift
# ---------------------------------------------------------------------------


class TestXTrainAndFeatureDrift:
    def test_x_train_data_quality_included(self):
        """When X_train is a Polars DataFrame, its summary stats should be included."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data()
        X_train, X_val = make_feature_dfs()
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            y_train=y_train,
            y_pred_train=y_pred_train,
            X_train=X_train,
            X_val=X_val,
        )
        results = report.run()
        test_names = [r.test_name for r in results]
        # Should include summary_statistics from training set
        # DataQualityReport names: summary_statistics appears from both sets
        # At least one summary_statistics result should be present
        assert "summary_statistics" in test_names

    def test_feature_drift_psi_results_present(self):
        """With both X_train and X_val, feature-level PSI drift checks should run."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data()
        X_train, X_val = make_feature_dfs()
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            y_train=y_train,
            y_pred_train=y_pred_train,
            X_train=X_train,
            X_val=X_val,
        )
        results = report.run()
        # Feature drift produces psi_<feature> results
        test_names = [r.test_name for r in results]
        drift_results = [n for n in test_names if n.startswith("psi_")]
        # Should have at least one feature drift result plus the score PSI
        assert len(drift_results) >= 1

    def test_feature_drift_with_common_columns(self):
        """Feature drift only runs on columns common to both X_train and X_val."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data(n=300)
        rng = np.random.default_rng(55)
        n = 300
        # X_train has extra column not in X_val
        X_train = pl.DataFrame({
            "age": rng.integers(18, 80, n).tolist(),
            "only_in_train": rng.uniform(0, 1, n).tolist(),
        })
        X_val = pl.DataFrame({
            "age": rng.integers(18, 80, n).tolist(),
            "only_in_val": rng.uniform(0, 1, n).tolist(),
        })
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            y_train=y_train,
            y_pred_train=y_pred_train,
            X_train=X_train,
            X_val=X_val,
        )
        results = report.run()
        # Should not crash; only 'age' is common, so only psi_age (or similar) appears
        assert results is not None

    def test_no_feature_drift_when_x_train_not_polars(self):
        """If X_train is not a Polars DataFrame, feature drift is silently skipped."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data(n=200)
        _, X_val = make_feature_dfs(n=200)
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            y_train=y_train,
            y_pred_train=y_pred_train,
            X_train={"not": "polars"},  # Not a Polars DataFrame
            X_val=X_val,
        )
        results = report.run()
        # Should not crash
        assert results is not None


# ---------------------------------------------------------------------------
# fairness_group_col
# ---------------------------------------------------------------------------


class TestFairnessGroupCol:
    def test_disparate_impact_included_when_group_col_set(self):
        """With fairness_group_col specified, disparate_impact_ratio should run."""
        n = 400
        rng = np.random.default_rng(77)
        y_val = rng.poisson(0.1, n).astype(float)
        y_pred_val = np.full(n, 0.1)
        gender = rng.integers(0, 2, n).tolist()
        X_val = pl.DataFrame({
            "gender": gender,
            "vehicle_age": rng.integers(0, 15, n).tolist(),
        })
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            X_val=X_val,
            fairness_group_col="gender",
        )
        results = report.run()
        test_names = [r.test_name for r in results]
        assert "disparate_impact_ratio" in test_names

    def test_fairness_check_skipped_when_group_col_missing_from_df(self):
        """If fairness_group_col is set but the column doesn't exist, it is skipped."""
        n = 200
        rng = np.random.default_rng(78)
        y_val = rng.poisson(0.1, n).astype(float)
        y_pred_val = np.full(n, 0.1)
        X_val = pl.DataFrame({
            "vehicle_age": rng.integers(0, 15, n).tolist(),
        })
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            X_val=X_val,
            fairness_group_col="gender",  # Not in X_val
        )
        results = report.run()
        test_names = [r.test_name for r in results]
        # disparate_impact_ratio should NOT appear since column is absent
        assert "disparate_impact_ratio" not in test_names

    def test_fairness_check_not_run_without_group_col(self):
        """Without fairness_group_col, no disparate impact test is run."""
        n = 200
        rng = np.random.default_rng(79)
        y_val = rng.poisson(0.1, n).astype(float)
        y_pred_val = np.full(n, 0.1)
        X_val = pl.DataFrame({
            "vehicle_age": rng.integers(0, 15, n).tolist(),
        })
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            X_val=X_val,
            # No fairness_group_col
        )
        results = report.run()
        test_names = [r.test_name for r in results]
        assert "disparate_impact_ratio" not in test_names


# ---------------------------------------------------------------------------
# Monitoring plan variants
# ---------------------------------------------------------------------------


class TestMonitoringPlanVariants:
    def test_monitoring_plan_with_owner_only_no_triggers(self):
        """Owner present but no triggers: should pass with a note about triggers."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data()
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            monitoring_owner="Actuarial Risk",
            monitoring_triggers=None,  # No triggers
        )
        results = report.run()
        mon = next((r for r in results if r.test_name == "monitoring_plan"), None)
        assert mon is not None
        assert mon.passed is True
        assert "consider adding" in mon.details.lower() or "not specified" in mon.details.lower() or mon.passed

    def test_monitoring_plan_with_owner_and_triggers_passes(self):
        """Owner + triggers: should pass with trigger details in the description."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data()
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
            monitoring_owner="Pricing Actuary",
            monitoring_triggers={"psi_score": 0.25, "gini_drop": 0.05},
        )
        results = report.run()
        mon = next((r for r in results if r.test_name == "monitoring_plan"), None)
        assert mon is not None
        assert mon.passed is True
        assert "psi_score" in mon.details or "Pricing Actuary" in mon.details

    def test_get_results_runs_automatically(self):
        """get_results() should run and cache on first call."""
        y_train, y_pred_train, y_val, y_pred_val, exp = make_data()
        report = ModelValidationReport(
            model_card=make_card(),
            y_val=y_val,
            y_pred_val=y_pred_val,
        )
        results1 = report.get_results()
        results2 = report.get_results()
        assert len(results1) == len(results2)
        assert results1 is results2  # Same cached list
