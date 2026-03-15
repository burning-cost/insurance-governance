"""
Tests for insurance_fairness.audit (FairnessAudit, FairnessReport).

These tests exercise the high-level audit orchestration from the insurance-fairness
package. The governance library wraps and depends on this package — covering it here
verifies the dependency chain behaves correctly end-to-end.

Focus areas:
  - FairnessAudit.run() end-to-end with no model (proxy detection disabled)
  - FairnessReport structure and to_dict() serialisation
  - Summary text generation
  - Edge cases: no exposure, multi-group protected characteristics
  - RAG status propagation
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from insurance_fairness.audit import FairnessAudit, FairnessReport, ProtectedCharacteristicReport


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="module")
def binary_policy_df(rng) -> pl.DataFrame:
    """
    1000-policy dataset with a binary protected characteristic.

    Designed to trigger a red demographic parity status (group 1 has ~30%
    higher premiums on average) while keeping calibration broadly intact.
    """
    n = 1000
    group = np.array([0] * 500 + [1] * 500, dtype=np.int32)
    base = rng.lognormal(4.6, 0.3, n)
    pred = base * np.where(group == 1, 1.3, 1.0)
    actual = pred * rng.lognormal(0.0, 0.2, n)
    exposure = rng.uniform(0.5, 1.0, n)
    vehicle_age = rng.integers(0, 20, n)
    ncd = rng.integers(0, 10, n)
    return pl.DataFrame({
        "gender": group,
        "predicted_premium": pred.tolist(),
        "claim_amount": actual.tolist(),
        "exposure": exposure.tolist(),
        "vehicle_age": vehicle_age.tolist(),
        "ncd_years": ncd.tolist(),
    })


@pytest.fixture(scope="module")
def multi_group_policy_df(rng) -> pl.DataFrame:
    """900-policy dataset with a 3-category protected characteristic."""
    n_per = 300
    groups = ["A"] * n_per + ["B"] * n_per + ["C"] * n_per
    means = {"A": 100.0, "B": 120.0, "C": 90.0}
    preds, actuals = [], []
    for g in groups:
        m = means[g]
        p = float(rng.lognormal(np.log(m), 0.25))
        preds.append(p)
        actuals.append(float(p * rng.lognormal(0.0, 0.2)))
    exposure = rng.uniform(0.5, 1.0, len(groups))
    return pl.DataFrame({
        "region": groups,
        "predicted_premium": preds,
        "claim_amount": actuals,
        "exposure": exposure.tolist(),
        "vehicle_age": rng.integers(1, 15, len(groups)).tolist(),
    })


# ---------------------------------------------------------------------------
# FairnessAudit construction
# ---------------------------------------------------------------------------


class TestFairnessAuditConstruction:
    def test_constructs_without_error(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        assert audit.protected_cols == ["gender"]

    def test_raises_on_missing_required_column(self):
        df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            FairnessAudit(
                model=None,
                data=df,
                protected_cols=["gender"],
                prediction_col="pred",
                outcome_col="actual",
            )

    def test_auto_detects_factor_cols_when_none_provided(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            factor_cols=None,
            run_proxy_detection=False,
        )
        assert "vehicle_age" in audit.factor_cols
        assert "ncd_years" in audit.factor_cols
        assert "gender" not in audit.factor_cols
        assert "predicted_premium" not in audit.factor_cols
        assert "claim_amount" not in audit.factor_cols
        assert "exposure" not in audit.factor_cols

    def test_explicit_factor_cols_used(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            factor_cols=["vehicle_age"],
            run_proxy_detection=False,
        )
        assert audit.factor_cols == ["vehicle_age"]

    def test_accepts_pandas_dataframe(self, binary_policy_df):
        pandas_df = binary_policy_df.to_pandas()
        audit = FairnessAudit(
            model=None,
            data=pandas_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        assert isinstance(audit.data, pl.DataFrame)

    def test_multiple_protected_cols_accepted(self, binary_policy_df):
        df = binary_policy_df.with_columns(
            (pl.col("vehicle_age") > 10).cast(pl.Int32).alias("age_group")
        )
        audit = FairnessAudit(
            model=None,
            data=df,
            protected_cols=["gender", "age_group"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        assert "gender" in audit.protected_cols
        assert "age_group" in audit.protected_cols


# ---------------------------------------------------------------------------
# FairnessAudit.run()
# ---------------------------------------------------------------------------


class TestFairnessAuditRun:
    def test_run_returns_fairness_report(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert isinstance(report, FairnessReport)

    def test_report_has_correct_n_policies(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.n_policies == len(binary_policy_df)

    def test_report_has_positive_total_exposure(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.total_exposure > 0.0

    def test_results_keyed_by_protected_col(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert "gender" in report.results

    def test_demographic_parity_populated(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.results["gender"].demographic_parity is not None

    def test_calibration_populated(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.results["gender"].calibration is not None

    def test_disparate_impact_populated(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.results["gender"].disparate_impact is not None

    def test_gini_populated(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.results["gender"].gini is not None

    def test_known_disparity_is_red(self, binary_policy_df):
        """30% premium uplift for group 1 should give an overall red status."""
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.results["gender"].demographic_parity.rag == "red"
        assert report.overall_rag == "red"

    def test_run_without_exposure_col(self, binary_policy_df):
        """Audit should complete cleanly with no exposure column."""
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col=None,
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.total_exposure == float(len(binary_policy_df))

    def test_multi_group_protected_characteristic(self, multi_group_policy_df):
        audit = FairnessAudit(
            model=None,
            data=multi_group_policy_df,
            protected_cols=["region"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert "region" in report.results
        dp = report.results["region"].demographic_parity
        assert "A" in dp.group_means
        assert "B" in dp.group_means
        assert "C" in dp.group_means

    def test_multiple_protected_cols_both_in_results(self, binary_policy_df):
        df = binary_policy_df.with_columns(
            (pl.col("vehicle_age") > 10).cast(pl.Int32).alias("age_group")
        )
        audit = FairnessAudit(
            model=None,
            data=df,
            protected_cols=["gender", "age_group"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert "gender" in report.results
        assert "age_group" in report.results

    def test_audit_date_is_iso_date_string(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        # ISO date: YYYY-MM-DD
        assert len(report.audit_date) == 10
        assert report.audit_date[4] == "-"
        assert report.audit_date[7] == "-"

    def test_flagged_factors_is_list(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert isinstance(report.flagged_factors, list)

    def test_custom_model_name_in_report(self, binary_policy_df):
        audit = FairnessAudit(
            model=None,
            data=binary_policy_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            model_name="Motor Frequency v2",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert report.model_name == "Motor Frequency v2"


# ---------------------------------------------------------------------------
# FairnessReport
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_report(binary_policy_df) -> FairnessReport:
    """A fully-run FairnessReport for use in report-level tests."""
    audit = FairnessAudit(
        model=None,
        data=binary_policy_df,
        protected_cols=["gender"],
        prediction_col="predicted_premium",
        outcome_col="claim_amount",
        exposure_col="exposure",
        run_proxy_detection=False,
    )
    return audit.run()


class TestFairnessReport:
    def test_summary_prints_model_name(self, sample_report, capsys):
        sample_report.summary()
        captured = capsys.readouterr()
        assert "Fairness Audit" in captured.out

    def test_summary_contains_protected_col(self, sample_report, capsys):
        sample_report.summary()
        captured = capsys.readouterr()
        assert "gender" in captured.out

    def test_summary_contains_overall_rag(self, sample_report, capsys):
        sample_report.summary()
        captured = capsys.readouterr()
        assert any(s in captured.out for s in ("GREEN", "AMBER", "RED"))

    def test_to_dict_is_json_serialisable(self, sample_report):
        d = sample_report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["protected_cols"] == ["gender"]

    def test_to_dict_has_expected_top_level_keys(self, sample_report):
        d = sample_report.to_dict()
        for key in ("model_name", "audit_date", "overall_rag", "results",
                    "n_policies", "total_exposure", "flagged_factors"):
            assert key in d

    def test_to_dict_results_contain_demographic_parity(self, sample_report):
        d = sample_report.to_dict()
        gender_dict = d["results"]["gender"]
        assert "demographic_parity" in gender_dict
        dp = gender_dict["demographic_parity"]
        assert "log_ratio" in dp
        assert "ratio" in dp
        assert "rag" in dp

    def test_to_dict_results_contain_calibration(self, sample_report):
        d = sample_report.to_dict()
        gender_dict = d["results"]["gender"]
        assert "calibration" in gender_dict
        cal = gender_dict["calibration"]
        assert "max_disparity" in cal
        assert "rag" in cal

    def test_to_dict_results_contain_disparate_impact(self, sample_report):
        d = sample_report.to_dict()
        gender_dict = d["results"]["gender"]
        assert "disparate_impact" in gender_dict

    def test_overall_rag_is_valid_value(self, sample_report):
        assert sample_report.overall_rag in ("green", "amber", "red")

    def test_protected_col_report_type(self, sample_report):
        assert isinstance(sample_report.results["gender"], ProtectedCharacteristicReport)

    def test_pareto_result_none_when_not_run(self, sample_report):
        assert sample_report.pareto_result is None


class TestFairnessReportSummaryEdgeCases:
    def test_summary_with_flagged_factors(self, binary_policy_df):
        """A report with flagged_factors should include them in the summary text."""
        from insurance_fairness.audit import FairnessReport, ProtectedCharacteristicReport
        from insurance_fairness.bias_metrics import demographic_parity_ratio, calibration_by_group, disparate_impact_ratio, gini_by_group

        # Build a minimal report manually to control flagged_factors
        dp = demographic_parity_ratio(binary_policy_df, "gender", "predicted_premium")
        cal = calibration_by_group(binary_policy_df, "gender", "predicted_premium", "claim_amount")
        di = disparate_impact_ratio(binary_policy_df, "gender", "predicted_premium")
        gini = gini_by_group(binary_policy_df, "gender", "predicted_premium")

        pc_report = ProtectedCharacteristicReport(
            protected_col="gender",
            demographic_parity=dp,
            calibration=cal,
            disparate_impact=di,
            gini=gini,
        )

        report = FairnessReport(
            model_name="Test Model",
            audit_date="2024-01-01",
            protected_cols=["gender"],
            factor_cols=["vehicle_age"],
            n_policies=1000,
            total_exposure=750.0,
            results={"gender": pc_report},
            flagged_factors=["postcode_district", "vehicle_group"],
            overall_rag="amber",
        )

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        report.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "postcode_district" in output or "vehicle_group" in output

    def test_summary_no_flagged_factors_message(self, binary_policy_df):
        """With no flagged_factors, summary should confirm no concerns."""
        from insurance_fairness.audit import FairnessReport, ProtectedCharacteristicReport
        from insurance_fairness.bias_metrics import demographic_parity_ratio, calibration_by_group, disparate_impact_ratio, gini_by_group

        dp = demographic_parity_ratio(binary_policy_df, "gender", "predicted_premium")
        cal = calibration_by_group(binary_policy_df, "gender", "predicted_premium", "claim_amount")
        di = disparate_impact_ratio(binary_policy_df, "gender", "predicted_premium")
        gini = gini_by_group(binary_policy_df, "gender", "predicted_premium")

        pc_report = ProtectedCharacteristicReport(
            protected_col="gender",
            demographic_parity=dp,
            calibration=cal,
            disparate_impact=di,
            gini=gini,
        )
        report = FairnessReport(
            model_name="Clean Model",
            audit_date="2024-06-01",
            protected_cols=["gender"],
            factor_cols=[],
            n_policies=500,
            total_exposure=400.0,
            results={"gender": pc_report},
            flagged_factors=[],
            overall_rag="green",
        )

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        report.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "No rating factors flagged" in output
