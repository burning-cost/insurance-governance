"""Tests for OutcomeTestingFramework."""
import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from insurance_governance.mrm.model_card import ModelCard as MRMModelCard
from insurance_governance.outcome import (
    CustomerSegment,
    OutcomeTestingFramework,
    OutcomeResult,
)
from insurance_governance.validation.results import RAGStatus, Severity


def make_card():
    return MRMModelCard(
        model_id="motor-freq-v3",
        model_name="Motor Frequency v3",
        version="3.0.0",
        model_class="pricing",
    )


def make_policy_df(n=100, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "gross_premium": rng.uniform(200, 500, n).tolist(),
        "claims_paid": rng.uniform(100, 400, n).tolist(),
        "is_renewal": (rng.random(n) > 0.5).astype(int).tolist(),
        "days_to_settlement": rng.uniform(1, 15, n).tolist(),
        "claim_outcome": (rng.random(n) > 0.85).astype(int).tolist(),
        "expenses": (rng.uniform(10, 50, n)).tolist(),
        "age": rng.integers(20, 80, n).tolist(),
        "policy_type": ["renewal" if i % 2 == 0 else "new" for i in range(n)],
    })


# --- Construction and basic run ---

def test_framework_run_returns_list():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
    )
    results = fw.run()
    assert isinstance(results, list)


def test_framework_run_cached():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card, policy_data=df, period="2025-Q4", price_col="gross_premium"
    )
    r1 = fw.run()
    r2 = fw.run()
    assert r1 is r2  # Same object — cached


def test_framework_no_claims_col_skips_fair_value():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card, policy_data=df, period="2025-Q4", price_col="gross_premium"
    )
    results = fw.run()
    test_names = [r.test_name for r in results]
    assert "fair_value_ratio" not in test_names


def test_framework_with_claims_col_includes_fair_value():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
    )
    results = fw.run()
    test_names = [r.test_name for r in results]
    assert "fair_value_ratio" in test_names


def test_framework_renewal_gap_included():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        renewal_indicator_col="is_renewal",
    )
    results = fw.run()
    test_names = [r.test_name for r in results]
    assert "renewal_vs_new_business_gap" in test_names


def test_framework_timeliness_included():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        days_to_settlement_col="days_to_settlement",
    )
    results = fw.run()
    test_names = [r.test_name for r in results]
    assert "timeliness_sla" in test_names


def test_framework_with_segments_produces_segment_results():
    card = make_card()
    df = make_policy_df()
    renewal_seg = CustomerSegment(
        name="Renewal",
        filter_fn=lambda df: df["policy_type"] == "renewal",
    )
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
        customer_segments=[renewal_seg],
    )
    results = fw.run()
    segment_results = [r for r in results if r.segment == "Renewal"]
    assert len(segment_results) > 0


def test_framework_extra_results_included():
    card = make_card()
    df = make_policy_df()
    extra = OutcomeResult(
        outcome="support",
        test_name="complaints_rate",
        passed=True,
        metric_value=0.01,
        period="2025-Q4",
        details="Manual: complaints rate 1%.",
    )
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        extra_results=[extra],
    )
    results = fw.run()
    extras = [r for r in results if r.test_name == "complaints_rate"]
    assert len(extras) == 1


def test_framework_rag_status_returns_enum():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card, policy_data=df, period="2025-Q4", price_col="gross_premium"
    )
    rag = fw.get_rag_status()
    assert isinstance(rag, RAGStatus)


def test_framework_to_dict_structure():
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
    )
    d = fw.to_dict()
    assert "period" in d
    assert "rag_status" in d
    assert "results" in d
    assert "summary" in d
    assert "model_card" in d


def test_framework_generate_html(tmp_path):
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
    )
    out = fw.generate(tmp_path / "outcome.html")
    assert out.exists()
    html = out.read_text()
    assert "Consumer Duty" in html
    assert "Motor Frequency" in html


def test_framework_to_json(tmp_path):
    card = make_card()
    df = make_policy_df()
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
    )
    out = fw.to_json(tmp_path / "outcome.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "period" in data
    assert data["period"] == "2025-Q4"


def test_framework_empty_segment_skipped():
    card = make_card()
    df = make_policy_df()
    nobody = CustomerSegment(
        name="Nobody",
        filter_fn=lambda df: pl.Series([False] * df.height),
    )
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
        customer_segments=[nobody],
    )
    results = fw.run()
    seg_results = [r for r in results if r.segment == "Nobody"]
    assert len(seg_results) == 0


def test_framework_all_columns_configured():
    """Smoke test: all columns configured, verify no crash and dict output."""
    card = make_card()
    df = make_policy_df()
    renewal_seg = CustomerSegment(
        name="Renewal",
        filter_fn=lambda df: df["is_renewal"] == 1,
    )
    fw = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
        claim_outcome_col="claim_outcome",
        days_to_settlement_col="days_to_settlement",
        expenses_col="expenses",
        renewal_indicator_col="is_renewal",
        customer_segments=[renewal_seg],
    )
    results = fw.run()
    assert len(results) > 0
    d = fw.to_dict()
    assert d["summary"]["total_tests"] > 0
