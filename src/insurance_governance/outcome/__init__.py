"""
insurance_governance.outcome — FCA Consumer Duty outcome testing framework.

Tests whether a pricing model and its associated claims handling are delivering
good outcomes for customers, as required under FCA PRIN 2A (Consumer Duty).

The four Consumer Duty outcomes are:
1. Products and services
2. Price and value
3. Consumer understanding
4. Consumer support

This subpackage implements outcome testing for price/value (outcome 2) and
claims handling (part of outcome 4). Consumer understanding tests require
qualitative assessment and are not automated here.

Quick start::

    import polars as pl
    from insurance_governance import MRMModelCard
    from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment

    card = MRMModelCard(
        model_id="motor-freq-v3",
        model_name="Motor Frequency v3",
        version="3.0.0",
    )
    df = pl.read_parquet("policies_2025q4.parquet")

    renewal_seg = CustomerSegment(
        name="Renewal",
        filter_fn=lambda df: df["policy_type"] == "renewal",
    )
    vulnerable_seg = CustomerSegment(
        name="Older Customers (65+)",
        filter_fn=lambda df: df["age"] >= 65,
        is_vulnerable=True,
    )

    framework = OutcomeTestingFramework(
        model_card=card,
        policy_data=df,
        period="2025-Q4",
        price_col="gross_premium",
        claim_amount_col="claims_paid",
        renewal_indicator_col="is_renewal",
        customer_segments=[renewal_seg, vulnerable_seg],
    )

    results = framework.run()
    print(framework.get_rag_status())
    framework.generate("outcome_report.html")
    framework.to_json("outcome_report.json")
"""
from .framework import OutcomeTestingFramework
from .metrics import ClaimsMetrics, PriceValueMetrics
from .report import OutcomeTestingReport
from .results import OutcomeResult, OutcomeSuite
from .segments import CustomerSegment, SegmentComparison

__all__ = [
    # Facade
    "OutcomeTestingFramework",
    # Metrics
    "PriceValueMetrics",
    "ClaimsMetrics",
    # Report
    "OutcomeTestingReport",
    # Results
    "OutcomeResult",
    "OutcomeSuite",
    # Segments
    "CustomerSegment",
    "SegmentComparison",
]
