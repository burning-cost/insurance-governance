# Databricks notebook source
# MAGIC %md
# MAGIC # Consumer Duty Outcome Testing — Demo Notebook
# MAGIC
# MAGIC This notebook demonstrates the `insurance_governance.outcome` subpackage
# MAGIC against synthetic UK motor insurance data.
# MAGIC
# MAGIC **FCA Consumer Duty context**: Under PRIN 2A, firms must monitor and
# MAGIC evidence that customers receive good outcomes across four areas:
# MAGIC - Products and services
# MAGIC - **Price and value** (covered here)
# MAGIC - Consumer understanding
# MAGIC - **Consumer support / claims** (covered here)
# MAGIC
# MAGIC The outcome testing framework produces board-ready HTML reports with RAG
# MAGIC status, corrective action lists, and segment breakdowns for vulnerable
# MAGIC customer groups.

# COMMAND ----------

# MAGIC %pip install insurance-governance>=0.2.0 --quiet

# COMMAND ----------

import polars as pl
import numpy as np
from pathlib import Path

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic policy data
# MAGIC
# MAGIC Realistic UK motor book: 5,000 policies, mix of new business and renewal,
# MAGIC with deliberate price-walking on renewals to trigger the GIPP test.

# COMMAND ----------

rng = np.random.default_rng(2025)
N = 5_000

# Policy-level data
ages = rng.integers(18, 80, N)
is_renewal = (rng.random(N) > 0.45).astype(int)
policy_type = np.where(is_renewal == 1, "renewal", "new_business")

# New business base premium: roughly risk-rated
nb_premium = 150 + (ages / 80) * 300 + rng.normal(0, 30, N)
nb_premium = np.clip(nb_premium, 120, 600)

# Renewals: deliberately priced ~12% above NB equivalent (GIPP violation)
renewal_premium = nb_premium * np.where(is_renewal == 1, 1.12, 1.0)
renewal_premium = np.clip(renewal_premium + rng.normal(0, 15, N), 100, 700)

premiums = renewal_premium  # final gross premium

# Claims: Poisson frequency ~0.07, gamma severity
claim_frequency = rng.poisson(0.07, N)
claim_severity = rng.gamma(shape=2.0, scale=1500, size=N)
claims_paid = (claim_frequency > 0) * claim_severity * rng.uniform(0.85, 1.15, N)
claims_paid = np.clip(claims_paid, 0, None)

# Claims processing: mostly within 5 days for standard, slower for older customers
days_base = rng.exponential(4, N)
days_base = np.where(ages >= 65, days_base * 1.8, days_base)  # slower for older
days_to_settlement = np.clip(days_base, 0.5, 60)

# Claim outcomes: slightly higher decline rate for older customers
decline_prob = np.where(ages >= 65, 0.12, 0.08)
claim_outcomes = (rng.random(N) < decline_prob).astype(int)

# Expenses: ~18% of premium
expenses = premiums * rng.uniform(0.15, 0.22, N)

df = pl.DataFrame({
    "age": ages.tolist(),
    "is_renewal": is_renewal.tolist(),
    "policy_type": policy_type.tolist(),
    "gross_premium": premiums.tolist(),
    "claims_paid": claims_paid.tolist(),
    "days_to_settlement": days_to_settlement.tolist(),
    "claim_outcome": claim_outcomes.tolist(),
    "expenses": expenses.tolist(),
})

print(f"Policy data shape: {df.shape}")
print(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define customer segments
# MAGIC
# MAGIC We define two segments:
# MAGIC - **Renewal** — policyholders on their second or subsequent term
# MAGIC - **Older Customers (65+)** — potentially vulnerable, higher scrutiny

# COMMAND ----------

from insurance_governance.outcome import CustomerSegment

renewal_seg = CustomerSegment(
    name="Renewal",
    filter_fn=lambda df: df["policy_type"] == "renewal",
    is_vulnerable=False,
)

older_seg = CustomerSegment(
    name="Older Customers (65+)",
    filter_fn=lambda df: df["age"] >= 65,
    is_vulnerable=True,
)

print(f"Renewal segment: {renewal_seg.count(df)} policies")
print(f"Older customers: {older_seg.count(df)} policies")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run individual metrics directly
# MAGIC
# MAGIC You can run metrics standalone if you only need specific tests.

# COMMAND ----------

from insurance_governance.outcome import PriceValueMetrics, ClaimsMetrics

# Fair value ratio — portfolio-wide
fvr = PriceValueMetrics.fair_value_ratio(
    premiums=df["gross_premium"].to_list(),
    claims_paid=df["claims_paid"].to_list(),
    expenses=df["expenses"].to_list(),
    period="2025-Q4",
)
print(f"Fair value ratio: {fvr.metric_value:.3f} (threshold: {fvr.threshold:.2f}) — {'PASS' if fvr.passed else 'FAIL'}")

# COMMAND ----------

# Renewal vs new business gap (GIPP price-walking check)
renewal_mask = df["is_renewal"].to_numpy() == 1
nb_mask = df["is_renewal"].to_numpy() == 0

gap = PriceValueMetrics.renewal_vs_new_business_gap(
    renewal_premiums=df.filter(pl.col("is_renewal") == 1)["gross_premium"].to_list(),
    new_business_premiums=df.filter(pl.col("is_renewal") == 0)["gross_premium"].to_list(),
    exposure=[1.0] * renewal_mask.sum(),
    period="2025-Q4",
)
print(f"Renewal gap: {gap.metric_value:+.2f}% (threshold: {gap.threshold:.1f}%) — {'PASS' if gap.passed else 'FAIL'}")
if not gap.passed:
    print("Corrective actions required:")
    for action in gap.corrective_actions:
        print(f"  - {action}")

# COMMAND ----------

# Claims timeliness
sla = ClaimsMetrics.timeliness_sla(
    days_to_settlement=df["days_to_settlement"].to_list(),
    period="2025-Q4",
    sla_days=5,
)
print(f"SLA compliance: {sla.metric_value:.1%} (floor: 80%) — {'PASS' if sla.passed else 'FAIL'}")
print(f"Mean settlement time: {sla.extra['mean_days']:.1f} days, P90: {sla.extra['p90_days']:.1f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run the full framework
# MAGIC
# MAGIC The `OutcomeTestingFramework` ties everything together: it runs all
# MAGIC applicable tests, applies segment filters, and produces a unified report.

# COMMAND ----------

from insurance_governance import MRMModelCard
from insurance_governance.outcome import OutcomeTestingFramework, OutcomeResult

card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor Frequency GLM v3",
    version="3.0.0",
    model_class="pricing",
    intended_use="Set gross written premiums for UK private motor insurance",
    portfolio_scope="UK private motor",
    monitoring_owner="Pricing Actuary",
    monitoring_frequency="Quarterly",
)

framework = OutcomeTestingFramework(
    model_card=card,
    policy_data=df,
    period="2025-Q4",
    price_col="gross_premium",
    claim_amount_col="claims_paid",
    claim_outcome_col="claim_outcome",
    days_to_settlement_col="days_to_settlement",
    expenses_col="expenses",
    renewal_indicator_col="is_renewal",
    customer_segments=[renewal_seg, older_seg],
)

results = framework.run()
rag = framework.get_rag_status()
print(f"Overall RAG status: {rag.value.upper()}")
print(f"Total tests: {len(results)}")
print(f"Passed: {sum(1 for r in results if r.passed)}")
print(f"Failed: {sum(1 for r in results if not r.passed)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Inspect failures

# COMMAND ----------

for r in results:
    if not r.passed:
        print(f"FAIL | {r.outcome} | {r.test_name} | segment={r.segment or 'portfolio'}")
        print(f"     {r.details[:120]}")
        if r.corrective_actions:
            print(f"     -> {r.corrective_actions[0]}")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate the board report

# COMMAND ----------

report_path = "/tmp/outcome_report_2025q4.html"
json_path = "/tmp/outcome_report_2025q4.json"

framework.generate(report_path)
framework.to_json(json_path)

print(f"HTML report written to: {report_path}")
print(f"JSON sidecar written to: {json_path}")

# Peek at JSON summary
import json
with open(json_path) as f:
    data = json.load(f)
print("\nReport summary:")
print(json.dumps(data["summary"], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Using OutcomeSuite for downstream processing

# COMMAND ----------

from insurance_governance.outcome import OutcomeSuite

suite = OutcomeSuite(results=results, period="2025-Q4")
print("Suite summary:", suite.summary())
print("\nFailed tests:")
for r in suite.failed:
    print(f"  {r.test_name} ({r.outcome}) — {r.severity.value}")

print(f"\nVulnerable segment results: {len(suite.vulnerable_segment_results())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Injecting custom tests
# MAGIC
# MAGIC Teams can inject any custom metric — complaints rate, NPS, call
# MAGIC abandonment — using `OutcomeResult` directly and passing via `extra_results`.

# COMMAND ----------

custom = OutcomeResult(
    outcome="support",
    test_name="complaints_rate_per_1000",
    passed=True,
    metric_value=3.2,
    threshold=5.0,
    period="2025-Q4",
    details="Complaints rate: 3.2 per 1,000 policies (threshold: 5.0). Source: complaints team.",
    severity=__import__("insurance_governance.validation.results", fromlist=["Severity"]).Severity.INFO,
)

framework2 = OutcomeTestingFramework(
    model_card=card,
    policy_data=df,
    period="2025-Q4",
    price_col="gross_premium",
    extra_results=[custom],
)
results2 = framework2.run()
print(f"With custom test: {len(results2)} results (was {len(results)})")
support_results = [r for r in results2 if r.outcome == "support"]
print(f"Support outcome results: {len(support_results)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This demo showed:
# MAGIC
# MAGIC 1. **Fair value ratio** — claims/premium check (PRIN 2A.3 fair value)
# MAGIC 2. **GIPP price-walking** — renewal vs new business gap (PS21/5 compliance)
# MAGIC 3. **Claims timeliness** — SLA compliance monitoring
# MAGIC 4. **Segment breakdowns** — renewal and vulnerable customer groups
# MAGIC 5. **Board report** — self-contained HTML + JSON audit trail
# MAGIC
# MAGIC The RAG status and corrective actions are designed to be dropped straight
# MAGIC into a quarterly Consumer Duty board report with no additional editing.
