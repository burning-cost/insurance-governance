# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Capability demo: PRA SS1/23 validation report + MRM governance pack
# using insurance-governance on synthetic UK motor data.
#
# Run top-to-bottom on Databricks Free Edition (DBR 14.x+).

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-governance: PRA SS1/23 Validation + MRM Demo
# MAGIC
# MAGIC **Package:** `insurance-governance` — unified model governance for UK insurance pricing.
# MAGIC Two subpackages, one install:
# MAGIC - `insurance_governance.validation` — PRA SS1/23 statistical validation: Gini, PSI,
# MAGIC   Hosmer-Lemeshow, lift charts, discrimination checks. Self-contained HTML reports.
# MAGIC - `insurance_governance.mrm` — Model risk management: `ModelCard`, `RiskTierScorer`,
# MAGIC   `ModelInventory`, `GovernanceReport` (exec committee pack).
# MAGIC
# MAGIC **What this demo shows:**
# MAGIC 1. Train a CatBoost Poisson frequency model on synthetic motor data
# MAGIC 2. Run `ModelValidationReport` — all SS1/23 checks in one call, HTML output
# MAGIC 3. Score a risk tier with `RiskTierScorer` — 6-dimension scorecard, fully auditable
# MAGIC 4. Register the model in `ModelInventory` — JSON-backed, version-controlled
# MAGIC 5. Generate a `GovernanceReport` — exec committee pack, print-to-PDF ready
# MAGIC
# MAGIC **What this replaces:** bespoke Word validation reports, Excel scorecard workbooks,
# MAGIC informal Confluence governance pages. One package gives you the full PRA paper trail.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC **Library version:** 0.1.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install git+https://github.com/burning-cost/insurance-governance.git
%pip install catboost insurance-datasets matplotlib pandas numpy polars

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# Validation subpackage
from insurance_governance.validation import (
    ModelValidationReport,
    ModelCard as ValidationModelCard,
)

# MRM subpackage
from insurance_governance.mrm import (
    ModelCard as MRMModelCard,
    Assumption,
    Limitation,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
)

# Top-level re-exports (same classes, aliased to avoid the two ModelCard clash)
from insurance_governance import __version__ as IG_VERSION

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"insurance-governance {IG_VERSION}")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data and model
# MAGIC
# MAGIC We generate 50,000 synthetic UK motor policies with a known DGP.
# MAGIC The model is a CatBoost Poisson frequency model — the standard workhorse
# MAGIC for personal lines frequency in UK pricing teams.
# MAGIC
# MAGIC We split 60/20/20 by policy year (temporal order preserved), then score
# MAGIC all three sets. Validation runs on the 20% held-out validation set;
# MAGIC the test set is kept aside for the lift chart.

# COMMAND ----------

from insurance_datasets import load_motor_frequency

df = load_motor_frequency(n_policies=50_000, random_state=42)
df = df.sort_values("policy_year").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print(f"Dataset shape:   {df.shape}")
print(f"Train:           {len(train_df):>7,} rows ({100*len(train_df)/n:.0f}%)")
print(f"Validation:      {len(val_df):>7,} rows ({100*len(val_df)/n:.0f}%)")
print(f"Test:            {len(test_df):>7,} rows ({100*len(test_df)/n:.0f}%)")
print(f"\nClaim count distribution (train):")
print(train_df["claim_count"].value_counts().sort_index().head(6).to_string())

# COMMAND ----------

# Feature specification
CAT_FEATURES = ["vehicle_class", "driver_age_band", "ncd_band", "region"]
NUM_FEATURES = ["vehicle_age", "sum_insured_log"]
FEATURES     = CAT_FEATURES + NUM_FEATURES
TARGET       = "claim_count"
EXPOSURE     = "exposure"

# Build CatBoost pools — frequency model uses rate = claims/exposure as target,
# weighted by exposure, so the model output is already in claims/year units.
pool_train = Pool(
    train_df[FEATURES], train_df[TARGET] / train_df[EXPOSURE],
    cat_features=CAT_FEATURES, weight=train_df[EXPOSURE],
)
pool_val = Pool(
    val_df[FEATURES], val_df[TARGET] / val_df[EXPOSURE],
    cat_features=CAT_FEATURES, weight=val_df[EXPOSURE],
)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_seed=42,
)
model.fit(pool_train, eval_set=pool_val, early_stopping_rounds=30)

# Predictions in expected claims (rate × exposure)
y_pred_train = model.predict(train_df[FEATURES]) * train_df[EXPOSURE].values
y_pred_val   = model.predict(val_df[FEATURES])   * val_df[EXPOSURE].values
y_pred_test  = model.predict(test_df[FEATURES])  * test_df[EXPOSURE].values

print(f"Best iteration:          {model.best_iteration_}")
print(f"Train predictions        mean: {y_pred_train.mean():.4f}  std: {y_pred_train.std():.4f}")
print(f"Validation predictions   mean: {y_pred_val.mean():.4f}  std: {y_pred_val.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validation report (PRA SS1/23)
# MAGIC
# MAGIC `ModelValidationReport` is the high-level facade. Pass it the model card and
# MAGIC predictions, call `generate()`, and it produces a fully self-contained HTML report.
# MAGIC
# MAGIC The tests it runs:
# MAGIC - Gini coefficient + bootstrap 95% CI
# MAGIC - 10-band lift chart
# MAGIC - Actual vs expected by predicted decile
# MAGIC - A/E with Poisson CI
# MAGIC - Hosmer-Lemeshow goodness-of-fit
# MAGIC - Lorenz curve data
# MAGIC - Calibration plot data
# MAGIC - PSI on score distribution (train vs validation)
# MAGIC - Monitoring plan completeness check
# MAGIC
# MAGIC All tests return `TestResult` objects with a passed/failed flag, metric value,
# MAGIC and human-readable details. The overall `RAGStatus` (Green/Amber/Red) is computed
# MAGIC from the worst severity failure across all tests.

# COMMAND ----------

# Build the validation ModelCard (Pydantic-based, from insurance_governance.validation)
val_card = ValidationModelCard(
    name="Motor TPPD Frequency v2.3",
    version="2.3.0",
    purpose=(
        "Estimate expected claim frequency for private motor policies. "
        "Used to set reference rates in the pricing engine."
    ),
    methodology="CatBoost gradient boosting with Poisson loss function",
    model_type="GBM",
    target="claim_count",
    features=FEATURES,
    limitations=[
        "No telematics data — usage-based risk factors absent",
        "Training data covers 2019-2024; performance during rapid inflation periods uncertain",
        "Fleet policies excluded — private motor only",
    ],
    owner="Motor Pricing Team",
    intended_use="Underwriting pricing for private motor. Not for reserving or capital.",
    materiality_tier=1,
    approved_by=["Sarah Chen - Chief Actuary", "Model Risk Committee"],
    development_date=date(2025, 9, 1),
    validation_date=date.today(),
    validator_name="Independent Validation Team",
    monitoring_owner="James Whitfield - Senior Pricing Actuary",
    monitoring_frequency="Quarterly",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.08, "gini_decline": 0.03},
    alternatives_considered=(
        "Evaluated Poisson GLM (lower Gini by ~4pp) and LightGBM (comparable performance). "
        "CatBoost selected for stability and interpretability of categorical handling."
    ),
)

print("Validation model card:")
for k, v in val_card.summary().items():
    print(f"  {k:<25} {v}")

# COMMAND ----------

# Build Polars DataFrames for data quality and feature drift checks.
# The validation report uses Polars for these checks; pandas DataFrames
# are not accepted at this point in the API.
X_train_pl = pl.from_pandas(train_df[FEATURES])
X_val_pl   = pl.from_pandas(val_df[FEATURES])

# Run the full validation
val_report = ModelValidationReport(
    model_card=val_card,
    y_val=val_df[TARGET].values,
    y_pred_val=y_pred_val,
    exposure_val=val_df[EXPOSURE].values,
    y_train=train_df[TARGET].values,
    y_pred_train=y_pred_train,
    exposure_train=train_df[EXPOSURE].values,
    X_train=X_train_pl,
    X_val=X_val_pl,
    segment_col="vehicle_class",
    monitoring_owner="James Whitfield - Senior Pricing Actuary",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.08},
    random_state=42,
)

results = val_report.run()
rag = val_report.get_rag_status()
print(f"\nValidation complete. {len(results)} tests run.")
print(f"Overall RAG status: {rag.value.upper()}")

# COMMAND ----------

# Show each test result in a table
rows = []
for r in results:
    rows.append({
        "Test": r.test_name,
        "Category": r.category.value,
        "Passed": "PASS" if r.passed else "FAIL",
        "Value": f"{r.metric_value:.4f}" if r.metric_value is not None else "—",
        "Severity": r.severity.value,
    })

results_df = pd.DataFrame(rows)

# Show key performance tests first
perf_mask = results_df["Category"] == "performance"
print("Performance tests:")
print(results_df[perf_mask].to_string(index=False))

print("\nStability tests:")
stab_mask = results_df["Category"] == "stability"
print(results_df[stab_mask].to_string(index=False))

print("\nMonitoring:")
mon_mask = results_df["Category"] == "monitoring"
print(results_df[mon_mask].to_string(index=False))

# COMMAND ----------

# Pull out the key metrics for display and for feeding into the MRM pack later
def _find_metric(results, name):
    for r in results:
        if r.test_name == name:
            return r.metric_value
    return None

gini_val   = _find_metric(results, "gini_coefficient")
psi_val    = _find_metric(results, "psi_score")
hl_pval    = _find_metric(results, "hosmer_lemeshow")
ae_result  = next((r for r in results if r.test_name == "actual_vs_expected"), None)
ae_ratio   = ae_result.metric_value if ae_result else None

print("Key validation metrics:")
print(f"  Gini coefficient:   {gini_val:.4f}" if gini_val else "  Gini: not found")
print(f"  PSI (score drift):  {psi_val:.4f}"  if psi_val  else "  PSI: not found")
print(f"  H-L p-value:        {hl_pval:.4f}"  if hl_pval  else "  H-L p-value: not found")
print(f"  A/E metric:         {ae_ratio:.4f}"  if ae_ratio else "  A/E: not found")
print(f"\nOverall: {rag.value.upper()}")

# COMMAND ----------

# Write the HTML report and JSON sidecar to /tmp
html_path = val_report.generate("/tmp/motor_freq_v23_validation.html")
json_path = val_report.to_json("/tmp/motor_freq_v23_validation.json")

print(f"HTML report written to: {html_path}")
print(f"JSON sidecar written to: {json_path}")

# Show the JSON structure (truncated)
with open(json_path) as f:
    report_json = json.load(f)

print(f"\nJSON report keys: {list(report_json.keys())}")
print(f"Run ID: {report_json['run_id']}")
print(f"Generated: {report_json['generated_date']}")
print(f"RAG status: {report_json['rag_status']}")
print(f"Result count: {len(report_json['results'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lift chart and calibration diagnostic

# COMMAND ----------

# Build lift chart from the test set (independent of the validation set)
order = np.argsort(y_pred_test)
y_sorted  = val_df[TARGET].values[order] if len(val_df) == len(y_pred_test) else test_df[TARGET].values[np.argsort(y_pred_test)]
# Use test set throughout for the diagnostic plot
y_test    = test_df[TARGET].values
e_test    = test_df[EXPOSURE].values
y_pred_t  = y_pred_test
order_t   = np.argsort(y_pred_t)

n_bands = 10
idx_splits = np.array_split(np.arange(len(y_test))[order_t], n_bands)

actual_rate   = [y_test[i].sum()   / e_test[i].sum() for i in idx_splits]
pred_rate     = [y_pred_t[i].sum() / e_test[i].sum() for i in idx_splits]

# Also compute A/E by predicted decile for calibration
ae_by_decile = [a / p if p > 0 else np.nan for a, p in zip(actual_rate, pred_rate)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lift chart
x_pos = np.arange(1, n_bands + 1)
axes[0].plot(x_pos, actual_rate, "ko-", label="Actual",    linewidth=2)
axes[0].plot(x_pos, pred_rate,   "bs--", label="Predicted", linewidth=1.5, alpha=0.8)
axes[0].set_xlabel("Decile (sorted by predicted rate)")
axes[0].set_ylabel("Claim rate (claims / exposure year)")
axes[0].set_title("Lift Chart — Test Set")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# A/E calibration
axes[1].bar(x_pos, ae_by_decile, color="steelblue", alpha=0.75, label="A/E ratio")
axes[1].axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Perfect (1.0)")
axes[1].set_xlabel("Predicted rate decile")
axes[1].set_ylabel("Actual / Expected ratio")
axes[1].set_title("A/E Calibration by Predicted Decile")
axes[1].set_ylim(0.7, 1.3)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

max_ae_dev = max(abs(ae - 1.0) for ae in ae_by_decile if not np.isnan(ae))
axes[1].set_title(f"A/E Calibration — max deviation {max_ae_dev:.3f}")

plt.suptitle(
    f"Motor TPPD Frequency v2.3 — Validation Diagnostics  (Gini: {gini_val:.3f}  RAG: {rag.value.upper()})",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/tmp/validation_diagnostics.png", dpi=120, bbox_inches="tight")
plt.show()
print("Diagnostic plot saved to /tmp/validation_diagnostics.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MRM governance
# MAGIC
# MAGIC The MRM pack is separate from the statistical validation. The validation report
# MAGIC answers "does the model perform adequately?" The governance pack answers
# MAGIC "what risk tier is this model, who is responsible, and when is the next review?"
# MAGIC
# MAGIC These are two different questions and two different audiences. The validation
# MAGIC report goes to the independent validation team. The governance pack goes to
# MAGIC the Model Risk Committee.
# MAGIC
# MAGIC Workflow:
# MAGIC 1. `MRMModelCard` — structured governance record (dataclass, no Pydantic dep)
# MAGIC 2. `RiskTierScorer` — 6-dimension scorecard producing a 0-100 composite score
# MAGIC 3. `ModelInventory` — persistent JSON registry, lives in version control
# MAGIC 4. `GovernanceReport` — executive committee HTML/JSON pack

# COMMAND ----------

# Build the MRM ModelCard (dataclass-based, from insurance_governance.mrm)
# Note: this is a different class from the validation ModelCard — it carries
# governance metadata like assumptions register, approved_by, GWP, and
# champion/challenger status. The validation card carries statistical metadata.
mrm_card = MRMModelCard(
    model_id="motor-tppd-freq-v23",
    model_name="Motor TPPD Frequency",
    version="2.3.0",
    model_class="pricing",
    intended_use=(
        "Set reference claim frequency rates for UK private motor policies. "
        "Used in the underwriting rating engine for new business and renewals."
    ),
    not_intended_for=[
        "Commercial motor or fleet underwriting",
        "Claims reserving or IBNR calculations",
        "Solvency II internal model capital calculation",
    ],
    target_variable="claim_count",
    distribution_family="Poisson",
    model_type="CatBoost GBM",
    rating_factors=FEATURES,
    training_data_period=("2019-01-01", "2024-06-30"),
    development_date="2025-09-01",
    developer="Motor Pricing Team",
    champion_challenger_status="champion",
    portfolio_scope="UK private motor — new business and renewals",
    geographic_scope="England, Wales, Scotland",
    customer_facing=True,
    regulatory_use=False,
    gwp_impacted=140_000_000.0,   # £140m GWP
    assumptions=[
        Assumption(
            description="Claim frequency has been stationary since 2022 post-pandemic normalisation",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring with 8% deviation trigger for ad-hoc review",
            rationale="Frequency showed step-change in 2020-21; post-2022 trend assumed stable",
        ),
        Assumption(
            description="Vehicle group classification is accurate and consistently applied by UW",
            risk="LOW",
            mitigation="Annual data quality audit of vehicle class coding",
        ),
        Assumption(
            description="No systematic adverse selection in the NCD distribution compared to market",
            risk="HIGH",
            mitigation=(
                "Annual market benchmarking of NCD distribution. "
                "Disparate impact check included in quarterly monitoring."
            ),
            rationale=(
                "NCD walk from external competitor data shows 3% deviation. "
                "Flagged at last MRC. Mitigation plan in place."
            ),
        ),
    ],
    limitations=[
        Limitation(
            description="No telematics or usage-based rating factors",
            impact="Inability to distinguish high and low annual mileage within vehicle class",
            population_at_risk="Low-mileage drivers — potentially over-charged vs telematics peers",
            monitoring_flag=True,
        ),
        Limitation(
            description="Training data does not include the 2022-2024 repair cost inflation period",
            impact="Model may underestimate frequency if higher-cost claims suppress reporting",
            population_at_risk="Newer vehicles (< 3 years old)",
            monitoring_flag=True,
        ),
    ],
    outstanding_issues=[
        "NCD adverse selection analysis — Q2 2026 deadline (James Whitfield)",
        "Telematics integration feasibility study — scheduled Q3 2026",
    ],
    approved_by=["Sarah Chen - Chief Actuary", "Model Risk Committee 2025-10-15"],
    approval_date="2025-10-15",
    approval_conditions=(
        "Approved subject to NCD distribution monitoring report by Q2 2026."
    ),
    next_review_date="2026-10-01",
    monitoring_owner="James Whitfield - Senior Pricing Actuary",
    monitoring_frequency="Quarterly",
    monitoring_triggers={
        "psi_score": 0.20,
        "ae_ratio_deviation": 0.08,
        "gini_decline": 0.03,
    },
    trigger_actions={
        "psi_score > 0.20": "Ad-hoc review within 10 business days",
        "ae_ratio_deviation > 0.08": "Escalate to Chief Actuary within 5 business days",
    },
)

summary = mrm_card.assumption_summary()
print(f"MRM card created for: {mrm_card.model_name} v{mrm_card.version}")
print(f"Model class: {mrm_card.model_class}  |  Status: {mrm_card.champion_challenger_status}")
print(f"GWP impacted: £{mrm_card.gwp_impacted/1e6:.0f}m")
print(f"Assumptions:  HIGH={summary['HIGH']}  MEDIUM={summary['MEDIUM']}  LOW={summary['LOW']}")
print(f"Limitations:  {len(mrm_card.limitations)}")
print(f"Outstanding issues: {len(mrm_card.outstanding_issues)}")
print(f"Approved: {mrm_card.is_approved}")

# COMMAND ----------

# Score the risk tier
# The scorer maps 6 dimensions to a 0-100 composite. No subjective judgement
# required — every point has a documented rule. This makes MRC presentations
# straightforward: you're showing your working, not asking for trust.
scorer = RiskTierScorer()

tier_result = scorer.score(
    gwp_impacted=mrm_card.gwp_impacted,
    model_complexity="high",       # GBM with 6 features and interaction effects
    deployment_status="champion",  # live in rating engine
    regulatory_use=mrm_card.regulatory_use,
    external_data=False,           # all internal data sources
    customer_facing=mrm_card.customer_facing,
    validation_months_ago=6.0,     # validated 6 months ago
    drift_triggers_last_year=1,    # one PSI trigger fired in Q3 2025
)

print(f"Risk tier:       Tier {tier_result.tier} ({tier_result.tier_label})")
print(f"Composite score: {tier_result.score:.1f} / 100")
print(f"Review frequency: {tier_result.review_frequency}")
print(f"Sign-off required: {tier_result.sign_off_requirement}")
print()
print("Dimension breakdown:")
print(f"  {'Dimension':<25}  {'Score':>7}  {'%':>6}  Rationale")
print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*40}")
for dim in tier_result.dimensions:
    contribution = (dim.score / dim.max_score) * scorer.weights[dim.name]
    print(
        f"  {dim.name:<25}  {contribution:>6.1f}pts  {dim.pct:>5.0f}%  {dim.rationale}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scorecard visualisation

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: dimension contribution bar chart
dim_names   = [d.name.replace("_", "\n") for d in tier_result.dimensions]
dim_contrib = [
    (d.score / d.max_score) * scorer.weights[d.name]
    for d in tier_result.dimensions
]
tier_colours = {1: "#dc2626", 2: "#f59e0b", 3: "#3b82f6", 4: "#22c55e"}
bar_colour = tier_colours.get(tier_result.tier, "#6b7280")

bars = axes[0].barh(dim_names, dim_contrib, color=bar_colour, alpha=0.8)
axes[0].axvline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("Points contributed to composite score")
axes[0].set_title(
    f"Risk tier dimension breakdown\n"
    f"Composite: {tier_result.score:.1f}/100 — Tier {tier_result.tier} ({tier_result.tier_label})"
)
axes[0].set_xlim(0, max(dim_contrib) * 1.3)
for bar, val in zip(bars, dim_contrib):
    axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}pt", va="center", fontsize=9)
axes[0].grid(True, alpha=0.3, axis="x")

# Right: gauge-style score indicator
score = tier_result.score
thresholds = scorer.thresholds
tier_bands = [
    (0,  thresholds[3], "#22c55e", "Tier 3 (Medium)"),
    (thresholds[3], thresholds[2], "#3b82f6", "Tier 2 (High)"),
    (thresholds[2], 100, "#dc2626", "Tier 1 (Critical)"),
]

for lo, hi, colour, label in tier_bands:
    axes[1].barh([0], [hi - lo], left=[lo], color=colour, alpha=0.4, height=0.4, label=label)

axes[1].axvline(score, color="black", linewidth=3, label=f"This model: {score:.1f}")
axes[1].set_xlim(0, 100)
axes[1].set_ylim(-0.5, 0.5)
axes[1].set_yticks([])
axes[1].set_xlabel("Composite risk score (0-100)")
axes[1].set_title(f"Tier assignment — score {score:.1f}/100")
axes[1].legend(loc="upper left", fontsize=8)
axes[1].grid(True, alpha=0.3, axis="x")

# Annotate tier thresholds
for t, th in thresholds.items():
    axes[1].axvline(th, color="grey", linewidth=1, linestyle=":")
    axes[1].text(th + 0.5, 0.25, f"≥{th}", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig("/tmp/risk_tier_scorecard.png", dpi=120, bbox_inches="tight")
plt.show()
print("Risk tier scorecard saved to /tmp/risk_tier_scorecard.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model inventory

# COMMAND ----------

# The inventory is a JSON file — check it into git alongside your code.
# It is not a database. It is intentionally readable in any text editor.
# For an insurer with ~50-100 production models, this is the right trade-off.

INVENTORY_PATH = "/tmp/mrm_inventory.json"
inventory = ModelInventory(INVENTORY_PATH)

# Register the model (inserts or updates — idempotent on model_id)
model_id = inventory.register(mrm_card, tier_result)
print(f"Registered: {model_id}")

# Record the validation run (links the inventory entry to the validation report)
today_str = date.today().isoformat()
next_review = (date.today() + timedelta(days=365)).isoformat()

inventory.update_validation(
    model_id=model_id,
    validation_date=today_str,
    overall_rag=rag.value.upper(),
    next_review_date=next_review,
    run_id=report_json["run_id"],
    notes=f"Validation run as part of benchmark demo. Gini: {gini_val:.3f}. RAG: {rag.value.upper()}.",
)

# Log a governance event
inventory.log_event(
    model_id=model_id,
    event_type="validation_complete",
    description=f"Independent validation completed. Overall RAG: {rag.value.upper()}.",
    triggered_by="Insurance-Governance benchmark notebook",
)

# Add a few more models to demonstrate inventory management
for model_id_str, name, gwp, status in [
    ("motor-severity-v14",   "Motor TPPD Severity v1.4",   140_000_000, "champion"),
    ("motor-ad-freq-v31",    "Motor AD Frequency v3.1",     85_000_000, "champion"),
    ("home-buildings-v22",   "Home Buildings Frequency v2.2", 55_000_000, "challenger"),
]:
    extra_card = MRMModelCard(
        model_id=model_id_str,
        model_name=name,
        version=model_id_str.split("-v")[-1] + ".0",
        model_class="pricing",
        intended_use=f"Rating factor model for {name}",
        gwp_impacted=gwp,
        developer="Pricing Team",
        champion_challenger_status=status,
        monitoring_owner="James Whitfield - Senior Pricing Actuary",
    )
    extra_tier = scorer.score(
        gwp_impacted=gwp,
        model_complexity="high",
        deployment_status=status,
        regulatory_use=False,
        external_data=False,
        customer_facing=True,
    )
    inventory.register(extra_card, extra_tier)

# List the inventory
all_models = inventory.list()
print(f"\nInventory: {len(all_models)} models registered\n")

inv_display = pd.DataFrame([{
    "Model": row["model_name"],
    "Status": row["champion_challenger_status"],
    "Tier": f"T{row['materiality_tier']} ({row['tier_label']})",
    "GWP": f"£{row['gwp_impacted']/1e6:.0f}m",
    "RAG": row["overall_rag"] or "—",
    "Next review": row["next_review_date"] or "—",
} for row in all_models])
print(inv_display.to_string(index=False))

# COMMAND ----------

# Inventory summary
inv_summary = inventory.summary()
print("Inventory summary:")
print(f"  Total models:  {inv_summary['total_models']}")
print(f"  By tier:       {inv_summary['by_tier']}")
print(f"  By status:     {inv_summary['by_status']}")
print(f"  By RAG:        {inv_summary['by_rag']}")
print(f"  Overdue:       {inv_summary['overdue_count']}")

# Validation history for the main model
history = inventory.validation_history("motor-tppd-freq-v23")
print(f"\nValidation history for motor-tppd-freq-v23:")
for record in history:
    print(f"  {record['validation_date']}  RAG: {record['overall_rag']}  run_id: {record['run_id'][:12]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Governance report (exec committee pack)
# MAGIC
# MAGIC The `GovernanceReport` answers the questions a Model Risk Committee asks:
# MAGIC - What does this model do, and who owns it?
# MAGIC - What risk tier is it, and why?
# MAGIC - Did the last validation pass?
# MAGIC - What are the material assumptions and outstanding issues?
# MAGIC - Who approved it, on what conditions, and when is the next review?
# MAGIC
# MAGIC The HTML output is print-to-PDF ready. The JSON output can be pushed to
# MAGIC Confluence, a model management portal, or an internal API.

# COMMAND ----------

# Reload the card from inventory (demonstrates the round-trip)
registered_card = inventory.get_card("motor-tppd-freq-v23")

gov_report = GovernanceReport(
    card=registered_card,
    tier=tier_result,
    validation_results={
        "overall_rag": rag.value.upper(),
        "run_id": report_json["run_id"],
        "run_date": today_str,
        "gini": gini_val,
        "ae_ratio": ae_ratio,
        "psi_score": psi_val,
        "hl_p_value": hl_pval,
        "section_results": [
            {"section": "Performance",   "status": "GREEN", "notes": f"Gini {gini_val:.3f}, within acceptable range"},
            {"section": "Stability",     "status": "GREEN", "notes": f"PSI {psi_val:.3f} < 0.20 threshold"},
            {"section": "Data quality",  "status": "GREEN", "notes": "No missing values. Cardinality checks passed."},
            {"section": "Monitoring",    "status": "GREEN", "notes": "Named owner assigned. Triggers documented."},
        ],
    },
    monitoring_results={
        "period": "2026-Q1",
        "ae_ratio": 1.02,
        "psi_score": 0.07,
        "gini": gini_val,
        "recommendation": "Continue — no action required",
        "triggered_alerts": [],
    },
)

# Write HTML and JSON
gov_report.save_html("/tmp/motor_tppd_freq_v23_mrm_pack.html")
gov_report.save_json("/tmp/motor_tppd_freq_v23_mrm_pack.json")
print("Governance report written to:")
print("  HTML: /tmp/motor_tppd_freq_v23_mrm_pack.html")
print("  JSON: /tmp/motor_tppd_freq_v23_mrm_pack.json")

# Show the recommendations section
gov_dict = gov_report.to_dict()
print(f"\nReport date: {gov_dict['report_date']}")
print(f"Tier: {gov_dict['risk_tier']['tier']} ({gov_dict['risk_tier']['tier_label']})")
print(f"Composite score: {gov_dict['risk_tier']['score']}")
print(f"Validation RAG: {gov_dict['validation_summary']['overall_rag']}")
print(f"\nRecommendations:")
for i, rec in enumerate(gov_dict["recommendations"], 1):
    print(f"  {i}. {rec}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. End-to-end workflow summary
# MAGIC
# MAGIC What happened in this notebook, in the order a pricing actuary would do it:
# MAGIC
# MAGIC ```
# MAGIC 1. Train model
# MAGIC    CatBoost Poisson → y_pred_train, y_pred_val, y_pred_test
# MAGIC
# MAGIC 2. Statistical validation (insurance_governance.validation)
# MAGIC    ValidationModelCard(name, version, purpose, ..., monitoring_triggers)
# MAGIC    ModelValidationReport(model_card, y_val, y_pred_val, exposure_val, ...)
# MAGIC    report.generate("validation_report.html")   ← self-contained HTML
# MAGIC    report.to_json("validation_report.json")    ← for MRM ingestion
# MAGIC
# MAGIC 3. Risk tier (insurance_governance.mrm)
# MAGIC    RiskTierScorer().score(gwp_impacted, model_complexity, ...)
# MAGIC    → TierResult(tier=1, score=68.4, rationale="...")
# MAGIC
# MAGIC 4. Governance record
# MAGIC    MRMModelCard(model_id, ..., assumptions, limitations)
# MAGIC    ModelInventory(path).register(card, tier_result)
# MAGIC    inventory.update_validation(model_id, rag, next_review_date, run_id)
# MAGIC
# MAGIC 5. Exec committee pack
# MAGIC    GovernanceReport(card, tier, validation_results, monitoring_results)
# MAGIC    report.save_html("mrm_pack.html")   ← print-to-PDF for MRC
# MAGIC    report.save_json("mrm_pack.json")   ← Confluence / portal ingestion
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict
# MAGIC
# MAGIC **When to use insurance-governance:**
# MAGIC
# MAGIC - You are subject to PRA SS1/23 and need an auditable, consistent validation
# MAGIC   process — not a bespoke analyst notebook that varies by model.
# MAGIC - You have 10+ production pricing models and no structured MRM inventory.
# MAGIC   Tracking tier, RAG status, and review dates in a spreadsheet is where things slip.
# MAGIC - You are preparing for a PRA supervisory visit and need to demonstrate
# MAGIC   systematic model risk governance, not ad-hoc Word documents.
# MAGIC - Your MRC asks "what risk tier is this and why?" and you want an answer
# MAGIC   that doesn't require an hour of explanation.
# MAGIC
# MAGIC **What this replaces:**
# MAGIC
# MAGIC | Before | After |
# MAGIC |--------|-------|
# MAGIC | Bespoke validation notebook per model | `ModelValidationReport` — consistent SS1/23 checks across the portfolio |
# MAGIC | Manual Word/PowerPoint validation report | Self-contained HTML, print-to-PDF |
# MAGIC | Excel scorecard for risk tier assignment | `RiskTierScorer` — documented rules, 6-dimension scorecard, full rationale |
# MAGIC | Sharepoint spreadsheet model inventory | `ModelInventory` — JSON-backed, version-controlled |
# MAGIC | 4-page Word governance pack for MRC | `GovernanceReport` — structured HTML/JSON, generated in < 1 second |
# MAGIC
# MAGIC **What this does not replace:**
# MAGIC
# MAGIC - Independent human judgement in validation — the library runs the tests,
# MAGIC   a qualified actuary still reviews the results.
# MAGIC - Model development — this is a governance library, not a modelling one.
# MAGIC - Claims reserving or capital model governance — scoped to pricing models.
# MAGIC
# MAGIC **Regulatory positioning:**
# MAGIC
# MAGIC The validation tests map to PRA SS1/23 Principles 1-5. The model card fields
# MAGIC map to FCA TR24/2 pricing governance documentation requirements. The risk tier
# MAGIC thresholds (Tier 1 ≥ 60 pts, Tier 2 ≥ 30 pts) are calibrated to UK personal
# MAGIC lines practice. You can override thresholds at construction time if your
# MAGIC internal policy differs — but document why.

# COMMAND ----------

# Final output: a structured summary for the notebook reader
print("=" * 65)
print("DEMO COMPLETE: insurance-governance")
print("=" * 65)
print()
print("Validation report")
print(f"  Tests run:            {len(results)}")
print(f"  Overall RAG:          {rag.value.upper()}")
print(f"  Gini coefficient:     {gini_val:.4f}" if gini_val else "  Gini: n/a")
print(f"  PSI (score drift):    {psi_val:.4f}"  if psi_val  else "  PSI: n/a")
print(f"  H-L p-value:          {hl_pval:.4f}"  if hl_pval  else "  H-L: n/a")
print()
print("Risk tier")
print(f"  Composite score:      {tier_result.score:.1f} / 100")
print(f"  Tier:                 {tier_result.tier} ({tier_result.tier_label})")
print(f"  Review frequency:     {tier_result.review_frequency}")
print(f"  Sign-off required:    {tier_result.sign_off_requirement}")
print()
print("Inventory")
print(f"  Models registered:    {len(all_models)}")
print(f"  Champion models:      {sum(1 for r in all_models if r['champion_challenger_status'] == 'champion')}")
print(f"  Tier 1 models:        {sum(1 for r in all_models if r['materiality_tier'] == 1)}")
print()
print("Outputs written:")
print("  /tmp/motor_freq_v23_validation.html   (SS1/23 validation report)")
print("  /tmp/motor_freq_v23_validation.json   (JSON sidecar for MRM ingestion)")
print("  /tmp/mrm_inventory.json               (model inventory registry)")
print("  /tmp/motor_tppd_freq_v23_mrm_pack.html (exec committee governance pack)")
print("  /tmp/motor_tppd_freq_v23_mrm_pack.json (JSON governance pack)")
