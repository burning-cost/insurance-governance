# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Benchmark: insurance-governance Consumer Duty and SS1/23 best-practice validation on freMTPL2
# (OpenML dataset 41214 — 677K French motor third-party liability claims)
#
# This is a real-data benchmark, not synthetic data. freMTPL2 is the standard
# public benchmark dataset for non-life frequency models. We fit a Poisson GLM
# and a CatBoost frequency model, then run insurance-governance's full validation
# suite on both to produce Consumer Duty and SS1/23 best-practice aligned reports.
#
# Run top-to-bottom on Databricks (DBR 14.x+).

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-governance: freMTPL2 Benchmark (Real-Data Validation)
# MAGIC
# MAGIC **Dataset:** freMTPL2 (OpenML 41214) — 677,991 French MTPL policies, 11 features.
# MAGIC Standard public benchmark for non-life frequency modelling. First published
# MAGIC in Charpentier (2014); used in GLM → GBM comparison studies.
# MAGIC
# MAGIC **Why this matters for governance:**
# MAGIC Synthetic benchmarks validate the library's logic. A real-data benchmark
# MAGIC shows what the validation reports look like in practice: how discriminatory
# MAGIC a real-world frequency model actually is (Gini is lower than you expect),
# MAGIC what PSI levels are normal, and where Hosmer-Lemeshow flags problems that
# MAGIC global A/E masks.
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC 1. Load freMTPL2 from OpenML (677K rows, no preprocessing shortcuts)
# MAGIC 2. Fit a Poisson GLM baseline (statsmodels) and a CatBoost GBM
# MAGIC 3. Run `ModelValidationReport` on both — full Consumer Duty and SS1/23 best-practice aligned suite
# MAGIC 4. Score risk tiers and generate governance packs for both models
# MAGIC 5. Side-by-side comparison: what the validation suite tells you about
# MAGIC    the gap between GLM and GBM on real data
# MAGIC
# MAGIC **Date:** 2026-03-28
# MAGIC **Library version:** 0.1.7

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install git+https://github.com/burning-cost/insurance-governance.git
%pip install catboost statsmodels scikit-learn matplotlib pandas numpy polars

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import json
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from catboost import CatBoostRegressor, Pool
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from insurance_governance.validation import (
    ModelCard as ValidationModelCard,
    ModelValidationReport,
)
from insurance_governance.mrm import (
    ModelCard as MRMModelCard,
    Assumption,
    Limitation,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
)
from insurance_governance import __version__ as IG_VERSION

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"insurance-governance {IG_VERSION}")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load freMTPL2
# MAGIC
# MAGIC OpenML dataset 41214. 677,991 French motor third-party liability policies.
# MAGIC Each row is one policy-year. The target is `ClaimNb` (claim count); the
# MAGIC exposure is `Exposure` (fraction of year at risk).
# MAGIC
# MAGIC Features:
# MAGIC - `Area` — area code (categorical, A–F)
# MAGIC - `VehPower` — vehicle power band
# MAGIC - `VehAge` — vehicle age in years
# MAGIC - `DrivAge` — driver age in years
# MAGIC - `BonusMalus` — French bonus-malus coefficient (50 = max discount)
# MAGIC - `VehBrand` — vehicle brand (categorical)
# MAGIC - `VehGas` — fuel type: Regular / Diesel
# MAGIC - `Density` — population density at policyholder's commune
# MAGIC - `Region` — French administrative region (categorical)
# MAGIC
# MAGIC The dataset has a known heavy zero-inflation: 93%+ of rows have zero claims.
# MAGIC That is realistic — UK motor is similar.

# COMMAND ----------

print("Fetching freMTPL2 from OpenML (dataset 41214)...")
data = fetch_openml(data_id=41214, as_frame=True, parser="auto")

df = data.frame.copy()
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes.to_string()}")

# COMMAND ----------

# Inspect the target and exposure
print("ClaimNb distribution:")
print(df["ClaimNb"].value_counts().sort_index().head(8).to_string())
print(f"\nClaim rate (unweighted):  {df['ClaimNb'].mean():.4f}")
print(f"Exposure stats:")
print(df["Exposure"].describe().to_string())
print(f"\nExposure-weighted claim rate: {df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Preprocessing
# MAGIC
# MAGIC Minimal preprocessing — we want the benchmark to be reproducible from the raw
# MAGIC OpenML download. The only transformations:
# MAGIC - Coerce numeric columns to float (OpenML sometimes returns them as object)
# MAGIC - Cap BonusMalus at 150 (extreme outliers, <0.1% of rows)
# MAGIC - Cap Density at 99th percentile (log-transform later for GLM only)
# MAGIC - Log-transform Density for the GLM (heavy right tail)
# MAGIC
# MAGIC CatBoost receives the raw features; the GLM receives numerics only with log-density.

# COMMAND ----------

# Coerce numeric columns
for col in ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Cap outliers
bm_cap = 150.0
df["BonusMalus"] = df["BonusMalus"].clip(upper=bm_cap)

density_cap = df["Density"].quantile(0.99)
df["Density_capped"] = df["Density"].clip(upper=density_cap)
df["LogDensity"] = np.log1p(df["Density_capped"])

# Encode categorical columns for CatBoost (keep as string; CatBoost handles natively)
CAT_COLS = ["Area", "VehBrand", "VehGas", "Region"]
for col in CAT_COLS:
    df[col] = df[col].astype(str)

# Drop any rows with NaN in key columns
key_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"] + CAT_COLS
df = df.dropna(subset=key_cols).reset_index(drop=True)

print(f"After preprocessing: {len(df):,} rows")
print(f"Exposure-weighted claim rate: {df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")
print(f"Zero-claim rows: {(df['ClaimNb'] == 0).mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Train / validation / test split
# MAGIC
# MAGIC No temporal ordering is available in freMTPL2, so we use a stratified
# MAGIC random split. For a real model this would be a time-based split; we
# MAGIC note this as a limitation in the model cards.
# MAGIC
# MAGIC Split: 60% train / 20% validation / 20% test.
# MAGIC The validation set is what `ModelValidationReport` sees; the test set
# MAGIC is held out for the lift chart.

# COMMAND ----------

# Split indices
train_idx, temp_idx = train_test_split(
    np.arange(len(df)), test_size=0.40, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42
)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

n = len(df)
print(f"Total:      {n:>9,}")
print(f"Train:      {len(train_df):>9,}  ({100*len(train_df)/n:.1f}%)")
print(f"Validation: {len(val_df):>9,}  ({100*len(val_df)/n:.1f}%)")
print(f"Test:       {len(test_df):>9,}  ({100*len(test_df)/n:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model A — Poisson GLM (baseline)
# MAGIC
# MAGIC The GLM is the standard baseline for frequency models. We use
# MAGIC statsmodels with a log link and log(Exposure) as an offset.
# MAGIC Features: VehAge, DrivAge, BonusMalus, LogDensity, VehPower.
# MAGIC Categorical features (Area, VehBrand, VehGas, Region) would require
# MAGIC dummy encoding; we exclude them here to keep the GLM tractable and
# MAGIC focus on the governance comparison, not the modelling competition.
# MAGIC
# MAGIC Note the GLM prediction is expected claim count (rate × exposure),
# MAGIC not claim rate. Both models output on the same scale so the validation
# MAGIC suite's A/E calculations are directly comparable.

# COMMAND ----------

GLM_NUM_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]

def build_glm_X(df):
    """Add intercept to numeric features for statsmodels."""
    X = df[GLM_NUM_FEATURES].copy().astype(float)
    return sm.add_constant(X)

offset_train = np.log(train_df["Exposure"].clip(lower=1e-6).values)
offset_val   = np.log(val_df["Exposure"].clip(lower=1e-6).values)
offset_test  = np.log(test_df["Exposure"].clip(lower=1e-6).values)

X_glm_train = build_glm_X(train_df)
X_glm_val   = build_glm_X(val_df)
X_glm_test  = build_glm_X(test_df)

glm = sm.GLM(
    train_df["ClaimNb"].values,
    X_glm_train,
    family=sm.families.Poisson(),
    offset=offset_train,
).fit(disp=False)

print(glm.summary2().tables[0].to_string())
print(f"\nConverged: {glm.converged}")

# COMMAND ----------

# Predictions: expected claim count (not rate)
glm_pred_train = glm.predict(X_glm_train, offset=offset_train)
glm_pred_val   = glm.predict(X_glm_val,   offset=offset_val)
glm_pred_test  = glm.predict(X_glm_test,  offset=offset_test)

ae_train = train_df["ClaimNb"].sum() / glm_pred_train.sum()
ae_val   = val_df["ClaimNb"].sum()   / glm_pred_val.sum()

print(f"GLM predictions (validation)   mean: {glm_pred_val.mean():.5f}")
print(f"GLM A/E (train):               {ae_train:.4f}")
print(f"GLM A/E (validation):          {ae_val:.4f}")
print(f"Validation claim count:        {val_df['ClaimNb'].sum():,}")
print(f"Validation predicted count:    {glm_pred_val.sum():,.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model B — CatBoost Poisson GBM
# MAGIC
# MAGIC CatBoost Poisson with all 9 features (including 4 categoricals handled
# MAGIC natively). We use rate = ClaimNb / Exposure as the target weighted by
# MAGIC Exposure — the standard approach that matches the offset trick in the GLM.
# MAGIC
# MAGIC Early stopping on the validation pool prevents overfitting.

# COMMAND ----------

GBM_FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
                 "VehBrand", "VehGas", "LogDensity", "Region"]

def make_pool(df, cat_cols=CAT_COLS):
    rate   = (df["ClaimNb"] / df["Exposure"].clip(lower=1e-6)).values
    weight = df["Exposure"].values
    return Pool(df[GBM_FEATURES], rate, cat_features=cat_cols, weight=weight)

pool_train = make_pool(train_df)
pool_val   = make_pool(val_df)

gbm = CatBoostRegressor(
    loss_function="Poisson",
    iterations=600,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    verbose=0,
    random_seed=42,
)
gbm.fit(pool_train, eval_set=pool_val, early_stopping_rounds=40)

# Predictions: expected claim count
gbm_rate_train = gbm.predict(train_df[GBM_FEATURES])
gbm_rate_val   = gbm.predict(val_df[GBM_FEATURES])
gbm_rate_test  = gbm.predict(test_df[GBM_FEATURES])

gbm_pred_train = gbm_rate_train * train_df["Exposure"].values
gbm_pred_val   = gbm_rate_val   * val_df["Exposure"].values
gbm_pred_test  = gbm_rate_test  * test_df["Exposure"].values

ae_gbm_train = train_df["ClaimNb"].sum() / gbm_pred_train.sum()
ae_gbm_val   = val_df["ClaimNb"].sum()   / gbm_pred_val.sum()

print(f"Best iteration:                {gbm.best_iteration_}")
print(f"GBM predictions (validation)   mean: {gbm_pred_val.mean():.5f}")
print(f"GBM A/E (train):               {ae_gbm_train:.4f}")
print(f"GBM A/E (validation):          {ae_gbm_val:.4f}")
print(f"GBM vs GLM improvement (val A/E dev): {abs(ae_gbm_val-1.0):.4f} vs {abs(ae_val-1.0):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validation reports — insurance-governance
# MAGIC
# MAGIC We run `ModelValidationReport` on both models using identical settings.
# MAGIC This is the point of the benchmark: the same governance machinery applies
# MAGIC to both a GLM and a GBM. The reports are comparable because the test
# MAGIC suite is fixed — the analyst doesn't get to choose which tests to run.
# MAGIC
# MAGIC Both models receive the Polars DataFrames for feature drift checks.
# MAGIC We use `VehGas` as the segment column (binary, clean split) for the
# MAGIC per-segment A/E output.

# COMMAND ----------

# Polars feature frames for data quality / drift checks
X_train_pl = pl.from_pandas(train_df[GBM_FEATURES])
X_val_pl   = pl.from_pandas(val_df[GBM_FEATURES])

# --- GLM validation card ---
glm_card = ValidationModelCard(
    name="freMTPL2 Poisson GLM (numeric features only)",
    version="1.0.0",
    purpose=(
        "Baseline frequency model on freMTPL2 French MTPL data. "
        "Uses 5 numeric features with log link and exposure offset."
    ),
    methodology="Poisson GLM (statsmodels), log link, log(Exposure) offset",
    model_type="GLM",
    target="ClaimNb",
    features=GLM_NUM_FEATURES,
    limitations=[
        "Excludes categorical features (Area, VehBrand, VehGas, Region) — simplification for baseline comparison",
        "No temporal split available in freMTPL2 — random train/val/test split used",
        "BonusMalus capped at 150; Density log-transformed — may not be optimal",
        "French data — not directly calibrated to UK market",
    ],
    owner="Burning Cost (benchmark)",
    development_date=date(2026, 3, 28),
    validation_date=date.today(),
    validator_name="insurance-governance automated suite",
    monitoring_owner="Burning Cost benchmark (no live monitoring)",
    monitoring_frequency="Not applicable — benchmark only",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.10},
)

# --- GBM validation card ---
gbm_card = ValidationModelCard(
    name="freMTPL2 CatBoost Poisson GBM (all features)",
    version="1.0.0",
    purpose=(
        "Full-feature frequency model on freMTPL2 French MTPL data. "
        "Uses all 9 features including 4 categoricals handled natively by CatBoost."
    ),
    methodology="CatBoost GBM, Poisson loss, rate target weighted by exposure",
    model_type="GBM",
    target="ClaimNb",
    features=GBM_FEATURES,
    limitations=[
        "No temporal split available in freMTPL2 — random train/val/test split used",
        "French data — not directly calibrated to UK market",
        "BonusMalus capped at 150; Density log-transformed",
    ],
    owner="Burning Cost (benchmark)",
    development_date=date(2026, 3, 28),
    validation_date=date.today(),
    validator_name="insurance-governance automated suite",
    monitoring_owner="Burning Cost benchmark (no live monitoring)",
    monitoring_frequency="Not applicable — benchmark only",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.10},
)

print("Model cards created.")
for label, card in [("GLM", glm_card), ("GBM", gbm_card)]:
    s = card.summary()
    print(f"\n{label}:")
    for k, v in s.items():
        print(f"  {k:<25} {v}")

# COMMAND ----------

# Run GLM validation
glm_report = ModelValidationReport(
    model_card=glm_card,
    y_val=val_df["ClaimNb"].values,
    y_pred_val=glm_pred_val,
    exposure_val=val_df["Exposure"].values,
    y_train=train_df["ClaimNb"].values,
    y_pred_train=glm_pred_train,
    exposure_train=train_df["Exposure"].values,
    X_train=X_train_pl,
    X_val=X_val_pl,
    segment_col="VehGas",
    monitoring_owner="Burning Cost benchmark",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.10},
    random_state=42,
)

glm_results = glm_report.run()
glm_rag     = glm_report.get_rag_status()

print(f"GLM validation: {len(glm_results)} tests  |  Overall RAG: {glm_rag.value.upper()}")

# COMMAND ----------

# Run GBM validation
gbm_report = ModelValidationReport(
    model_card=gbm_card,
    y_val=val_df["ClaimNb"].values,
    y_pred_val=gbm_pred_val,
    exposure_val=val_df["Exposure"].values,
    y_train=train_df["ClaimNb"].values,
    y_pred_train=gbm_pred_train,
    exposure_train=train_df["Exposure"].values,
    X_train=X_train_pl,
    X_val=X_val_pl,
    segment_col="VehGas",
    monitoring_owner="Burning Cost benchmark",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_deviation": 0.10},
    random_state=42,
)

gbm_results = gbm_report.run()
gbm_rag     = gbm_report.get_rag_status()

print(f"GBM validation: {len(gbm_results)} tests  |  Overall RAG: {gbm_rag.value.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Results comparison table

# COMMAND ----------

def _find_metric(results, name):
    for r in results:
        if r.test_name == name:
            return r.metric_value
    return None

def _find_passed(results, name):
    for r in results:
        if r.test_name == name:
            return r.passed
    return None

# Extract key metrics
metrics = {}
for label, results in [("GLM", glm_results), ("GBM", gbm_results)]:
    metrics[label] = {
        "gini":       _find_metric(results, "gini_coefficient"),
        "psi":        _find_metric(results, "psi_score"),
        "hl_pval":    _find_metric(results, "hosmer_lemeshow"),
        "ae":         _find_metric(results, "actual_vs_expected"),
    }

# Print side-by-side
print(f"{'Metric':<25}  {'GLM':>10}  {'GBM':>10}  {'GBM better?':>12}")
print("-" * 65)
for metric_name, fmt in [
    ("gini", "{:.4f}"),
    ("psi",  "{:.4f}"),
    ("hl_pval", "{:.6f}"),
    ("ae",   "{:.4f}"),
]:
    glm_v = metrics["GLM"][metric_name]
    gbm_v = metrics["GBM"][metric_name]
    glm_s = fmt.format(glm_v) if glm_v is not None else "—"
    gbm_s = fmt.format(gbm_v) if gbm_v is not None else "—"

    if metric_name == "gini":
        better = "YES" if (gbm_v or 0) > (glm_v or 0) else "no"
    elif metric_name == "hl_pval":
        better = "YES" if (gbm_v or 0) > (glm_v or 0) else "no"
    elif metric_name == "psi":
        better = "YES" if (gbm_v or 1) < (glm_v or 1) else "no"
    elif metric_name == "ae":
        better = "YES" if abs((gbm_v or 1) - 1.0) < abs((glm_v or 1) - 1.0) else "no"
    else:
        better = "—"

    print(f"  {metric_name:<23}  {glm_s:>10}  {gbm_s:>10}  {better:>12}")

print()
print(f"  {'Overall RAG':<23}  {'GLM: '+glm_rag.value.upper():>10}  {'GBM: '+gbm_rag.value.upper():>10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Test-level detail

# COMMAND ----------

# Show all test results for each model
for label, results, rag in [("GLM", glm_results, glm_rag), ("GBM", gbm_results, gbm_rag)]:
    rows = []
    for r in results:
        rows.append({
            "Test": r.test_name,
            "Category": r.category.value,
            "Passed": "PASS" if r.passed else "FAIL",
            "Value": f"{r.metric_value:.4f}" if r.metric_value is not None else "—",
            "Severity": r.severity.value,
        })
    res_df = pd.DataFrame(rows)
    print(f"\n--- {label} ({rag.value.upper()}) ---")
    print(res_df.to_string(index=False))

# COMMAND ----------

# Generate HTML reports and JSON sidecars
glm_html = glm_report.generate("/tmp/fremtpl2_glm_validation.html")
glm_json = glm_report.to_json("/tmp/fremtpl2_glm_validation.json")

gbm_html = gbm_report.generate("/tmp/fremtpl2_gbm_validation.html")
gbm_json = gbm_report.to_json("/tmp/fremtpl2_gbm_validation.json")

with open(glm_json) as f:
    glm_report_json = json.load(f)
with open(gbm_json) as f:
    gbm_report_json = json.load(f)

print(f"Reports written:")
print(f"  GLM HTML: {glm_html}")
print(f"  GLM JSON: {glm_json}  (run_id: {glm_report_json['run_id'][:12]}...)")
print(f"  GBM HTML: {gbm_html}")
print(f"  GBM JSON: {gbm_json}  (run_id: {gbm_report_json['run_id'][:12]}...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Diagnostic plots

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

n_bands = 10

for row_idx, (label, y_pred_t, results) in enumerate([
    ("GLM", glm_pred_test, glm_results),
    ("GBM", gbm_pred_test, gbm_results),
]):
    y_test = test_df["ClaimNb"].values
    e_test = test_df["Exposure"].values

    order_t     = np.argsort(y_pred_t)
    idx_splits  = np.array_split(np.arange(len(y_test))[order_t], n_bands)

    actual_rate = [y_test[i].sum() / e_test[i].sum() for i in idx_splits]
    pred_rate   = [y_pred_t[i].sum() / e_test[i].sum() for i in idx_splits]
    ae_decile   = [a / p if p > 0 else np.nan for a, p in zip(actual_rate, pred_rate)]

    x_pos = np.arange(1, n_bands + 1)
    gini_v = _find_metric(results, "gini_coefficient")

    # Lift chart
    axes[row_idx, 0].plot(x_pos, actual_rate, "ko-", label="Actual",    linewidth=2)
    axes[row_idx, 0].plot(x_pos, pred_rate,   "bs--", label="Predicted", linewidth=1.5, alpha=0.8)
    axes[row_idx, 0].set_xlabel("Predicted rate decile")
    axes[row_idx, 0].set_ylabel("Claim rate (claims / exposure year)")
    axes[row_idx, 0].set_title(f"{label} — Lift Chart (test set)")
    axes[row_idx, 0].legend(fontsize=8)
    axes[row_idx, 0].grid(True, alpha=0.3)

    # A/E by decile
    axes[row_idx, 1].bar(x_pos, ae_decile, color="steelblue", alpha=0.75)
    axes[row_idx, 1].axhline(1.0, color="black", linewidth=1.5, linestyle="--")
    axes[row_idx, 1].set_xlabel("Predicted rate decile")
    axes[row_idx, 1].set_ylabel("A/E ratio")
    axes[row_idx, 1].set_title(f"{label} — A/E by Decile")
    axes[row_idx, 1].set_ylim(0.5, 1.5)
    axes[row_idx, 1].grid(True, alpha=0.3, axis="y")

    # Predicted distribution (log scale)
    axes[row_idx, 2].hist(
        np.log1p(y_pred_t),
        bins=60, color="steelblue", alpha=0.7, edgecolor="none"
    )
    axes[row_idx, 2].set_xlabel("log(1 + predicted claim count)")
    axes[row_idx, 2].set_ylabel("Count")
    axes[row_idx, 2].set_title(
        f"{label} — Predicted distribution  (Gini: {gini_v:.3f})"
        if gini_v else f"{label} — Predicted distribution"
    )
    axes[row_idx, 2].grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "freMTPL2 Benchmark — GLM vs CatBoost GBM (insurance-governance validation)",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("/tmp/fremtpl2_diagnostics.png", dpi=120, bbox_inches="tight")
plt.show()
print("Diagnostic plots saved to /tmp/fremtpl2_diagnostics.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MRM governance — risk tier scoring
# MAGIC
# MAGIC Both models get a risk tier assessment. In a real deployment:
# MAGIC - The GBM is the champion model (higher Gini, more features, more complexity)
# MAGIC - The GLM is the challenger / fallback (interpretable, auditable by hand)
# MAGIC
# MAGIC The RiskTierScorer reflects this: the GBM scores higher on complexity
# MAGIC and external data dimensions (because it uses region/brand), while the
# MAGIC GLM scores lower — a real-world Tier 2 vs Tier 2/3 split is typical.

# COMMAND ----------

scorer = RiskTierScorer()

glm_tier = scorer.score(
    gwp_impacted=0.0,           # benchmark — no live GWP
    model_complexity="low",     # numeric GLM, interpretable by hand
    deployment_status="challenger",
    regulatory_use=False,
    external_data=False,
    customer_facing=False,      # benchmark, not live
    validation_months_ago=0.0,
    drift_triggers_last_year=0,
)

gbm_tier = scorer.score(
    gwp_impacted=0.0,           # benchmark — no live GWP
    model_complexity="high",    # GBM with 9 features including categoricals
    deployment_status="challenger",
    regulatory_use=False,
    external_data=False,        # all features in OpenML dataset
    customer_facing=False,
    validation_months_ago=0.0,
    drift_triggers_last_year=0,
)

print("Risk tier results:")
print(f"  GLM:  Tier {glm_tier.tier} ({glm_tier.tier_label})  — score {glm_tier.score:.1f}/100")
print(f"  GBM:  Tier {gbm_tier.tier} ({gbm_tier.tier_label})  — score {gbm_tier.score:.1f}/100")
print()
print("Note: both models score low because GWP=0 (benchmark), customer_facing=False,")
print("and regulatory_use=False. In live deployment the GBM would score higher on")
print("complexity, pushing it toward Tier 1 if GWP is material (>£50m).")

# COMMAND ----------

# Build MRM cards for both models
glm_mrm = MRMModelCard(
    model_id="fremtpl2-glm-v1",
    model_name="freMTPL2 Poisson GLM",
    version="1.0.0",
    model_class="pricing",
    intended_use="Benchmark baseline on French MTPL open data. Not for production use.",
    target_variable="ClaimNb",
    distribution_family="Poisson",
    model_type="GLM",
    rating_factors=GLM_NUM_FEATURES,
    developer="Burning Cost",
    champion_challenger_status="challenger",
    assumptions=[
        Assumption(
            description="Claim frequency is Poisson-distributed conditional on features",
            risk="MEDIUM",
            mitigation="Hosmer-Lemeshow goodness-of-fit test included in validation suite",
        ),
        Assumption(
            description="Log-linear relationship between features and claim rate",
            risk="LOW",
            mitigation="Lift chart by decile; A/E by segment reviewed",
        ),
    ],
    limitations=[
        Limitation(
            description="Excludes categorical features — underfits high-risk segments",
            impact="Higher-risk segments (young drivers, urban areas) likely underpredicted",
            population_at_risk="Urban / young driver segment",
            monitoring_flag=True,
        ),
    ],
    monitoring_owner="Burning Cost benchmark",
    monitoring_frequency="Not applicable",
)

gbm_mrm = MRMModelCard(
    model_id="fremtpl2-gbm-v1",
    model_name="freMTPL2 CatBoost GBM",
    version="1.0.0",
    model_class="pricing",
    intended_use="Full-feature benchmark on French MTPL open data. Not for production use.",
    target_variable="ClaimNb",
    distribution_family="Poisson",
    model_type="GBM",
    rating_factors=GBM_FEATURES,
    developer="Burning Cost",
    champion_challenger_status="champion",
    assumptions=[
        Assumption(
            description="Claim frequency is Poisson-distributed conditional on features",
            risk="MEDIUM",
            mitigation="Hosmer-Lemeshow goodness-of-fit included in validation",
        ),
        Assumption(
            description="Random train/val/test split is representative — no temporal shift",
            risk="HIGH",
            mitigation="For production use, temporal split is required",
            rationale="freMTPL2 does not include a policy date field",
        ),
    ],
    limitations=[
        Limitation(
            description="French MTPL product — pricing structure differs from UK motor",
            impact="BonusMalus system has no direct UK equivalent; feature effects not transferable",
            population_at_risk="Not applicable — benchmark data only",
            monitoring_flag=False,
        ),
    ],
    monitoring_owner="Burning Cost benchmark",
    monitoring_frequency="Not applicable",
)

print("MRM cards created for GLM and GBM.")

# COMMAND ----------

# Generate governance reports
for label, mrm_card, tier_result, rag, results, rep_json in [
    ("GLM", glm_mrm, glm_tier, glm_rag, glm_results, glm_report_json),
    ("GBM", gbm_mrm, gbm_tier, gbm_rag, gbm_results, gbm_report_json),
]:
    gini_v  = _find_metric(results, "gini_coefficient")
    psi_v   = _find_metric(results, "psi_score")
    hl_v    = _find_metric(results, "hosmer_lemeshow")
    ae_v    = _find_metric(results, "actual_vs_expected")

    gov = GovernanceReport(
        card=mrm_card,
        tier=tier_result,
        validation_results={
            "overall_rag": rag.value.upper(),
            "run_id": rep_json["run_id"],
            "run_date": date.today().isoformat(),
            "gini": gini_v,
            "ae_ratio": ae_v,
            "psi_score": psi_v,
            "hl_p_value": hl_v,
            "section_results": [
                {
                    "section": "Performance",
                    "status": "GREEN" if _find_passed(results, "gini_coefficient") else "RED",
                    "notes": f"Gini {gini_v:.3f}" if gini_v else "—",
                },
                {
                    "section": "Stability",
                    "status": "GREEN" if _find_passed(results, "psi_score") else "AMBER",
                    "notes": f"PSI {psi_v:.3f}" if psi_v else "—",
                },
                {
                    "section": "Calibration",
                    "status": "GREEN" if _find_passed(results, "hosmer_lemeshow") else "RED",
                    "notes": f"H-L p={hl_v:.4f}" if hl_v else "—",
                },
            ],
        },
    )
    gov.save_html(f"/tmp/fremtpl2_{label.lower()}_mrm_pack.html")
    gov.save_json(f"/tmp/fremtpl2_{label.lower()}_mrm_pack.json")
    print(f"{label} governance pack written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Findings
# MAGIC
# MAGIC ### What the freMTPL2 benchmark tells us about real-data model governance
# MAGIC
# MAGIC **Gini is lower than synthetic benchmarks suggest.**
# MAGIC Real frequency models on large datasets with heterogeneous exposure typically
# MAGIC achieve Gini coefficients of 0.15–0.30 for the GLM and 0.25–0.40 for a
# MAGIC well-tuned GBM. Synthetic benchmarks with known DGPs produce artificially
# MAGIC high Gini values because there is no irreducible noise in the features.
# MAGIC The governance suite's Green/Amber/Red thresholds should be calibrated to
# MAGIC your actual portfolio — not to synthetic data expectations.
# MAGIC
# MAGIC **The GLM's H-L failure is instructive.**
# MAGIC On 677K rows, Hosmer-Lemeshow has enough power to detect small systematic
# MAGIC miscalibration that is invisible in global A/E. The GLM's exclusion of
# MAGIC categorical features leaves systematic residuals in high-risk segments.
# MAGIC A global A/E close to 1.0 masks this. The validation suite catches it.
# MAGIC
# MAGIC **PSI from train → validation on a random split is low.**
# MAGIC This is expected — we did not introduce population shift. A temporal
# MAGIC split with a trend break (e.g., post-COVID frequency normalisation)
# MAGIC would produce materially higher PSI. Use real temporal splits in
# MAGIC production; the PSI test is only diagnostic here.
# MAGIC
# MAGIC **The governance machinery is model-agnostic.**
# MAGIC The same `ModelValidationReport` API works on GLM output (numpy array from
# MAGIC statsmodels predict) and GBM output (CatBoost predict × exposure) without
# MAGIC modification. The risk tier scorer and governance pack are equally
# MAGIC model-agnostic. This is the core value proposition.

# COMMAND ----------

# Summary output
print("=" * 70)
print("BENCHMARK COMPLETE: insurance-governance on freMTPL2 (OpenML 41214)")
print("=" * 70)
print()
print(f"Dataset:  freMTPL2  {len(df):,} rows  |  {df['ClaimNb'].sum():,} claims")
print(f"Exposure-weighted frequency: {df['ClaimNb'].sum()/df['Exposure'].sum():.4f}")
print()
print(f"{'Metric':<28}  {'GLM (numeric only)':>20}  {'GBM (all features)':>20}")
print("-" * 75)

for metric_name, label in [
    ("gini",    "Gini coefficient"),
    ("psi",     "PSI (train→val)"),
    ("hl_pval", "Hosmer-Lemeshow p"),
    ("ae",      "A/E ratio"),
]:
    glm_v = metrics["GLM"][metric_name]
    gbm_v = metrics["GBM"][metric_name]
    glm_s = f"{glm_v:.4f}" if glm_v is not None else "—"
    gbm_s = f"{gbm_v:.4f}" if gbm_v is not None else "—"
    print(f"  {label:<26}  {glm_s:>20}  {gbm_s:>20}")

print()
print(f"  {'Overall RAG':<26}  {glm_rag.value.upper():>20}  {gbm_rag.value.upper():>20}")
print(f"  {'Risk tier':<26}  {'Tier '+str(glm_tier.tier)+' ('+glm_tier.tier_label+')':>20}  {'Tier '+str(gbm_tier.tier)+' ('+gbm_tier.tier_label+')':>20}")
print()
print("Outputs written:")
print("  /tmp/fremtpl2_glm_validation.html   — GLM Consumer Duty and SS1/23 best-practice aligned validation report")
print("  /tmp/fremtpl2_glm_validation.json   — GLM JSON sidecar")
print("  /tmp/fremtpl2_gbm_validation.html   — GBM Consumer Duty and SS1/23 best-practice aligned validation report")
print("  /tmp/fremtpl2_gbm_validation.json   — GBM JSON sidecar")
print("  /tmp/fremtpl2_glm_mrm_pack.html     — GLM governance pack")
print("  /tmp/fremtpl2_gbm_mrm_pack.html     — GBM governance pack")
print("  /tmp/fremtpl2_diagnostics.png       — lift charts + A/E + predicted distributions")
