"""
Quickstart: insurance-governance

Runs a five-test statistical validation suite on synthetic motor data and
produces an HTML validation report plus an MRM governance pack.

This mirrors the README quickstart exactly — replace the numpy arrays with
your real model outputs and you have a PRA-aligned validation artefact.
"""

import numpy as np
from insurance_governance import (
    ModelValidationReport,
    ValidationModelCard,
    MRMModelCard,
    RiskTierScorer,
    GovernanceReport,
    Assumption,
)

# ---------------------------------------------------------------------------
# Synthetic motor validation data (replace with your real outputs)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 5_000
y_val = rng.poisson(0.08, n).astype(float)          # observed claim counts
y_pred_val = np.clip(rng.normal(0.08, 0.02, n), 0.001, None)  # model predictions
y_train = rng.poisson(0.08, 15_000).astype(float)
y_pred_train = np.clip(rng.normal(0.08, 0.02, 15_000), 0.001, None)
exposure_val = rng.uniform(0.5, 1.0, n)             # policy years

# ---------------------------------------------------------------------------
# Step 1 — Statistical validation
# ---------------------------------------------------------------------------
val_card = ValidationModelCard(
    name="Motor Frequency v3.2",
    version="3.2.0",
    purpose="Predict claim frequency for UK private motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count",
    features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data"],
    owner="Pricing Team",
)

report = ModelValidationReport(
    model_card=val_card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    y_train=y_train,
    y_pred_train=y_pred_train,
    exposure_val=exposure_val,
    monitoring_owner="Head of Pricing",
    monitoring_triggers={"ae_ratio": 1.10, "psi": 0.25},
)

print(f"Overall RAG: {report.get_rag_status()}")
report.generate("validation_report.html")
report.to_json("validation_report.json")
print("Validation report written to validation_report.html")

# ---------------------------------------------------------------------------
# Step 2 — MRM governance pack
# ---------------------------------------------------------------------------
mrm_card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor TPPD Frequency",
    version="3.2.0",
    model_class="pricing",
    intended_use="Frequency pricing for UK private motor. Not for commercial fleet.",
    assumptions=[
        Assumption(
            description="Claim frequency stationarity since 2022",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring; ad-hoc review if PSI > 0.25",
        ),
    ],
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=125_000_000,
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)

print(f"Risk tier: Tier {tier.tier} ({tier.tier_label}), score {tier.score}/100")
print(f"Sign-off required: {tier.sign_off_requirement}")

GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
print("Governance pack written to mrm_pack.html")
