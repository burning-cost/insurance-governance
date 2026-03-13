# insurance-governance

Unified model governance for UK insurance pricing teams. Combines PRA SS1/23 statistical validation and model risk management into one package.

The problem this solves: validation tests and MRM governance packs were built separately and had separate installs, separate version pinning, and separate import paths. Pricing teams either installed both and managed the coupling themselves, or skipped one. This package resolves that by providing a single install.

## Subpackages

### `insurance_governance.validation`

PRA SS1/23 compliant model validation report generator. Runs statistical tests (Gini, PSI, discrimination checks, Hosmer-Lemeshow, lift charts) and produces self-contained HTML reports.

### `insurance_governance.mrm`

Model risk management framework. ModelCard metadata container, RiskTierScorer (objective 0-100 composite score mapping to Tier 1/2/3), ModelInventory (JSON file registry), GovernanceReport (executive committee pack).

## Install

```bash
pip install insurance-governance
```

## Quick start

```python
from insurance_governance.validation import ModelValidationReport, ModelCard as ValidationModelCard
from insurance_governance.mrm import ModelCard as MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport

# Run statistical validation
card = ValidationModelCard(
    name="Motor Frequency v3.2",
    version="3.2.0",
    purpose="Predict claim frequency for UK motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count",
    features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data"],
    owner="Pricing Team",
)
report = ModelValidationReport(model_card=card, y_val=y_val, y_pred_val=y_pred_val, ...)
report.generate("validation_report.html")

# MRM governance pack
mrm_card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor TPPD Frequency",
    version="3.2.0",
    model_class="pricing",
    intended_use="Frequency pricing for private motor.",
)
scorer = RiskTierScorer()
tier = scorer.score(gwp_impacted=125_000_000, model_complexity="high", ...)
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

## Note on ModelCard

Both subpackages define a `ModelCard` class, but they serve different purposes:

- `insurance_governance.validation.ModelCard` — Pydantic schema, anchors the statistical validation report, captures features, methodology, limitations.
- `insurance_governance.mrm.ModelCard` — dataclass, anchors the MRM governance pack, captures assumptions, risk tier, Model Risk Committee metadata.

Import with explicit aliases to avoid confusion.
