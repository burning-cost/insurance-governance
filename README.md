# insurance-governance

Unified model governance for UK insurance pricing teams. Combines PRA SS1/23 statistical validation and model risk management into one package.

Merged from: `insurance-validation` (PRA SS1/23 validation reports) and `insurance-mrm` (model risk management).

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
import numpy as np
from insurance_governance import (
    ModelValidationReport,
    ValidationModelCard,
    MRMModelCard,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
)

# --- Synthetic model outputs (replace with your real model predictions) ---
rng = np.random.default_rng(42)
n_val = 5_000
y_val      = rng.poisson(0.08, n_val).astype(float)   # observed claim counts
y_pred_val = np.clip(rng.normal(0.08, 0.02, n_val), 0.001, None)  # model predictions

# --- Run statistical validation ---
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
report = ModelValidationReport(model_card=card, y_val=y_val, y_pred_val=y_pred_val)
report.generate("validation_report.html")

# --- MRM governance pack ---
mrm_card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor TPPD Frequency",
    version="3.2.0",
    model_class="pricing",
    intended_use="Frequency pricing for private motor.",
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
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

Or import from subpackages directly:

```python
from insurance_governance.validation import ModelValidationReport, ModelCard as ValidationModelCard
from insurance_governance.mrm import ModelCard as MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport
```

## Note on ModelCard

Both subpackages define a `ModelCard` class, but they serve different purposes:

- `insurance_governance.validation.ModelCard` (`ValidationModelCard` at top level) — Pydantic schema, anchors the statistical validation report, captures features, methodology, limitations.
- `insurance_governance.mrm.ModelCard` (`MRMModelCard` at top level) — dataclass, anchors the MRM governance pack, captures assumptions, risk tier, Model Risk Committee metadata.

At the top level they are re-exported as `ValidationModelCard` and `MRMModelCard` to avoid ambiguity.

## Capabilities Demo

Demonstrated on synthetic motor data: 50,000 UK motor policies, CatBoost Poisson frequency model, 60/20/20 temporal train/validation/test split. Full notebook: `notebooks/benchmark.py`.

- Runs the full PRA SS1/23 validation suite in a single `ModelValidationReport` call: Gini coefficient with bootstrap 95% CI, 10-band lift chart, A/E by predicted decile with Poisson CI, Hosmer-Lemeshow goodness-of-fit, PSI on score distribution (train vs validation), monitoring plan completeness check — all returning `TestResult` objects with a pass/fail flag and human-readable detail
- Computes an overall RAG status (Green/Amber/Red) from the worst-severity failure across all tests
- Produces a self-contained HTML validation report and JSON sidecar, print-to-PDF ready, in under one second
- Scores model risk tier via `RiskTierScorer`: 6 dimensions (GWP, model complexity, deployment status, regulatory use, external data, customer-facing) mapped to a 0-100 composite with documented rules per point — no subjective judgement required at the MRC presentation
- Registers models in `ModelInventory` (JSON file, check into git alongside your code); records validation run history linked by `run_id`; lists overdue reviews
- Generates a `GovernanceReport` executive committee pack (HTML + JSON) covering model purpose, risk tier rationale, last validation RAG, assumptions register with risk ratings, outstanding issues, approval conditions, and next review date

**When to use:** You are subject to PRA SS1/23, have 10+ production pricing models, and want consistent, auditable validation and governance output rather than bespoke analyst notebooks that vary by model. Particularly useful before a PRA supervisory visit.

**When NOT to use:** You need reserving or capital model governance — this package is scoped to pricing models. It also does not replace independent human review of validation results; it automates the tests, not the judgement.

## Performance

Benchmarked on synthetic UK motor data — 50,000 policies, CatBoost Poisson frequency model, 60/20/20 temporal split. See `notebooks/benchmark.py` for the full demo workflow.

| Task | Time |
|------|------|
| Full SS1/23 validation suite (9 tests) | < 5 seconds |
| HTML validation report generation | < 1 second |
| JSON sidecar generation | < 1 second |
| RiskTierScorer.score() | < 1ms |
| ModelInventory.register() | < 10ms |
| GovernanceReport HTML generation | < 1 second |

The computational cost of this library is negligible — it runs statistical tests on pre-computed predictions, not raw data. The bottleneck is always the model fitting that produces `y_pred_val`, not the governance layer. A team generating quarterly validation reports for 15 models can run the full suite in under 10 minutes of wall clock time, most of which is the CatBoost fits.

The value is not in computation speed but in consistency: every model in the portfolio gets identical tests, identical thresholds, and identical output format. A pricing team relying on bespoke analyst notebooks has no guarantee that the Gini in one report means the same thing as the Gini in another.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — fairness audit outputs are a required input to the governance sign-off pack |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger deployment with ENBP audit logging — governance documents the model; deploy manages its lifecycle |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — ongoing monitoring evidence feeds into governance review cycles |

## Licence

MIT
