# insurance-governance

[![PyPI](https://img.shields.io/pypi/v/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-governance/blob/main/notebooks/quickstart.ipynb)

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-governance/discussions). Found it useful? A ⭐ helps others find it.

Unified model governance for UK insurance pricing teams. Combines model validation and model risk management into one package, with tests and outputs structured to align with the principles of PRA SS1/23 (as adapted for insurance).

Merged from: `insurance-validation` (model validation reports) and `insurance-mrm` (model risk management).

The problem this solves: validation tests and MRM governance packs were built separately and had separate installs, separate version pinning, and separate import paths. Pricing teams either installed both and managed the coupling themselves, or skipped one. This package resolves that by providing a single install.

**Regulatory note:** PRA SS1/23 is a supervisory statement directed at banks and building societies, not insurers. Insurance model risk management is governed directly by PS12/22, Solvency II internal model requirements, and EIOPA validation guidelines. In practice, many UK insurance MRM frameworks reference SS1/23 by analogy — it articulates sound MRM principles regardless of firm type — and the PRA has encouraged insurers to take note. This library uses SS1/23 as a reference framework in that spirit: the validation tests and governance structure reflect its principles, but you should map your own obligations to your actual regulatory basis (PS12/22 or equivalent).

## Subpackages

### `insurance_governance.validation`

Model validation report generator, aligned with the principles of PRA SS1/23 (as adapted for insurance). Runs statistical tests (Gini, PSI, discrimination checks, Hosmer-Lemeshow, lift charts) and produces self-contained HTML reports.

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
y_val        = rng.poisson(0.08, n_val).astype(float)          # observed claim counts
y_pred_val   = np.clip(rng.normal(0.08, 0.02, n_val), 0.001, None)  # model predictions
exposure_val = rng.uniform(0.5, 1.0, n_val)                    # policy years (required for A/E)

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
report = ModelValidationReport(
    model_card=card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exposure_val,
)
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

- Runs a full validation suite in a single `ModelValidationReport` call: Gini coefficient with bootstrap 95% CI, 10-band lift chart, A/E by predicted decile with Poisson CI, Hosmer-Lemeshow goodness-of-fit, PSI on score distribution (train vs validation), monitoring plan completeness check — all returning `TestResult` objects with a pass/fail flag and human-readable detail
- Computes an overall RAG status (Green/Amber/Red) from the worst-severity failure across all tests
- Produces a self-contained HTML validation report and JSON sidecar, print-to-PDF ready, in under one second
- Scores model risk tier via `RiskTierScorer`: 6 dimensions (GWP, model complexity, deployment status, regulatory use, external data, customer-facing) mapped to a 0-100 composite with documented rules per point — no subjective judgement required at the MRC presentation
- Registers models in `ModelInventory` (JSON file, check into git alongside your code); records validation run history linked by `run_id`; lists overdue reviews
- Generates a `GovernanceReport` executive committee pack (HTML + JSON) covering model purpose, risk tier rationale, last validation RAG, assumptions register with risk ratings, outstanding issues, approval conditions, and next review date

**When to use:** You have 10+ production pricing models and want consistent, auditable validation and governance output rather than bespoke analyst notebooks that vary by model. The framework is structured around the principles of PRA SS1/23 — insurers should map those principles to their own regulatory basis (PS12/22, EIOPA guidelines). Particularly useful before a PRA supervisory visit.

**When NOT to use:** You need reserving or capital model governance — this package is scoped to pricing models. It also does not replace independent human review of validation results; it automates the tests, not the judgement.


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_governance_demo.py).

## Performance

Benchmarked on Databricks (2026-03-16) using synthetic UK motor data: 20,000 training + 8,000 validation policies, three model scenarios — well-specified (Model A), miscalibrated (Model B, A/E=1.18 with age-band bias), and drifted (Model C, trained on a shifted population). The comparison is the library's automated 5-test suite against a manual 4-check checklist. See `benchmarks/benchmark_insurance_governance.py` for the full script.

**Runtime.** On an 8,000-row validation set:

| Approach | Time |
|----------|------|
| Manual 4-check checklist | 0.09s |
| Automated 5-test suite (Gini + bootstrap CI, A/E + Poisson CI, Hosmer-Lemeshow, lift chart, PSI) | 1.17s |

The automated suite is ~13× slower in wall clock time; that 1-second overhead is entirely the 500-resample bootstrap for the Gini confidence interval.

**What the automated suite catches that the manual checklist misses.**

The key test is Model B (miscalibrated). Both methods flag the A/E deviation. But only the automated suite runs Hosmer-Lemeshow, which detects the age-band-level miscalibration that averages out in the global A/E: HL p < 0.0001 (reject calibration by group). The manual checklist, which computes one aggregate A/E number, cannot surface this pattern without additional code.

For Model C (drifted population), PSI on the score distribution = 0.189, flagging distributional shift. The manual checklist includes PSI too, so both methods agree here — but only the automated suite attaches a Poisson confidence interval to the A/E ratio, which is what lets you distinguish genuine drift from sampling noise.

| Scenario | Manual verdict | Automated verdict | Key diagnostic |
|----------|---------------|-------------------|----------------|
| Model A (well-specified) | 4/4 pass | 5/5 pass | Gini CI, A/E CI both tight |
| Model B (miscalibrated) | Flags A/E | Flags A/E + HL | HL p<0.0001 — age-band bias |
| Model C (drifted) | Flags PSI | Flags PSI + A/E CI | PSI=0.189, CI excludes 1.0 |

The runtime difference does not matter in practice — governance validation runs once per model release, not in a hot loop. The return is consistent, audit-ready output for all three scenarios: every test produces a `TestResult` with `passed`, `severity`, and a detail string ready for a validation pack.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — fairness audit outputs are a required input to the governance sign-off pack |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger deployment with ENBP audit logging — governance documents the model; deploy manages its lifecycle |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — ongoing monitoring evidence feeds into governance review cycles |

## Licence

MIT
