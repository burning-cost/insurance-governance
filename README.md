# insurance-governance

[![PyPI](https://img.shields.io/pypi/v/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![Tests](https://github.com/burning-cost/insurance-governance/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-governance/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-governance/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-governance/blob/main/notebooks/quickstart.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-governance/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-governance/discussions). Found it useful? A star helps others find it.

Every UK pricing team managing 10+ production models has the same problem: model validation reports are bespoke analyst notebooks, MRM governance packs are Word documents produced by hand for each Model Risk Committee, and there is no consistent way to demonstrate to a PRA supervisor that you have a functioning model risk framework. Under PS12/22 and EIOPA Solvency II validation guidelines, insurers are required to document model performance, assumptions, and risk tier — but nothing enforces that this documentation is comparable across models, or that the statistical tests actually match what SS1/23 Principle 3 describes.

This library automates both sides: it runs a five-test statistical validation suite (Gini with bootstrap CI, A/E with Poisson CI, Hosmer-Lemeshow, lift chart, PSI) and produces MRM governance packs (risk tier scoring, assumptions register, approval conditions) as self-contained HTML files. The output is the same structure for every model, every release.

Merged from: `insurance-validation` (model validation reports) and `insurance-mrm` (model risk management).

**Blog post:** [One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified](https://burning-cost.github.io/2026/03/14/insurance-governance-unified-pra-ss123-validation/)

**Regulatory note:** PRA SS1/23 is directed at banks and building societies. Insurance model risk management is governed by PS12/22, Solvency II internal model requirements, and EIOPA validation guidelines. In practice many UK insurance MRM frameworks reference SS1/23 by analogy — it articulates sound MRM principles regardless of firm type — and the PRA has encouraged insurers to take note. This library uses SS1/23 as a reference framework in that spirit. Map your own obligations to your actual regulatory basis.

---

## Part of the Burning Cost stack

Takes statistical test outputs from validation runs and fairness audit results from [insurance-fairness](https://github.com/burning-cost/insurance-fairness). Feeds governance packs and model inventory records into pricing committee sign-off workflows. Receives monitoring evidence from [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) to surface overdue reviews. → [See the full stack](https://burning-cost.github.io/stack/)

---

## Manual governance vs this library

| Task | Manual approach | insurance-governance |
|------|----------------|----------------------|
| Statistical validation | Bespoke notebook per model — different tests, different output formats, incomparable across models | `ModelValidationReport` — five fixed tests, same HTML structure, every model |
| A/E miscalibration detection | One aggregate A/E ratio — misses segment-level bias | A/E with Poisson CI + Hosmer-Lemeshow; detects age-band bias that averages out in the global A/E (HL p < 0.0001 on Model B in benchmark) |
| Score distribution drift | Paste PSI value into Word | PSI test result in JSON sidecar, with pass/fail flag and threshold detail |
| Risk tier assignment | Subjective judgement in MRC pre-read | `RiskTierScorer` — 6 dimensions, 0–100 composite, documented rationale string per dimension; no judgement call at the committee |
| Governance pack | Word document, rebuilt for each committee cycle | `GovernanceReport.save_html()` — self-contained HTML with model purpose, tier rationale, assumptions register, outstanding issues, approval conditions; print-to-PDF in under 1 second |
| Model inventory | Spreadsheet or SharePoint list | `ModelInventory` — JSON file, check into git; records validation history by `run_id`, lists models with overdue reviews |
| Consistency across 10+ models | Each model owner formats differently | Same import, same structure, same output format for every model |

---

## Quick start

```python
import numpy as np
from insurance_governance import (
    ModelValidationReport, ValidationModelCard,
    MRMModelCard, RiskTierScorer, GovernanceReport, Assumption,
)

rng = np.random.default_rng(42)
n = 5_000
y_val      = rng.poisson(0.08, n).astype(float)
y_pred_val = np.clip(rng.normal(0.08, 0.02, n), 0.001, None)
exposure   = rng.uniform(0.5, 1.0, n)

val_card = ValidationModelCard(
    name="Motor Frequency v3.2", version="3.2.0",
    purpose="Predict claim frequency for UK private motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count", features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data"], owner="Pricing Team",
)
report = ModelValidationReport(
    model_card=val_card, y_val=y_val, y_pred_val=y_pred_val, exposure_val=exposure,
    monitoring_owner="Head of Pricing", monitoring_triggers={"ae_ratio": 1.10, "psi": 0.25},
)
print(report.get_rag_status())   # GREEN / AMBER / RED
report.generate("validation_report.html")

mrm_card = MRMModelCard(
    model_id="motor-freq-v3", model_name="Motor TPPD Frequency",
    version="3.2.0", model_class="pricing",
    intended_use="Frequency pricing for UK private motor. Not for commercial fleet.",
    assumptions=[Assumption("Claim frequency stationarity since 2022", risk="MEDIUM",
                            mitigation="Quarterly A/E monitoring")],
)
tier = RiskTierScorer().score(
    gwp_impacted=125_000_000, model_complexity="high",
    deployment_status="champion", regulatory_use=False,
    external_data=False, customer_facing=True,
)
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

See `examples/quickstart.py` for a fully self-contained example with synthetic data, training/validation split, and JSON sidecar output.

---

## Features

**Validation (`insurance_governance.validation`)**

- `ModelValidationReport` — single-call facade: pass your model card, validation arrays, and optional training arrays; get back a `RAGStatus` and an HTML report
- Gini coefficient with 500-resample bootstrap 95% CI
- Actual vs expected by predicted decile with Poisson confidence intervals — catches segment-level miscalibration that a single aggregate A/E misses
- Hosmer-Lemeshow goodness-of-fit test — flags calibration failure by group
- 10-band lift chart with per-band A/E ratios
- PSI on score distribution (train vs validation) — flags population drift
- Monitoring plan completeness check (SS1/23 Principle 5) — requires named owner and trigger thresholds
- Double-lift chart against incumbent model when `incumbent_pred_val` is provided
- Optional data quality checks (missing values, outliers, cardinality) when Polars feature DataFrames are supplied
- Optional fairness/disparate impact section when `fairness_group_col` is supplied
- Self-contained HTML output (no CDN, no JS) and JSON sidecar for downstream MRM system ingestion
- All tests return `TestResult(passed, severity, detail)` — extend with your own results via `extra_results`

**MRM (`insurance_governance.mrm`)**

- `RiskTierScorer` — stateless, deterministic; 6 dimensions (materiality/GWP, complexity, external data, validation recency, drift history, regulatory exposure); 0–100 composite; verbose rationale string per dimension; configurable weights and thresholds
- Tier 1 (Critical, 60+): annual review, MRC sign-off; Tier 2 (Significant, 30–59): 18-month, Chief Actuary; Tier 3 (Informational, <30): 24-month, Head of Pricing
- `MRMModelCard` — structured governance record: model identity, intended use, assumptions with risk ratings (LOW/MEDIUM/HIGH) and mitigations, limitations, approval history, monitoring plan, last validation run ID and RAG, next review date
- `ModelInventory` — JSON file registry; `register()`, `list_overdue()`, `get_history()` by model ID; designed to be checked into git alongside model code
- `GovernanceReport` — executive committee HTML pack covering model purpose, risk tier rationale, last validation RAG, assumptions register, outstanding issues, approval conditions, and next review date; JSON output for Confluence/MRC portal ingestion

---

## Installation

```bash
pip install insurance-governance
# or
uv add insurance-governance
```

**Dependencies:** numpy, jinja2 (for HTML reports). Polars is optional — data quality and feature drift checks activate when it is present.

---

## Subpackage imports

```python
# Top-level (recommended for most users)
from insurance_governance import (
    ModelValidationReport, ValidationModelCard,
    MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport, Assumption,
)

# Subpackage imports (for custom workflows)
from insurance_governance.validation import ModelValidationReport, ModelCard as ValidationModelCard
from insurance_governance.mrm import ModelCard as MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport
```

Both subpackages define a `ModelCard` class serving different purposes. At the top level they are re-exported as `ValidationModelCard` and `MRMModelCard` to avoid ambiguity.

---

## Note on ModelCard

- `ValidationModelCard` (`insurance_governance.validation.ModelCard`) — Pydantic schema anchoring the statistical validation report; captures features, methodology, limitations, monitoring plan.
- `MRMModelCard` (`insurance_governance.mrm.ModelCard`) — dataclass anchoring the MRM governance pack; captures assumptions with risk ratings, approval history, monitoring triggers, last validation run linkage.

---

## Performance

Benchmarked on Databricks (2026-03-16) using synthetic UK motor data: 20,000 training + 8,000 validation policies, three model scenarios — well-specified (Model A), miscalibrated (Model B, A/E=1.18 with age-band bias), and drifted (Model C, trained on a shifted population). Full script: `benchmarks/benchmark_insurance_governance.py`.

| Approach | Time |
|----------|------|
| Manual 4-check checklist | 0.09s |
| Automated 5-test suite | 1.17s |

The 1-second overhead is entirely the 500-resample bootstrap for the Gini CI.

**What the automated suite catches that manual does not.**

Model B (miscalibrated): both methods flag the A/E deviation. Only the automated suite runs Hosmer-Lemeshow, which detects the age-band-level miscalibration that averages out in the global A/E: HL p < 0.0001. A manual checklist computing one aggregate A/E number cannot surface this pattern without additional code.

Model C (drifted): PSI = 0.189 — below the 0.25 threshold, so the manual checklist passes on PSI. The automated suite catches it because the A/E CI excludes 1.0. PSI alone is not sufficient to detect this type of drift.

| Scenario | Manual verdict | Automated verdict | Key diagnostic |
|----------|---------------|-------------------|----------------|
| Model A (well-specified) | 4/4 pass | 5/5 pass | Gini CI, A/E CI both tight |
| Model B (miscalibrated) | Flags A/E | Flags A/E + HL | HL p<0.0001 — age-band bias |
| Model C (drifted) | Passes PSI | Flags A/E CI | PSI=0.189 (below threshold); A/E CI excludes 1.0 |

---

## Databricks Notebook

A ready-to-run Databricks notebook is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_governance_demo.py).

---

## When to use / when not to use

**Use this when** you have 10+ production pricing models and want consistent, auditable validation and governance output rather than bespoke analyst notebooks that vary by model. Particularly useful before a PRA supervisory visit or ahead of a Consumer Duty fair value assessment cycle.

**Do not use this for** reserving or capital model governance — this package is scoped to pricing models. It does not replace independent human review of validation results; it automates the tests, not the judgement.

---

## Related libraries

| Library | Description |
|---------|-------------|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — ongoing monitoring evidence feeds into governance review cycles |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — fairness audit outputs are a required input to the governance sign-off pack |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals with finite-sample coverage guarantees, for PRA SS1/23 validation packs |

---

## Licence

MIT
