# insurance-governance

Automated statistical validation and MRM governance pack generation for UK pricing models — so your next PRA supervisory visit has consistent, auditable evidence across every model in production, not a folder of bespoke analyst notebooks.

[![PyPI](https://img.shields.io/pypi/v/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-governance)](https://pypi.org/project/insurance-governance/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-governance/blob/main/LICENSE)

---

## Why this?

Under PS12/22, Solvency II validation requirements, and EIOPA guidelines, UK insurers must document model performance, assumptions, and risk tier — but nothing enforces that this documentation is comparable across models, or that the statistical tests match what SS1/23 Principle 3 describes. In practice, pricing teams with 10+ production models end up with bespoke validation notebooks that vary by analyst, Word-document MRM packs rebuilt by hand each committee cycle, and no machine-readable inventory of overdue reviews.

This library runs a five-test validation suite (Gini with bootstrap CI, A/E with Poisson CI, Hosmer-Lemeshow, lift chart, PSI) and produces MRM governance packs as self-contained HTML — the same structure for every model, every release.

> **Scope:** This package is for pricing models. It does not cover reserving or capital model governance, and does not replace human review of validation results — it automates the tests, not the judgement. SS1/23 is directed at banks; many UK insurance MRM frameworks reference it by analogy. Map to your own regulatory basis.

---

## Manual governance vs this library

| Task | Manual approach | insurance-governance |
|------|----------------|----------------------|
| Statistical validation | Bespoke notebook per model — different tests, incomparable output | `ModelValidationReport` — five fixed tests, same HTML structure, every model |
| A/E miscalibration | One aggregate ratio — misses segment-level bias | A/E with Poisson CI + Hosmer-Lemeshow; catches age-band bias averaging out in global A/E |
| Score distribution drift | PSI value pasted into Word | PSI in JSON sidecar with pass/fail flag and threshold detail |
| Risk tier assignment | Subjective judgement in MRC pre-read | `RiskTierScorer` — 6 dimensions, 0–100 composite, documented rationale per dimension |
| Governance pack | Word document rebuilt each cycle | `GovernanceReport.save_html()` — self-contained HTML; print-to-PDF in under 1 second |
| Model inventory | Spreadsheet or SharePoint list | `ModelInventory` — JSON file, check into git; tracks validation history and overdue reviews |

---

## Installation

```bash
pip install insurance-governance
```

---

## Quick start: statistical validation

Run the five-test validation suite and produce an HTML report for a motor frequency model.

```python
import numpy as np
from insurance_governance import ModelValidationReport, ValidationModelCard

rng = np.random.default_rng(42)
n = 5_000
y_val      = rng.poisson(0.08, n).astype(float)
y_pred_val = np.clip(rng.normal(0.08, 0.02, n), 0.001, None)
exposure   = rng.uniform(0.5, 1.0, n)

card = ValidationModelCard(
    name="Motor Frequency v3.2", version="3.2.0",
    purpose="Predict claim frequency for UK private motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count", features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data"], owner="Pricing Team",
)
report = ModelValidationReport(
    model_card=card, y_val=y_val, y_pred_val=y_pred_val, exposure_val=exposure,
    monitoring_owner="Head of Pricing", monitoring_triggers={"ae_ratio": 1.10, "psi": 0.25},
)
print(report.get_rag_status())   # "green", "amber", or "red"
report.generate("validation_report.html")
```

## Quick start: MRM governance pack

Score a model's risk tier and produce a governance HTML for the Model Risk Committee.

```python
from insurance_governance import MRMModelCard, RiskTierScorer, GovernanceReport, Assumption

card = MRMModelCard(
    model_id="motor-freq-v3", model_name="Motor TPPD Frequency",
    version="3.2.0", model_class="pricing",
    intended_use="Frequency pricing for UK private motor. Not for commercial fleet.",
    assumptions=[Assumption("Claim frequency stationarity since 2022",
                            risk="MEDIUM", mitigation="Quarterly A/E monitoring")],
)
tier = RiskTierScorer().score(
    gwp_impacted=125_000_000, model_complexity="high",
    deployment_status="champion", regulatory_use=False,
    external_data=False, customer_facing=True,
)
GovernanceReport(card=card, tier=tier).save_html("mrm_pack.html")
```

---

## What the validation suite catches

Benchmarked on Databricks (2026-03-16), three synthetic UK motor scenarios: well-specified (A), miscalibrated (B, A/E=1.18 with age-band bias), drifted (C, trained on shifted population).

| Scenario | Manual 4-check checklist | Automated 5-test suite | Key diagnostic |
|----------|--------------------------|------------------------|----------------|
| Model A (well-specified) | 4/4 pass | 5/5 pass | Gini CI and A/E CI both tight |
| Model B (miscalibrated) | Flags A/E | Flags A/E + HL | HL p<0.0001 — age-band bias averages out in global A/E |
| Model C (drifted) | Passes PSI | Flags A/E CI | PSI=0.189 (below 0.25 threshold); A/E CI excludes 1.0 |

The 1-second overhead over a manual checklist is entirely the 500-resample Gini bootstrap. PSI alone is not sufficient to detect population drift of this type.

---

## freMTPL2 real-data benchmark

**[notebooks/benchmark_fremtpl2.py](notebooks/benchmark_fremtpl2.py)** — Databricks notebook running the full validation suite on freMTPL2 (OpenML 41214), 677,991 French MTPL policies.

This is the benchmark to look at if you want to understand what validation outputs look like on real data — not synthetic. It runs a Poisson GLM and a CatBoost GBM on the same dataset and produces side-by-side PRA SS1/23-aligned reports.

Key findings from the real-data benchmark:

- **Gini is lower than synthetic data suggests.** Real-world motor frequency models achieve Gini of 0.15–0.30 (GLM) and 0.25–0.40 (GBM) on heterogeneous populations. Synthetic benchmarks with clean DGPs produce inflated Gini values. Calibrate your Green/Amber/Red thresholds to your actual portfolio.
- **Hosmer-Lemeshow catches what global A/E misses.** On 677K rows, H-L has power to detect systematic miscalibration that averages out in a single A/E ratio. The GLM's exclusion of categorical features leaves residuals in urban/young-driver segments — invisible in global A/E, visible in H-L.
- **The governance API is model-agnostic.** `ModelValidationReport` takes a numpy array. It does not care whether that array came from statsmodels, CatBoost, or anything else. The same validation structure applies to both models without modification.

---

## Key classes

**Validation**

- `ModelValidationReport` — facade: pass model card and validation arrays, get `RAGStatus` and HTML. Optionally add `incumbent_pred_val` for a double-lift chart, or `fairness_group_col` for a disparate impact section.
- `ValidationModelCard` — Pydantic schema: model name, version, features, methodology, monitoring plan.
- All tests return `TestResult(passed, severity, detail)` — extend with your own via `extra_results`.

**MRM**

- `RiskTierScorer` — stateless, deterministic; 6 dimensions (GWP materiality, complexity, external data, validation recency, drift history, regulatory exposure); 0–100 composite; verbose rationale per dimension. Tier 1 (≥60): annual review, MRC sign-off. Tier 2 (30–59): 18-month, Chief Actuary. Tier 3 (<30): 24-month, Head of Pricing.
- `MRMModelCard` — assumptions register (risk ratings LOW/MEDIUM/HIGH, mitigations), approval history, last validation run linkage.
- `ModelInventory` — JSON file registry checked into git; `register()`, `list_overdue()`, `get_history()`.
- `GovernanceReport` — executive HTML pack: model purpose, tier rationale, last RAG, assumptions, outstanding issues, approval conditions, next review date.

---

## Part of the Burning Cost stack

Takes validation outputs from [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) to surface overdue reviews. Accepts fairness audit results from [insurance-fairness](https://github.com/burning-cost/insurance-fairness) as a governance pack input. Feeds into pricing committee sign-off workflows. → [See the full stack](https://burning-cost.github.io/stack/)

---

## Databricks notebooks

- **Synthetic data demo** — [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_governance_demo.py): full end-to-end workflow on 50K synthetic UK motor policies.
- **Real-data benchmark** — [notebooks/benchmark_fremtpl2.py](notebooks/benchmark_fremtpl2.py): Poisson GLM vs CatBoost GBM on freMTPL2 (677K French MTPL rows, OpenML 41214). Shows what validation outputs look like in practice.

---

## Licence

MIT
