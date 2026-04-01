# Changelog

## [0.3.0] - 2026-04-01

### Added
- New `audit/` subpackage: explainability audit trail for insurance pricing models.
- `ExplainabilityAuditEntry`: tamper-evident dataclass capturing one model prediction
  event. SHA-256 hash of all fields except `entry_hash` enables immutability
  verification. Fields include SHAP feature importances, raw prediction, final
  premium, human reviewer identity (SM&CR CF reference), override flag and reason,
  and decision basis.
- `ExplainabilityAuditLog`: append-only JSONL log. Supports `read_since()` filtering,
  `verify_chain()` hash integrity check across all entries, and `export_period()` for
  regulatory submission with a metadata header line.
- `SHAPExplainer`: wraps the optional `shap` library. Supports tree, linear, kernel,
  and deep explainer types. Returns signed SHAP values as plain dicts keyed by
  feature name. `shap` is an optional dependency; raises `ImportError` with clear
  install instructions if not present.
- `PlainLanguageExplainer`: converts SHAP values to plain English sentences suitable
  for FCA PRIN 2A Consumer Duty customer communications. Supports GBP/EUR/USD, per-
  factor pound impact scaling, override and rule-fallback notes, and bullet-list output.
- `AuditSummaryReport`: builds HTML and JSON audit summaries covering decision volume,
  feature importance distribution (mean absolute SHAP), human override rates, per-
  segment analysis, and hash integrity status. HTML is fully self-contained.
- Top-level re-exports for all five audit classes added to `insurance_governance`.
- 65+ tests in `tests/test_audit.py` covering entry creation and hash verification,
  log operations, plain language output, report generation, and SHAPExplainer with
  mocked shap library.

### Changed
- Version bumped from 0.2.0 to 0.3.0.
- `pyproject.toml` keywords updated to include `explainability`, `SHAP`, `audit trail`.


## [0.1.10] - 2026-03-31

### Changed
- Corrected SS1/23 regulatory framing throughout: SS1/23 is a banking supervisory statement and does not apply directly to Solvency II insurers. All documentation now uses "aligned with SS1/23 best practice" language rather than implying direct applicability.
- Added Consumer Duty (PRIN 2A) and TR24/2 as the primary mandatory regulatory hooks for GI pricing model governance in README and governance pack template.
- Added PRA SoP3/24 (IMOR annual attestation) as the PRA-side governance expectation in README regulatory framework table.
- Updated pyproject.toml description and keywords to reflect Consumer Duty, TR24/2, SoP3/24, and IMOR rather than SS1/23 as the primary framing.
- Updated report.html.j2 template: regulatory reference table now cites Consumer Duty (PRIN 2A) + TR24/2 for performance validation and model inventory; SS1/23 references replaced with "good MRM practice (aligned with SS1/23 best practice)" framing; report header and footer updated.
- Updated CONTRIBUTING.md: clarified that SS1/23 citations in code should use "aligned with SS1/23 best practice" language.
- Updated source module docstrings to use consistent "aligned with SS1/23 best practice" phrasing.


## [0.1.5] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.1.3 (2026-03-22) [unreleased]
- fix: correct license badge (BSD-3 -> MIT) and add missing Homepage URL
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)
- fix: correct Model C PSI verdict and benchmark filename reference

## v0.1.3 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Remove duplicate CTA from README
- Add blog post link and community CTA to README
- Add MIT license
- Add discussions link and star CTA
- Add real benchmark numbers to Performance section (Databricks 2026-03-16)
- Remove pydantic v2 dependency for Databricks serverless compatibility
- Fix regulatory accuracy: SS1/23 applies by analogy to insurers, not directly
- Add PyPI classifiers for financial/insurance audience
- Add Colab quickstart notebook and Open in Colab badge
- Fix P1 bugs: exposure in A/E ratio, HL degrees of freedom, tier assignment order, Tier 4 dead code
- Fix docs workflow: use pdoc not pdoc3 syntax (no --html flag)
