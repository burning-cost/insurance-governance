# Changelog

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
- Add pdoc API documentation workflow with GitHub Pages deployment
- Add benchmark: automated MRM governance validation vs manual checklist
- fix: add insurance-fairness to dev deps for fairness test coverage
- Improve test coverage from 66% to 77% by testing dependency code paths
- Fix ModelInventory._save_registry on Databricks serverless (v0.1.1)
- Add shields.io badge row to README
- docs: add Databricks notebook link
- Add Related Libraries section to README
- Merge branch 'main' of https://github.com/burning-cost/insurance-governance
