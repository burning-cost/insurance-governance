# Changelog

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
- Add benchmark: automated PRA SS1/23 validation vs manual checklist
- fix: add insurance-fairness to dev deps for fairness test coverage
- Improve test coverage from 66% to 77% by testing dependency code paths
- Fix ModelInventory._save_registry on Databricks serverless (v0.1.1)
- Add shields.io badge row to README
- docs: add Databricks notebook link
- Add Related Libraries section to README
- Merge branch 'main' of https://github.com/burning-cost/insurance-governance

