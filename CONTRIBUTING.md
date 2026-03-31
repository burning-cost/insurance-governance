# Contributing to insurance-governance

This library produces validation and MRM governance artefacts for UK insurance pricing models. The regulatory context — FCA Consumer Duty (PRIN 2A), TR24/2, PRA SoP3/24, and SS1/23 best practice — is specific and the outputs need to be accurate. Contributions that improve regulatory alignment or practical usability are welcome.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_governance; print(insurance_governance.__version__)`)
- Which subpackage the bug is in (`validation` or `mrm`)
- A minimal reproducible example using the synthetic data generators
- What you expected and what actually happened

If the bug relates to a specific validation test being computed incorrectly, cite the relevant methodology reference. Regulatory accuracy matters here.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Useful areas: additional PRA/FCA regulatory frameworks (Solvency II internal model validation, IFRS 17 model documentation), integration with specific pricing platforms, and MRM workflow automation.

If you are in a UK insurer's pricing or model risk team and have a specific governance requirement this library does not cover, that is exactly the kind of input that shapes the roadmap.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-governance.git
cd insurance-governance
uv sync --dev
uv run pytest
```

The library uses `uv` for dependency management. Python 3.10+ is required. Both subpackages (`validation` and `mrm`) are tested together.

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation — this is a UK regulatory library and the language should reflect that
- Docstrings follow NumPy format and cite the relevant regulatory reference (Consumer Duty principle, FCA rule, or Equality Act section) where a method implements a specific requirement
- Where SS1/23 is cited, use language like "aligned with SS1/23 best practice" — SS1/23 is a banking supervisory statement; insurers follow it by analogy, not by direct obligation
- Generated reports are Markdown — keep the templates readable as source, not just as rendered output
- Tests should verify that report outputs contain the required sections, not just that the code runs
