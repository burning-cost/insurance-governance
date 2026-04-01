"""
OutcomeTestingReport — HTML and JSON report generator for Consumer Duty outcome testing.

Follows the same pattern as ReportGenerator in the validation module:
a class that wraps results and renders them into a self-contained HTML document.

The HTML is completely self-contained: no external dependencies, no CDN.
A single file that can be emailed, stored in SharePoint, or attached to a
Confluence page as board evidence.
"""
from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from insurance_governance.mrm.model_card import ModelCard as MRMModelCard
from insurance_governance.validation.results import RAGStatus

from .results import OutcomeResult, _compute_outcome_rag


class OutcomeTestingReport:
    """
    Generate a Consumer Duty outcome testing report from an MRMModelCard
    and a list of OutcomeResults.

    Parameters
    ----------
    model_card:
        The MRM model card for the model under review.
    results:
        List of OutcomeResult objects from PriceValueMetrics, ClaimsMetrics,
        or custom tests.
    period:
        The reporting period these results cover, e.g. ``'2025-Q4'``.
    generated_date:
        Date to stamp on the report. Defaults to today.
    run_id:
        UUID string for this reporting run. Auto-generated if None.
    rag_status:
        Overall RAG status. Auto-computed from results if None.
    """

    def __init__(
        self,
        model_card: MRMModelCard,
        results: list[OutcomeResult],
        period: str,
        generated_date: date | None = None,
        run_id: str | None = None,
        rag_status: RAGStatus | None = None,
    ) -> None:
        self._model_card = model_card
        self._results = results
        self._period = period
        self._generated_date = generated_date or date.today()
        self._run_id = run_id or str(uuid.uuid4())

        if rag_status is None:
            self._rag_status = _compute_outcome_rag(results)
        else:
            self._rag_status = rag_status

        self._env = Environment(
            loader=PackageLoader("insurance_governance.outcome", "templates"),
            autoescape=select_autoescape(["html", "j2"]),
        )

    @property
    def rag_status(self) -> RAGStatus:
        return self._rag_status

    @property
    def run_id(self) -> str:
        return self._run_id

    def render_html(self) -> str:
        """
        Render the outcome testing report to an HTML string.

        Returns
        -------
        str
            Complete, self-contained HTML document.
        """
        template = self._env.get_template("outcome_report.html.j2")

        result_dicts = []
        for r in self._results:
            d = r.to_dict()
            d["severity"] = r.severity.value
            result_dicts.append(d)

        return template.render(
            model_card=self._model_card,
            results=result_dicts,
            period=self._period,
            generated_date=str(self._generated_date),
            run_id=self._run_id,
            rag_status=self._rag_status.value,
        )

    def write_html(self, path: str | Path) -> Path:
        """
        Write the HTML report to a file.

        Parameters
        ----------
        path:
            Output file path. Parent directories must exist.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        out = Path(path).resolve()
        out.write_text(self.render_html(), encoding="utf-8")
        return out

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the full report to a plain dict for JSON export.

        Returns
        -------
        dict
        """
        return {
            "run_id": self._run_id,
            "period": self._period,
            "generated_date": str(self._generated_date),
            "rag_status": self._rag_status.value,
            "model_card": self._model_card.to_dict(),
            "results": [r.to_dict() for r in self._results],
            "summary": {
                "total_tests": len(self._results),
                "passed": sum(1 for r in self._results if r.passed),
                "failed": sum(1 for r in self._results if not r.passed),
                "critical": sum(
                    1 for r in self._results
                    if not r.passed and r.severity.value == "critical"
                ),
                "warnings": sum(
                    1 for r in self._results
                    if not r.passed and r.severity.value == "warning"
                ),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        """
        Write a JSON sidecar for audit trail ingestion.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        Path
        """
        out = Path(path).resolve()
        out.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return out
