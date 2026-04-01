"""AuditSummaryReport — HTML and JSON summary of an audit log period.

Regulators, compliance teams, and model risk committees need a periodic summary
of what an AI pricing model has been doing. This module takes a populated
ExplainabilityAuditLog and generates a report covering:

- Total decision volume, broken down by decision_basis.
- Human review rate and override rate.
- Feature importance distribution (mean absolute SHAP, sorted).
- Per-segment analysis by any categorical column in input_features.

The HTML output is fully self-contained — no external CSS, no CDN. The JSON
output contains the same data in machine-readable form for downstream ingestion.

Usage::

    from datetime import datetime, timezone
    from insurance_governance.audit import ExplainabilityAuditLog, AuditSummaryReport

    log = ExplainabilityAuditLog('audit.jsonl', 'motor-freq-v2', '3.1.0')
    report = AuditSummaryReport(log, period='2025-Q4')
    report.save_html('audit_report_2025q4.html')
    report.save_json('audit_report_2025q4.json')
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .entry import ExplainabilityAuditEntry
from .log import ExplainabilityAuditLog


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


class AuditSummaryReport:
    """Generate a summary report from an ExplainabilityAuditLog.

    All entries in the log are loaded and summarised. Use the ``start``
    and ``end`` parameters to restrict analysis to a time window.

    Parameters
    ----------
    log:
        The audit log to summarise.
    period:
        A label for the reporting period, e.g. ``'2025-Q4'``. Used in
        headings.
    start:
        Optional start of the analysis window (inclusive). Defaults to the
        earliest entry in the log.
    end:
        Optional end of the analysis window (inclusive). Defaults to the
        latest entry in the log.
    segment_feature:
        Optional feature name to use for per-segment analysis. Must exist
        as a key in ``input_features`` for each entry. If None, no segment
        breakdown is produced.
    """

    def __init__(
        self,
        log: ExplainabilityAuditLog,
        period: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        segment_feature: Optional[str] = None,
    ) -> None:
        self._log = log
        self._period = period
        self._start = start
        self._end = end
        self._segment_feature = segment_feature
        self._entries: Optional[list[ExplainabilityAuditEntry]] = None

    def _load_entries(self) -> list[ExplainabilityAuditEntry]:
        """Load and filter entries, caching the result."""
        if self._entries is not None:
            return self._entries

        all_entries = self._log.read_all()

        if self._start is None and self._end is None:
            self._entries = all_entries
            return self._entries

        start = self._start
        end = self._end
        if start is not None and start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is not None and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        filtered: list[ExplainabilityAuditEntry] = []
        for entry in all_entries:
            ts = datetime.fromisoformat(entry.timestamp_utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            filtered.append(entry)

        self._entries = filtered
        return self._entries

    def _decision_volume(
        self, entries: list[ExplainabilityAuditEntry]
    ) -> dict[str, Any]:
        """Summarise decision counts by basis."""
        total = len(entries)
        basis_counts: dict[str, int] = Counter(e.decision_basis for e in entries)
        reviewed = sum(1 for e in entries if e.human_reviewed)
        overridden = sum(1 for e in entries if e.override_applied)
        return {
            "total": total,
            "by_basis": dict(basis_counts),
            "human_reviewed": reviewed,
            "human_reviewed_pct": round(100.0 * reviewed / total, 2) if total else 0.0,
            "overridden": overridden,
            "override_rate_pct": round(100.0 * overridden / total, 2) if total else 0.0,
        }

    def _feature_importance_distribution(
        self, entries: list[ExplainabilityAuditEntry]
    ) -> list[dict[str, Any]]:
        """Compute mean absolute SHAP value per feature, sorted descending."""
        feature_abs: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            for feat, val in entry.feature_importances.items():
                feature_abs[feat].append(abs(val))

        rows: list[dict[str, Any]] = []
        for feat, values in feature_abs.items():
            rows.append({
                "feature": feat,
                "mean_abs_shap": round(_mean(values), 6),
                "std_abs_shap": round(_stdev(values), 6),
                "n_observations": len(values),
            })

        rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        return rows

    def _segment_analysis(
        self, entries: list[ExplainabilityAuditEntry], feature: str
    ) -> list[dict[str, Any]]:
        """Summarise key metrics per segment value of a categorical feature."""
        segments: dict[str, list[ExplainabilityAuditEntry]] = defaultdict(list)
        for entry in entries:
            val = entry.input_features.get(feature, "__missing__")
            segments[str(val)].append(entry)

        rows: list[dict[str, Any]] = []
        for seg_val, seg_entries in sorted(segments.items()):
            premiums = [
                e.final_premium for e in seg_entries if e.final_premium is not None
            ]
            rows.append({
                "segment": seg_val,
                "count": len(seg_entries),
                "mean_prediction": round(
                    _mean([e.prediction for e in seg_entries]), 4
                ),
                "mean_final_premium": round(_mean(premiums), 4) if premiums else None,
                "override_rate_pct": round(
                    100.0
                    * sum(1 for e in seg_entries if e.override_applied)
                    / len(seg_entries),
                    2,
                ),
            })
        return rows

    def _integrity_check(
        self, entries: list[ExplainabilityAuditEntry]
    ) -> dict[str, Any]:
        """Run verify_chain and summarise the result."""
        failures = self._log.verify_chain()
        # Filter to failures within the analysis window
        entry_ids = {e.entry_id for e in entries}
        relevant_failures = [f for f in failures if f.get("entry_id") in entry_ids]
        return {
            "total_checked": len(entries),
            "failures": len(relevant_failures),
            "pass": len(relevant_failures) == 0,
            "failure_details": relevant_failures,
        }

    def build(self) -> dict[str, Any]:
        """Compute and return the full report as a plain dict.

        The dict has the following top-level keys:
        - ``metadata``: period, model, generation timestamp.
        - ``decision_volume``: counts and rates.
        - ``feature_importance``: sorted list of mean absolute SHAP values.
        - ``segment_analysis``: per-segment breakdown (if ``segment_feature``
          was supplied).
        - ``integrity``: hash verification summary.

        Returns:
            Dict suitable for JSON serialisation.
        """
        entries = self._load_entries()
        data: dict[str, Any] = {
            "metadata": {
                "period": self._period,
                "model_id": self._log.model_id,
                "model_version": self._log.model_version,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "entry_count": len(entries),
                "start": self._start.isoformat() if self._start else None,
                "end": self._end.isoformat() if self._end else None,
            },
            "decision_volume": self._decision_volume(entries),
            "feature_importance": self._feature_importance_distribution(entries),
            "integrity": self._integrity_check(entries),
        }

        if self._segment_feature:
            data["segment_analysis"] = {
                "feature": self._segment_feature,
                "rows": self._segment_analysis(entries, self._segment_feature),
            }

        return data

    def save_json(self, path: str | Path) -> Path:
        """Write the report to a JSON file.

        Args:
            path: Output file path.

        Returns:
            The resolved :class:`~pathlib.Path` of the written file.
        """
        out = Path(path)
        data = self.build()
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return out

    def save_html(self, path: str | Path) -> Path:
        """Write the report as a self-contained HTML file.

        The HTML contains no external dependencies. It includes inline CSS
        for basic formatting and renders all sections from the :meth:`build`
        dict.

        Args:
            path: Output file path.

        Returns:
            The resolved :class:`~pathlib.Path` of the written file.
        """
        out = Path(path)
        data = self.build()
        html = self._render_html(data)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
        return out

    def _render_html(self, data: dict[str, Any]) -> str:  # noqa: C901
        """Render the report dict as an HTML string."""
        meta = data["metadata"]
        vol = data["decision_volume"]
        features = data["feature_importance"]
        integrity = data["integrity"]

        integrity_colour = "#28a745" if integrity["pass"] else "#dc3545"
        integrity_label = "PASS" if integrity["pass"] else "FAIL"

        feature_rows = ""
        for row in features[:20]:  # cap at 20 for readability
            feature_rows += (
                f"<tr><td>{row['feature']}</td>"
                f"<td>{row['mean_abs_shap']:.6f}</td>"
                f"<td>{row['std_abs_shap']:.6f}</td>"
                f"<td>{row['n_observations']}</td></tr>\n"
            )

        segment_html = ""
        if "segment_analysis" in data:
            seg = data["segment_analysis"]
            seg_rows = ""
            for row in seg["rows"]:
                mean_prem = (
                    f"{row['mean_final_premium']:.2f}"
                    if row["mean_final_premium"] is not None
                    else "n/a"
                )
                seg_rows += (
                    f"<tr><td>{row['segment']}</td>"
                    f"<td>{row['count']}</td>"
                    f"<td>{row['mean_prediction']:.4f}</td>"
                    f"<td>{mean_prem}</td>"
                    f"<td>{row['override_rate_pct']:.1f}%</td></tr>\n"
                )
            segment_html = f"""
<h2>Per-Segment Analysis — {seg['feature']}</h2>
<table>
<thead>
  <tr>
    <th>Segment</th><th>Count</th><th>Mean Prediction</th>
    <th>Mean Final Premium</th><th>Override Rate</th>
  </tr>
</thead>
<tbody>
{seg_rows}
</tbody>
</table>"""

        by_basis_rows = ""
        for basis, count in sorted(vol["by_basis"].items()):
            pct = 100.0 * count / vol["total"] if vol["total"] else 0
            by_basis_rows += (
                f"<tr><td>{basis}</td><td>{count}</td>"
                f"<td>{pct:.1f}%</td></tr>\n"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Explainability Audit Report — {meta['period']}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th {{ background: #f0f4f8; text-align: left; padding: 8px 12px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover td {{ background: #f9f9f9; }}
  .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat-card {{
    background: #f7f9fc; border: 1px solid #d0d8e4; border-radius: 6px;
    padding: 16px 20px; min-width: 160px;
  }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
  .stat-card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  .integrity-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    color: white; font-weight: bold; background: {integrity_colour};
  }}
  .meta-table td {{ border-bottom: none; padding: 4px 12px; }}
  footer {{ margin-top: 48px; font-size: 12px; color: #999; }}
</style>
</head>
<body>

<h1>Explainability Audit Report</h1>
<table class="meta-table">
<tr><td><strong>Period</strong></td><td>{meta['period']}</td></tr>
<tr><td><strong>Model ID</strong></td><td>{meta['model_id']}</td></tr>
<tr><td><strong>Model Version</strong></td><td>{meta['model_version']}</td></tr>
<tr><td><strong>Generated</strong></td><td>{meta['generated_at']}</td></tr>
<tr><td><strong>Entries Analysed</strong></td><td>{meta['entry_count']}</td></tr>
</table>

<h2>Decision Volume</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="value">{vol['total']}</div>
    <div class="label">Total Decisions</div>
  </div>
  <div class="stat-card">
    <div class="value">{vol['human_reviewed_pct']:.1f}%</div>
    <div class="label">Human-Reviewed Rate</div>
  </div>
  <div class="stat-card">
    <div class="value">{vol['override_rate_pct']:.1f}%</div>
    <div class="label">Override Rate</div>
  </div>
  <div class="stat-card">
    <div class="value"><span class="integrity-badge">{integrity_label}</span></div>
    <div class="label">Integrity Check</div>
  </div>
</div>

<h3>Decisions by Basis</h3>
<table>
<thead><tr><th>Decision Basis</th><th>Count</th><th>Share</th></tr></thead>
<tbody>
{by_basis_rows}
</tbody>
</table>

<h2>Feature Importance Distribution</h2>
<p>Mean absolute SHAP value per feature, sorted by importance. Larger values
indicate the feature had a greater average influence on premium outcomes.</p>
<table>
<thead>
  <tr>
    <th>Feature</th><th>Mean |SHAP|</th><th>Std |SHAP|</th><th>Observations</th>
  </tr>
</thead>
<tbody>
{feature_rows}
</tbody>
</table>

{segment_html}

<h2>Integrity Verification</h2>
<p>Status: <span class="integrity-badge">{integrity_label}</span>
&nbsp; Entries checked: {integrity['total_checked']}
&nbsp; Failures: {integrity['failures']}</p>

<footer>Generated by insurance-governance audit module.
Model: {meta['model_id']} v{meta['model_version']}.</footer>

</body>
</html>"""
