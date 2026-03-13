"""GovernanceReport: executive committee format report for insurance pricing models.

This is the 2-3 page governance pack for a Model Risk Committee or Chief Actuary.
It is NOT the full technical validation report — that lives in insurance-validation.
This report answers the questions a committee member asks:
  - What does this model do and who is responsible for it?
  - What risk tier is it and why?
  - Did the last validation pass?
  - What are the material risks and outstanding issues?
  - Who approved it, when, and when is the next review?

HTML output: self-contained, no CDN dependencies, print-to-PDF ready.
JSON output: structured dict for downstream ingestion (Confluence, MRC portals, etc.)

The report accepts optional validation_results and monitoring_results as plain
dicts, so it works standalone without insurance-validation or insurance-monitoring
installed. If you pass richer objects from those libraries, wrap them in a dict
first using their to_dict() or to_json() methods.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from .model_card import ModelCard
from .scorer import TierResult, TIER_LABELS


# ---------------------------------------------------------------------------
# Colour constants for HTML output
# ---------------------------------------------------------------------------

_RAG_COLOURS = {
    "GREEN": ("#22c55e", "#f0fdf4", "#166534"),
    "AMBER": ("#f59e0b", "#fffbeb", "#92400e"),
    "RED": ("#ef4444", "#fef2f2", "#991b1b"),
}
_TIER_COLOURS = {
    1: ("#dc2626", "#fef2f2"),   # red
    2: ("#f59e0b", "#fffbeb"),   # amber
    3: ("#3b82f6", "#eff6ff"),   # blue
    4: ("#22c55e", "#f0fdf4"),   # green
}
_DEFAULT_COLOUR = ("#6b7280", "#f9fafb")


class GovernanceReport:
    """Executive committee format governance report for a pricing model.

    Args:
        card: The :class:`~insurance_mrm.model_card.ModelCard` for the model.
        tier: Optional :class:`~insurance_mrm.scorer.TierResult`. If omitted,
            the tier information on the card (if any) is used.
        validation_results: Optional dict of validation results. Expected keys
            (all optional): ``overall_rag``, ``run_id``, ``run_date``,
            ``gini``, ``ae_ratio``, ``psi_score``, ``hl_p_value``,
            ``section_results`` (list of dicts with ``section``, ``status``,
            ``notes``).
        monitoring_results: Optional dict of monitoring results. Expected keys
            (all optional): ``period``, ``ae_ratio``, ``psi_score``, ``gini``,
            ``recommendation``, ``triggered_alerts`` (list of strings).
        report_date: ISO date string for the report. Defaults to today.

    Examples::

        report = GovernanceReport(
            card=card,
            tier=tier_result,
            validation_results={
                'overall_rag': 'GREEN',
                'run_id': 'abc-123',
                'run_date': '2025-10-01',
                'gini': 0.42,
                'ae_ratio': 1.01,
                'psi_score': 0.08,
            },
            monitoring_results={
                'period': '2025-Q3',
                'ae_ratio': 1.02,
                'psi_score': 0.06,
                'recommendation': 'Continue',
            },
        )
        html = report.to_html()
        d = report.to_dict()
    """

    def __init__(
        self,
        card: ModelCard,
        tier: Optional[TierResult] = None,
        validation_results: Optional[dict[str, Any]] = None,
        monitoring_results: Optional[dict[str, Any]] = None,
        report_date: Optional[str] = None,
    ) -> None:
        self.card = card
        self.tier = tier
        self.validation_results = validation_results or {}
        self.monitoring_results = monitoring_results or {}
        self.report_date = report_date or datetime.now(timezone.utc).date().isoformat()

    # ------------------------------------------------------------------
    # Resolved accessors
    # ------------------------------------------------------------------

    @property
    def _tier_number(self) -> Optional[int]:
        if self.tier is not None:
            return self.tier.tier
        return self.card.materiality_tier

    @property
    def _tier_label(self) -> str:
        t = self._tier_number
        if t is None:
            return "Not assessed"
        return TIER_LABELS.get(t, "Unknown")

    @property
    def _overall_rag(self) -> str:
        """RAG from validation results if available, else from card."""
        rag = self.validation_results.get("overall_rag", "")
        if not rag:
            rag = self.card.overall_rag
        return rag or "Not assessed"

    # ------------------------------------------------------------------
    # dict output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a structured dict.

        Returns a dict with the following top-level keys:
        ``report_date``, ``model_identity``, ``risk_tier``,
        ``validation_summary``, ``monitoring_summary``,
        ``assumptions_register``, ``outstanding_issues``,
        ``governance``, ``recommendations``.
        """
        card = self.card

        # Tier info
        tier_number = self._tier_number
        tier_dict: dict[str, Any] = {
            "tier": tier_number,
            "tier_label": self._tier_label,
            "score": self.tier.score if self.tier is not None else None,
            "rationale": (
                self.tier.rationale if self.tier is not None else card.tier_rationale
            ),
            "review_frequency": (
                self.tier.review_frequency if self.tier is not None else None
            ),
            "sign_off_requirement": (
                self.tier.sign_off_requirement if self.tier is not None else None
            ),
            "dimensions": (
                [d.to_dict() for d in self.tier.dimensions]
                if self.tier is not None
                else []
            ),
        }

        return {
            "report_date": self.report_date,
            "model_identity": {
                "model_id": card.model_id,
                "model_name": card.model_name,
                "version": card.version,
                "model_class": card.model_class,
                "model_type": card.model_type,
                "distribution_family": card.distribution_family,
                "target_variable": card.target_variable,
                "champion_challenger_status": card.champion_challenger_status,
                "developer": card.developer,
                "portfolio_scope": card.portfolio_scope,
                "geographic_scope": card.geographic_scope,
                "customer_facing": card.customer_facing,
                "regulatory_use": card.regulatory_use,
                "gwp_impacted": card.gwp_impacted,
                "training_data_period": list(card.training_data_period),
                "intended_use": card.intended_use,
                "not_intended_for": card.not_intended_for,
            },
            "risk_tier": tier_dict,
            "validation_summary": {
                "overall_rag": self._overall_rag,
                "run_id": self.validation_results.get(
                    "run_id", card.last_validation_run_id
                ),
                "run_date": self.validation_results.get(
                    "run_date", card.last_validation_run
                ),
                "key_metrics": {
                    "gini": self.validation_results.get("gini"),
                    "ae_ratio": self.validation_results.get("ae_ratio"),
                    "psi_score": self.validation_results.get("psi_score"),
                    "hl_p_value": self.validation_results.get("hl_p_value"),
                },
                "section_results": self.validation_results.get("section_results", []),
            },
            "monitoring_summary": {
                "period": self.monitoring_results.get("period", ""),
                "ae_ratio": self.monitoring_results.get("ae_ratio"),
                "psi_score": self.monitoring_results.get("psi_score"),
                "gini": self.monitoring_results.get("gini"),
                "recommendation": self.monitoring_results.get("recommendation", ""),
                "triggered_alerts": self.monitoring_results.get("triggered_alerts", []),
                "last_run": card.last_monitoring_run,
                "owner": card.monitoring_owner,
                "frequency": card.monitoring_frequency,
                "triggers": card.monitoring_triggers,
            },
            "assumptions_register": {
                "total": len(card.assumptions),
                "high_risk": len(card.high_risk_assumptions),
                "medium_risk": len(card.medium_risk_assumptions),
                "assumptions": [a.to_dict() for a in card.assumptions],
            },
            "outstanding_issues": card.outstanding_issues,
            "limitations": [lim.to_dict() for lim in card.limitations],
            "governance": {
                "approved_by": card.approved_by,
                "approval_date": card.approval_date,
                "approval_conditions": card.approval_conditions,
                "next_review_date": card.next_review_date,
                "review_frequency": (
                    self.tier.review_frequency if self.tier is not None else None
                ),
            },
            "recommendations": self._build_recommendations(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # HTML output
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        """Generate a self-contained HTML governance report.

        The HTML is styled inline and uses no external CDN dependencies.
        Print to PDF from a browser using Ctrl+P -> Save as PDF.

        Returns:
            Complete HTML document as a string.
        """
        return _render_html(self.to_dict())

    def save_html(self, path: str) -> None:
        """Write the HTML report to a file.

        Args:
            path: File path to write to (e.g. ``'motor_freq_mrm_pack.html'``).
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_html())

    def save_json(self, path: str) -> None:
        """Write the JSON report to a file.

        Args:
            path: File path to write to.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(self) -> list[str]:
        """Generate a list of actionable recommendations based on current state."""
        recs: list[str] = []
        card = self.card

        if not card.is_approved:
            recs.append(
                "Model has not been formally approved. "
                "Obtain sign-off from the appropriate authority before production use."
            )
        if not card.next_review_date:
            recs.append(
                "No next review date set. Set a review date consistent with the "
                "tier validation frequency."
            )

        rag = self._overall_rag
        if rag == "RED":
            recs.append(
                "Most recent validation result is RED. "
                "Address flagged issues before the next production cycle."
            )
        elif rag == "AMBER":
            recs.append(
                "Most recent validation result is AMBER. "
                "Review flagged items and document mitigations."
            )
        if not card.last_validation_run:
            recs.append(
                "No validation run recorded. "
                "Run an independent validation using insurance-validation."
            )

        high_count = len(card.high_risk_assumptions)
        if high_count > 0:
            recs.append(
                f"{high_count} assumption(s) rated HIGH risk. "
                "Ensure each has a documented mitigation and is on the monitoring plan."
            )

        if card.outstanding_issues:
            recs.append(
                f"{len(card.outstanding_issues)} outstanding issue(s) recorded. "
                "Confirm resolution status before next sign-off."
            )

        triggered = self.monitoring_results.get("triggered_alerts", [])
        if triggered:
            recs.append(
                f"{len(triggered)} monitoring alert(s) active. "
                "Review and document response actions."
            )

        if not recs:
            recs.append(
                "No material governance actions identified. "
                "Proceed to next scheduled review."
            )
        return recs


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _e(s: str) -> str:
    """Minimal HTML escaping for user-supplied text."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _rag_badge(rag: str) -> str:
    if rag in _RAG_COLOURS:
        border, bg, text = _RAG_COLOURS[rag]
    else:
        border, bg, text = ("#6b7280", "#f9fafb", "#374151")
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{bg};color:{text};border:1px solid {border};'
        f'font-weight:600;font-size:0.85em;">{_e(rag)}</span>'
    )


def _tier_badge(tier: Optional[int]) -> str:
    if tier is None:
        border, bg = _DEFAULT_COLOUR
        label = "Not assessed"
    else:
        border, bg = _TIER_COLOURS.get(tier, _DEFAULT_COLOUR)
        label = f"Tier {tier} — {TIER_LABELS.get(tier, '')}"
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{bg};color:#111;border:1px solid {border};'
        f'font-weight:600;font-size:0.85em;">{_e(label)}</span>'
    )


def _metric_box(label: str, value: Any, threshold: Optional[str] = None) -> str:
    if value is None:
        val_str = "—"
    elif isinstance(value, float):
        val_str = f"{value:.3f}"
    else:
        val_str = str(value)
    threshold_html = (
        f'<div style="font-size:0.75em;color:#6b7280;">threshold: {_e(threshold)}</div>'
        if threshold
        else ""
    )
    return (
        f'<div style="display:inline-block;text-align:center;padding:12px 16px;'
        f'margin:4px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
        f'min-width:100px;">'
        f'<div style="font-size:1.3em;font-weight:700;color:#1e293b;">{_e(val_str)}</div>'
        f'<div style="font-size:0.8em;color:#64748b;margin-top:2px;">{_e(label)}</div>'
        f'{threshold_html}'
        f'</div>'
    )


def _assumption_risk_colour(risk: str) -> str:
    return {
        "HIGH": "#dc2626",
        "MEDIUM": "#f59e0b",
        "LOW": "#22c55e",
    }.get(risk, "#6b7280")


def _render_dimension_table(dimensions: list[dict[str, Any]]) -> str:
    if not dimensions:
        return ""
    rows = "".join(
        f'<tr>'
        f'<td style="padding:4px 8px;font-weight:500;">'
        f'{_e(d["name"].replace("_", " ").title())}</td>'
        f'<td style="padding:4px 8px;">{d["score"]:.0f}/{d["max_score"]:.0f}</td>'
        f'<td style="padding:4px 8px;">{d["pct"]}%</td>'
        f'<td style="padding:4px 8px;color:#64748b;font-size:0.9em;">{_e(d["rationale"])}</td>'
        f'</tr>'
        for d in dimensions
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px;">'
        '<thead><tr style="background:#f1f5f9;">'
        '<th style="padding:4px 8px;text-align:left;">Dimension</th>'
        '<th style="padding:4px 8px;text-align:left;">Score</th>'
        '<th style="padding:4px 8px;text-align:left;">%</th>'
        '<th style="padding:4px 8px;text-align:left;">Rationale</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _render_triggers(triggers: dict[str, Any]) -> str:
    if not triggers:
        return ""
    rows = "".join(
        f'<tr><td style="padding:4px 8px;">{_e(k)}</td>'
        f'<td style="padding:4px 8px;">{_e(str(v))}</td></tr>'
        for k, v in triggers.items()
    )
    return (
        '<h3 style="font-size:0.9em;margin:12px 0 4px;">Review triggers</h3>'
        '<table style="width:50%;border-collapse:collapse;font-size:0.85em;">'
        '<thead><tr style="background:#f1f5f9;">'
        '<th style="padding:4px 8px;text-align:left;">Metric</th>'
        '<th style="padding:4px 8px;text-align:left;">Threshold</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _render_html(d: dict[str, Any]) -> str:
    """Render the report dict to a self-contained HTML document."""
    identity = d["model_identity"]
    tier_info = d["risk_tier"]
    val = d["validation_summary"]
    mon = d["monitoring_summary"]
    assumptions_info = d["assumptions_register"]
    gov = d["governance"]
    recs = d["recommendations"]
    outstanding = d.get("outstanding_issues", [])

    rag = val.get("overall_rag", "Not assessed")
    tier_num = tier_info.get("tier")
    metrics = val.get("key_metrics", {})

    gwp = identity.get("gwp_impacted", 0.0) or 0.0
    gwp_str = f"£{gwp/1_000_000:.1f}m" if gwp >= 1_000_000 else f"£{gwp:,.0f}"

    # Section results table
    section_html = ""
    if val.get("section_results"):
        rows_html = "".join(
            f'<tr><td style="padding:4px 8px;">{_e(s.get("section", ""))}</td>'
            f'<td style="padding:4px 8px;">{_rag_badge(s.get("status", ""))}</td>'
            f'<td style="padding:4px 8px;color:#64748b;">{_e(s.get("notes", ""))}</td></tr>'
            for s in val["section_results"]
        )
        section_html = (
            '<h3 style="font-size:1em;margin:16px 0 8px;">Validation sections</h3>'
            '<table style="width:100%;border-collapse:collapse;font-size:0.9em;">'
            '<thead><tr style="background:#f1f5f9;">'
            '<th style="padding:4px 8px;text-align:left;">Section</th>'
            '<th style="padding:4px 8px;text-align:left;">Status</th>'
            '<th style="padding:4px 8px;text-align:left;">Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )

    # Assumptions — MEDIUM and HIGH only in exec summary
    material_assumptions = [
        a for a in assumptions_info.get("assumptions", [])
        if a.get("risk") in ("MEDIUM", "HIGH")
    ]
    if material_assumptions:
        rows_html = "".join(
            f'<tr><td style="padding:4px 8px;">'
            f'<span style="color:{_assumption_risk_colour(a["risk"])};font-weight:600;">'
            f'{_e(a["risk"])}</span></td>'
            f'<td style="padding:4px 8px;">{_e(a["description"])}</td>'
            f'<td style="padding:4px 8px;color:#64748b;">{_e(a.get("mitigation", ""))}</td>'
            f'</tr>'
            for a in material_assumptions
        )
        assumptions_html = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.9em;">'
            '<thead><tr style="background:#f1f5f9;">'
            '<th style="padding:4px 8px;text-align:left;">Risk</th>'
            '<th style="padding:4px 8px;text-align:left;">Assumption</th>'
            '<th style="padding:4px 8px;text-align:left;">Mitigation</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )
    else:
        assumptions_html = (
            '<p style="color:#64748b;font-size:0.9em;">'
            'No material (MEDIUM/HIGH) assumptions recorded.</p>'
        )

    # Outstanding issues
    if outstanding:
        items = "".join(
            f'<li style="margin-bottom:4px;">{_e(iss)}</li>' for iss in outstanding
        )
        issues_html = f'<ul style="margin:0;padding-left:20px;">{items}</ul>'
    else:
        issues_html = (
            '<p style="color:#64748b;font-size:0.9em;">No outstanding issues.</p>'
        )

    # Recommendations
    recs_html = "".join(
        f'<li style="margin-bottom:6px;">{_e(rec)}</li>' for rec in recs
    )

    # Monitoring alerts
    triggered = mon.get("triggered_alerts", [])
    alerts_html = ""
    if triggered:
        items = "".join(
            f'<li style="color:#dc2626;margin-bottom:4px;">{_e(a)}</li>'
            for a in triggered
        )
        alerts_html = f'<ul style="margin:0;padding-left:20px;">{items}</ul>'

    not_intended = identity.get("not_intended_for", [])
    approved_by_str = ", ".join(gov.get("approved_by", [])) or "Not approved"
    approval_conditions = gov.get("approval_conditions", "") or "—"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MRM Governance Report — {_e(identity['model_name'])} v{_e(identity['version'])}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
            color: #1e293b; background: #fff; line-height: 1.5; }}
    .page {{ max-width: 900px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.4em; font-weight: 700; color: #0f172a; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1em; font-weight: 600; color: #1e293b;
          border-bottom: 2px solid #3b82f6; padding-bottom: 4px;
          margin: 24px 0 12px; }}
    h3 {{ font-size: 0.95em; font-weight: 600; color: #334155; }}
    .meta {{ color: #64748b; font-size: 0.88em; margin-bottom: 16px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0;
             border-radius: 8px; padding: 12px 16px; }}
    .kv {{ display: flex; flex-direction: column; margin-bottom: 6px; }}
    .kv-label {{ font-size: 0.8em; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.04em; }}
    .kv-value {{ font-weight: 500; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    th {{ background: #f1f5f9; padding: 6px 8px; text-align: left;
          font-weight: 600; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #f1f5f9; }}
    .rec-list {{ padding-left: 20px; }}
    .rec-list li {{ margin-bottom: 6px; }}
    .footer {{ margin-top: 32px; padding-top: 12px;
               border-top: 1px solid #e2e8f0;
               font-size: 0.8em; color: #94a3b8; }}
    @media print {{
      body {{ font-size: 11px; }}
      .page {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
<div class="page">

  <h1>Model Risk Management — Governance Report</h1>
  <div class="meta">
    {_e(identity['model_name'])} &nbsp;|&nbsp; v{_e(identity['version'])}
    &nbsp;|&nbsp; {_e(identity['model_class'].title())}
    &nbsp;|&nbsp; Report date: {_e(d['report_date'])}
  </div>

  <div style="display:flex;gap:16px;align-items:center;
              background:#f8fafc;border:1px solid #e2e8f0;
              border-radius:8px;padding:12px 16px;margin-bottom:20px;
              flex-wrap:wrap;">
    <div>
      <div class="kv-label">Risk tier</div>
      <div>{_tier_badge(tier_num)}</div>
    </div>
    <div>
      <div class="kv-label">Validation status</div>
      <div>{_rag_badge(rag)}</div>
    </div>
    <div>
      <div class="kv-label">Deployment status</div>
      <div><strong>{_e(identity.get('champion_challenger_status', '').title())}</strong></div>
    </div>
    <div>
      <div class="kv-label">GWP impacted</div>
      <div><strong>{_e(gwp_str)}</strong></div>
    </div>
    <div>
      <div class="kv-label">Next review</div>
      <div><strong>{_e(gov.get('next_review_date') or '—')}</strong></div>
    </div>
  </div>

  <h2>1. Model identity</h2>
  <div class="grid-2">
    <div class="card">
      <div class="kv"><span class="kv-label">Model ID</span>
        <span class="kv-value">{_e(identity['model_id'])}</span></div>
      <div class="kv"><span class="kv-label">Target variable</span>
        <span class="kv-value">{_e(identity.get('target_variable') or '—')}</span></div>
      <div class="kv"><span class="kv-label">Distribution</span>
        <span class="kv-value">{_e(identity.get('distribution_family') or '—')}</span></div>
      <div class="kv"><span class="kv-label">Model type</span>
        <span class="kv-value">{_e(identity.get('model_type') or '—')}</span></div>
      <div class="kv"><span class="kv-label">Developer</span>
        <span class="kv-value">{_e(identity.get('developer') or '—')}</span></div>
      <div class="kv"><span class="kv-label">Training period</span>
        <span class="kv-value">
          {_e(' – '.join(str(x) for x in identity.get('training_data_period', ['', ''])))}
        </span></div>
    </div>
    <div class="card">
      <div class="kv"><span class="kv-label">Intended use</span>
        <span class="kv-value" style="font-size:0.92em;">
          {_e(identity.get('intended_use') or '—')}</span></div>
      <div class="kv"><span class="kv-label">Not intended for</span>
        <span class="kv-value" style="font-size:0.92em;">
          {_e(', '.join(not_intended) if not_intended else '—')}</span></div>
      <div class="kv"><span class="kv-label">Customer-facing</span>
        <span class="kv-value">{'Yes' if identity.get('customer_facing') else 'No'}</span></div>
      <div class="kv"><span class="kv-label">Regulatory use</span>
        <span class="kv-value">{'Yes' if identity.get('regulatory_use') else 'No'}</span></div>
    </div>
  </div>

  <h2>2. Risk tier</h2>
  <div class="card">
    <div style="display:flex;gap:32px;margin-bottom:8px;flex-wrap:wrap;">
      <div>
        <div class="kv-label">Tier</div>
        <div style="font-size:1.5em;font-weight:700;">
          {tier_num if tier_num is not None else '—'}
          <span style="font-size:0.6em;font-weight:400;color:#64748b;">
            {_e(tier_info.get('tier_label', ''))}
          </span>
        </div>
      </div>
      <div>
        <div class="kv-label">Score</div>
        <div style="font-size:1.5em;font-weight:700;">
          {f"{tier_info['score']:.1f}" if tier_info.get('score') is not None else '—'}
          <span style="font-size:0.6em;font-weight:400;color:#64748b;">/100</span>
        </div>
      </div>
      <div>
        <div class="kv-label">Review frequency</div>
        <div style="font-weight:600;">{_e(tier_info.get('review_frequency') or '—')}</div>
      </div>
      <div>
        <div class="kv-label">Sign-off required</div>
        <div style="font-weight:600;">{_e(tier_info.get('sign_off_requirement') or '—')}</div>
      </div>
    </div>
    {_render_dimension_table(tier_info.get('dimensions', []))}
  </div>

  <h2>3. Validation summary</h2>
  <div class="card">
    <div style="margin-bottom:8px;">
      Overall: {_rag_badge(rag)}
      &nbsp; Run date: {_e(val.get('run_date') or '—')}
      &nbsp; ID: <code style="font-size:0.85em;">{_e(val.get('run_id') or '—')}</code>
    </div>
    <div>
      {_metric_box('Gini', metrics.get('gini'), '>0.35')}
      {_metric_box('A/E ratio', metrics.get('ae_ratio'), '0.95–1.05')}
      {_metric_box('PSI score', metrics.get('psi_score'), '<0.10')}
      {_metric_box('H-L p-value', metrics.get('hl_p_value'), '>0.05')}
    </div>
    {section_html}
  </div>

  <h2>4. Monitoring</h2>
  <div class="card">
    <div class="grid-2" style="gap:8px;margin-bottom:8px;">
      <div>
        <div class="kv"><span class="kv-label">Owner</span>
          <span class="kv-value">{_e(mon.get('owner') or '—')}</span></div>
        <div class="kv"><span class="kv-label">Frequency</span>
          <span class="kv-value">{_e(mon.get('frequency') or '—')}</span></div>
        <div class="kv"><span class="kv-label">Last run</span>
          <span class="kv-value">{_e(mon.get('last_run') or '—')}</span></div>
        <div class="kv"><span class="kv-label">Recommendation</span>
          <span class="kv-value">{_e(mon.get('recommendation') or '—')}</span></div>
      </div>
      <div>
        {_metric_box('A/E ratio', mon.get('ae_ratio'))}
        {_metric_box('PSI score', mon.get('psi_score'))}
        {_metric_box('Gini', mon.get('gini'))}
      </div>
    </div>
    {alerts_html}
    {_render_triggers(mon.get('triggers', {}))}
  </div>

  <h2>5. Material assumptions (MEDIUM / HIGH risk)</h2>
  <div class="card">
    <div class="meta" style="margin-bottom:8px;">
      Total: {assumptions_info.get('total', 0)} &nbsp;|&nbsp;
      HIGH: {assumptions_info.get('high_risk', 0)} &nbsp;|&nbsp;
      MEDIUM: {assumptions_info.get('medium_risk', 0)}
    </div>
    {assumptions_html}
  </div>

  <h2>6. Outstanding issues</h2>
  <div class="card">{issues_html}</div>

  <h2>7. Governance</h2>
  <div class="card">
    <div class="grid-2">
      <div>
        <div class="kv"><span class="kv-label">Approved by</span>
          <span class="kv-value">{_e(approved_by_str)}</span></div>
        <div class="kv"><span class="kv-label">Approval date</span>
          <span class="kv-value">{_e(gov.get('approval_date') or '—')}</span></div>
        <div class="kv"><span class="kv-label">Approval conditions</span>
          <span class="kv-value" style="font-size:0.92em;">
            {_e(approval_conditions)}</span></div>
      </div>
      <div>
        <div class="kv"><span class="kv-label">Next review date</span>
          <span class="kv-value">{_e(gov.get('next_review_date') or '—')}</span></div>
        <div class="kv"><span class="kv-label">Review frequency</span>
          <span class="kv-value">{_e(gov.get('review_frequency') or '—')}</span></div>
      </div>
    </div>
  </div>

  <h2>8. Recommendations</h2>
  <div class="card">
    <ul class="rec-list">{recs_html}</ul>
  </div>

  <div class="footer">
    Generated by insurance-mrm &middot; {_e(d['report_date'])} &middot;
    {_e(identity['model_name'])} v{_e(identity['version'])} &middot;
    Tier {tier_num if tier_num is not None else 'N/A'}
  </div>

</div>
</body>
</html>"""
