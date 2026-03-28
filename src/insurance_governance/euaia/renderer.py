"""Markdown and HTML rendering for EU AI Act compliance documents.

Produces human-readable output for:
- ``Article13Document`` — full Article 13(3) transparency document
- ``ConformityAssessment`` — Annex VI self-assessment pack

Both renderers produce structured Markdown that maps directly to the
regulation's sub-paragraph numbering so compliance reviewers can check
each item against the source text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .article13 import Article13Document
    from .conformity import ConformityAssessment


def _bullet_list(items: list[str], indent: int = 0) -> str:
    """Render a list of strings as Markdown bullet points."""
    if not items:
        return "_None specified._"
    prefix = "  " * indent
    return "\n".join(f"{prefix}- {item}" for item in items)


def _metric_table(metrics: dict[str, float]) -> str:
    """Render a dict of metric -> value as a Markdown table."""
    if not metrics:
        return "_No metrics recorded._"
    lines = ["| Metric | Value |", "| --- | --- |"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _subgroup_table(subgroups: dict[str, dict[str, float]]) -> str:
    """Render nested subgroup performance as a Markdown table."""
    if not subgroups:
        return "_No subgroup performance data recorded._"

    # Collect all metric names across all groups
    all_metrics: list[str] = []
    for metrics in subgroups.values():
        for m in metrics:
            if m not in all_metrics:
                all_metrics.append(m)

    header = "| Group | " + " | ".join(all_metrics) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(all_metrics)) + " |"
    rows = [header, divider]
    for group, metrics in subgroups.items():
        vals = " | ".join(str(metrics.get(m, "—")) for m in all_metrics)
        rows.append(f"| {group} | {vals} |")
    return "\n".join(rows)


def _feature_table(features: list[dict]) -> str:
    """Render input feature specs as a Markdown table."""
    if not features:
        return "_No input features documented._"
    header = "| Name | Type | Range | Missing handling |"
    divider = "| --- | --- | --- | --- |"
    rows = [header, divider]
    for f in features:
        name = f.get("name", "")
        ftype = f.get("type", "")
        frange = f.get("range", "")
        missing = f.get("missing_handling", "")
        rows.append(f"| {name} | {ftype} | {frange} | {missing} |")
    return "\n".join(rows)


def render_article13_markdown(doc: "Article13Document") -> str:
    """Render an ``Article13Document`` as a structured Markdown document.

    The output follows the Article 13(3)(a)–(e) structure so each section
    can be checked against the regulation directly.

    Parameters
    ----------
    doc:
        Populated ``Article13Document`` instance.

    Returns
    -------
    str
        Markdown-formatted transparency document.
    """
    gaps = doc.flag_gaps()
    gap_section = ""
    if gaps:
        gap_section = (
            "\n\n> **Compliance gaps detected** — the following mandatory fields "
            "are missing or empty. This document is not ready for publication.\n\n"
            + _bullet_list(gaps)
        )

    lines: list[str] = [
        f"# Article 13 Transparency Document",
        f"",
        f"**Model:** {doc.model_name} {doc.model_version}  ",
        f"**Provider:** {doc.provider_name}  ",
        f"**Document date:** {doc.document_date}",
        gap_section,
        "",
        "---",
        "",
        "## Article 13(3)(a) — Provider Identity",
        "",
        f"**Provider name:** {doc.provider_name}  ",
        f"**Contact:** {doc.provider_contact}  ",
        f"**Model name:** {doc.model_name}  ",
        f"**Model version:** {doc.model_version}  ",
        f"**Document date:** {doc.document_date}",
        "",
        "---",
        "",
        "## Article 13(3)(b) — Performance Characteristics",
        "",
        "### (b)(i) Intended Purpose",
        "",
        doc.intended_purpose or "_Not specified._",
        "",
        "**Out-of-scope uses:**",
        "",
        _bullet_list(doc.out_of_scope_uses),
        "",
        "### (b)(ii) Accuracy",
        "",
        _metric_table(doc.accuracy_metrics),
        "",
        "**Known accuracy limitations:**",
        "",
        _bullet_list(doc.known_accuracy_limitations),
        "",
        "### (b)(iii) Known Risks",
        "",
        _bullet_list(doc.known_risks),
        "",
        "### (b)(iv) Explanation Tools",
        "",
        _bullet_list(doc.explanation_tools),
        "",
        "### (b)(v) Subgroup Performance",
        "",
        _subgroup_table(doc.subgroup_performance),
        "",
        "### (b)(vi) Input Feature Specifications",
        "",
        _feature_table(doc.input_features),
        "",
        "### (b)(vii) Output Interpretation Guide",
        "",
        doc.output_interpretation_guide or "_Not specified._",
        "",
        "---",
        "",
        "## Article 13(3)(c) — Planned Changes",
        "",
        _bullet_list(doc.planned_changes) if doc.planned_changes else "_No planned changes at time of publication._",
        "",
        "---",
        "",
        "## Article 13(3)(d) — Human Oversight",
        "",
        "**Oversight measures:**",
        "",
        _bullet_list(doc.human_oversight_measures),
        "",
        "**Override procedure:**",
        "",
        doc.override_procedure or "_Not specified._",
        "",
        "**Anomaly thresholds triggering human review:**",
        "",
        _metric_table(doc.anomaly_thresholds),
        "",
        "---",
        "",
        "## Article 13(3)(e) — Expected Lifetime and Maintenance",
        "",
        f"**Expected operational lifetime:** {doc.expected_lifetime_months} months  ",
        f"**Next scheduled retraining:** {doc.next_retraining_date or 'Not yet scheduled'}",
        "",
        "**Retraining triggers:**",
        "",
        _bullet_list(doc.retraining_triggers),
        "",
        "**Monitoring metrics:**",
        "",
        _bullet_list(doc.monitoring_metrics),
        "",
        "---",
        "",
        "_This document was generated by insurance-governance. "
        "It provides a structured starting point for Article 13 compliance and "
        "does not constitute legal advice._",
    ]

    return "\n".join(lines)


def render_conformity_markdown(assessment: "ConformityAssessment") -> str:
    """Render an Annex VI ``ConformityAssessment`` as a structured Markdown document.

    Parameters
    ----------
    assessment:
        Populated ``ConformityAssessment`` instance.

    Returns
    -------
    str
        Markdown-formatted conformity assessment pack.
    """
    overall = assessment.overall_status().value.upper()
    incomplete = assessment.flag_incomplete()

    summary_section = ""
    if incomplete:
        summary_section = (
            "\n\n> **Assessment incomplete** — the following steps require action:\n\n"
            + _bullet_list(incomplete)
        )

    lines: list[str] = [
        "# Annex VI Conformity Assessment Pack",
        "",
        f"**Model:** {assessment.model_name}  ",
        f"**Assessor:** {assessment.assessor_name}  ",
        f"**Assessment date:** {assessment.assessment_date}  ",
        f"**Overall status:** {overall}",
        summary_section,
        "",
        "---",
        "",
        "## Assessment Steps",
        "",
    ]

    _STATUS_EMOJI = {
        "complete": "COMPLETE",
        "incomplete": "INCOMPLETE",
        "not_applicable": "N/A",
    }

    for step in assessment.steps:
        status_label = _STATUS_EMOJI.get(step.status.value, step.status.value)
        lines += [
            f"### Step {step.step_number}: {step.title}",
            "",
            f"**Regulatory reference:** {step.regulatory_reference}  ",
            f"**Status:** {status_label}",
            "",
        ]
        if step.evidence:
            lines += ["**Evidence:**", "", step.evidence, ""]
        else:
            lines += ["**Evidence:** _None recorded._", ""]

        if step.findings:
            lines += ["**Findings:**", "", _bullet_list(step.findings), ""]

        lines.append("---")
        lines.append("")

    lines += [
        "_This assessment pack was generated by insurance-governance following "
        "Annex VI of Regulation (EU) 2024/1689. It does not constitute a formal "
        "declaration of conformity and does not replace qualified legal review._",
    ]

    return "\n".join(lines)


def article13_to_html(doc: "Article13Document") -> str:
    """Convert an Article13Document to a minimal HTML document.

    Uses a simple Markdown-to-HTML approach: renders the Markdown first,
    then wraps it in a basic HTML shell. For rich HTML with CSS, pass the
    Markdown output to a dedicated converter such as ``markdown`` or
    ``mistune``.

    Parameters
    ----------
    doc:
        Populated ``Article13Document`` instance.

    Returns
    -------
    str
        HTML string. Requires no external dependencies.
    """
    md = render_article13_markdown(doc)

    # Basic Markdown -> HTML: headings, bullets, bold, horizontal rules
    lines = []
    in_list = False
    for line in md.split("\n"):
        if line.startswith("### "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{line[2:]}</li>")
        elif line == "---":
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append("<hr>")
        elif line.strip() == "":
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append("")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            # Bold
            import re
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            # Italic
            line = re.sub(r"_(.+?)_", r"<em>\1</em>", line)
            lines.append(f"<p>{line}</p>")

    if in_list:
        lines.append("</ul>")

    body = "\n".join(lines)
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='UTF-8'>\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
        f"  <title>Article 13 — {doc.model_name} {doc.model_version}</title>\n"
        "  <style>\n"
        "    body { font-family: sans-serif; max-width: 900px; margin: 2em auto; "
        "padding: 0 1em; line-height: 1.6; }\n"
        "    h1, h2, h3 { color: #1a1a2e; }\n"
        "    hr { border: 1px solid #ccc; }\n"
        "    table { border-collapse: collapse; width: 100%; }\n"
        "    td, th { border: 1px solid #ccc; padding: 0.4em 0.8em; }\n"
        "    blockquote { border-left: 4px solid #e55; padding-left: 1em; color: #c00; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>"
    )
