"""Article 13 transparency document for high-risk AI systems.

Regulation (EU) 2024/1689 Article 13 requires providers of high-risk AI systems
to ensure that the systems are sufficiently transparent that deployers can
understand the system's output and use it appropriately.

Article 13(3) specifies the minimum content of the instructions for use. This
module maps those sub-paragraphs directly to a structured dataclass, with methods
to compute the accuracy metrics required by Article 13(3)(b)(ii) and to render the
full document in Markdown or as a JSON-serialisable dict.

Subparagraph mapping:
    (a)  Provider identity and contact
    (b)  Performance characteristics — seven sub-items (i)–(vii)
    (c)  Planned changes subject to advance notification
    (d)  Human oversight measures and override procedures
    (e)  Expected lifetime and maintenance/monitoring plan
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Gini coefficient (normalised area between Lorenz curves).

    Equivalent to 2 * AUC - 1 for binary outcomes. For frequency/severity
    models with continuous y_true, uses the standard actuarial formulation
    (ordered by predicted, cumulated actual).

    Returns a value in [-1, 1]; values close to 1 indicate a model that
    perfectly ranks risks.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    n = len(y_true)
    if n == 0:
        return float("nan")

    order = np.argsort(y_pred)
    y_sorted = y_true[order]

    cum_true = np.cumsum(y_sorted)
    total_true = cum_true[-1]

    if total_true == 0:
        return float("nan")

    lorenz = cum_true / total_true
    # Area under the Lorenz curve via trapezoidal rule
    steps = np.arange(1, n + 1) / n
    auc_lorenz = float(np.trapz(lorenz, steps))
    return 2.0 * auc_lorenz - 1.0


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the Gini coefficient."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g = _gini_coefficient(y_true[idx], y_pred[idx])
        if not math.isnan(g):
            estimates.append(g)

    if not estimates:
        return (float("nan"), float("nan"))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(estimates, alpha))
    hi = float(np.quantile(estimates, 1.0 - alpha))
    return lo, hi


@dataclass
class Article13Document:
    """Transparency document per Article 13(3) of Regulation (EU) 2024/1689.

    Instantiate with the fields you know at model build time, then call
    ``compute_accuracy()`` and ``compute_subgroup_performance()`` to populate
    the metric fields programmatically. Use ``flag_gaps()`` to identify
    missing mandatory content before publication.

    The ``to_markdown()`` method produces a self-contained document suitable
    for attaching to a model deployment pack or inserting into a model card.

    Parameters
    ----------
    provider_name:
        Legal name of the provider as defined in Article 3(3).
    provider_contact:
        Email or postal address for transparency enquiries.
    model_name:
        Short identifier for the model (e.g. "Motor Frequency XGB v3").
    model_version:
        Semantic version string or date-based identifier.
    document_date:
        ISO 8601 date string (e.g. "2025-11-01").
    intended_purpose:
        Plain-language description of the model's intended use per Art 13(3)(b)(i).
    out_of_scope_uses:
        Explicit list of uses the model must not be applied to.
    accuracy_metrics:
        Dict mapping metric name to float value (populated by ``compute_accuracy()``
        or supplied directly). E.g. ``{"gini": 0.42, "ae_ratio": 1.01}``.
    known_accuracy_limitations:
        Free-text items describing known conditions under which accuracy degrades.
    known_risks:
        Free-text risk items per Art 13(3)(b)(iii).
    explanation_tools:
        Tools used to interpret model output, e.g. ``["SHAP TreeExplainer"]``.
    subgroup_performance:
        Nested dict: group label -> metric name -> value. Populated by
        ``compute_subgroup_performance()`` or supplied directly.
    input_features:
        List of dicts with keys ``name``, ``type``, ``range``,
        ``missing_handling``.
    output_interpretation_guide:
        Plain-language guide to reading the model's output per Art 13(3)(b)(vii).
    planned_changes:
        Substantive changes planned for future versions per Art 13(3)(c).
    human_oversight_measures:
        Concrete measures implementing Article 14 requirements.
    override_procedure:
        Documented procedure for a human to override the model's output.
    anomaly_thresholds:
        Dict of metric name -> threshold value triggering human review.
    expected_lifetime_months:
        Expected operational lifetime in months per Art 13(3)(e).
    next_retraining_date:
        ISO 8601 date or None if not yet scheduled.
    retraining_triggers:
        Conditions that would trigger unscheduled retraining.
    monitoring_metrics:
        Metrics tracked in production to detect model drift.
    """

    # Art 13(3)(a) — Provider identity
    provider_name: str = ""
    provider_contact: str = ""
    model_name: str = ""
    model_version: str = ""
    document_date: str = ""

    # Art 13(3)(b)(i) — Purpose
    intended_purpose: str = ""
    out_of_scope_uses: list[str] = field(default_factory=list)

    # Art 13(3)(b)(ii) — Accuracy
    accuracy_metrics: dict[str, float] = field(default_factory=dict)
    known_accuracy_limitations: list[str] = field(default_factory=list)

    # Art 13(3)(b)(iii) — Risks
    known_risks: list[str] = field(default_factory=list)

    # Art 13(3)(b)(iv) — Explainability
    explanation_tools: list[str] = field(default_factory=list)

    # Art 13(3)(b)(v) — Subgroup performance
    subgroup_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    # Art 13(3)(b)(vi) — Input specs
    input_features: list[dict[str, Any]] = field(default_factory=list)

    # Art 13(3)(b)(vii) — Output interpretation
    output_interpretation_guide: str = ""

    # Art 13(3)(c) — Changes
    planned_changes: list[str] = field(default_factory=list)

    # Art 13(3)(d) — Human oversight
    human_oversight_measures: list[str] = field(default_factory=list)
    override_procedure: str = ""
    anomaly_thresholds: dict[str, float] = field(default_factory=dict)

    # Art 13(3)(e) — Maintenance
    expected_lifetime_months: int = 0
    next_retraining_date: str | None = None
    retraining_triggers: list[str] = field(default_factory=list)
    monitoring_metrics: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Accuracy computation                                                 #
    # ------------------------------------------------------------------ #

    def compute_accuracy(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        exposure: ArrayLike | None = None,
        n_boot: int = 500,
        ci: float = 0.95,
    ) -> dict[str, float]:
        """Compute Gini coefficient with bootstrap CI and A/E ratio.

        Results are stored in ``self.accuracy_metrics`` and also returned.

        Parameters
        ----------
        y_true:
            Observed outcomes (claims counts, amounts, or binary indicators).
        y_pred:
            Model predictions on the same scale as y_true.
        exposure:
            Optional exposure weights (e.g. earned premium, policy years).
            Used to weight the A/E ratio. If None, uniform weights are used.
        n_boot:
            Bootstrap resamples for the Gini CI (default 500).
        ci:
            Confidence level for the bootstrap CI (default 0.95).

        Returns
        -------
        dict[str, float]
            Keys: ``gini``, ``gini_ci_lo``, ``gini_ci_hi``, ``ae_ratio``,
            ``n_obs``.
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        if yt.shape != yp.shape:
            raise ValueError(
                f"y_true shape {yt.shape} does not match y_pred shape {yp.shape}."
            )

        gini = _gini_coefficient(yt, yp)
        ci_lo, ci_hi = _bootstrap_ci(yt, yp, n_boot=n_boot, ci=ci, seed=42)

        if exposure is not None:
            exp = np.asarray(exposure, dtype=float)
            total_exp = exp.sum()
            ae = (yt.sum() / yp.dot(exp)) if total_exp > 0 else float("nan")
        else:
            ae = (yt.sum() / yp.sum()) if yp.sum() > 0 else float("nan")

        metrics: dict[str, float] = {
            "gini": round(gini, 4),
            f"gini_ci_lo_{int(ci * 100)}": round(ci_lo, 4),
            f"gini_ci_hi_{int(ci * 100)}": round(ci_hi, 4),
            "ae_ratio": round(ae, 4),
            "n_obs": float(len(yt)),
        }
        self.accuracy_metrics.update(metrics)
        return metrics

    def compute_subgroup_performance(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        groups: dict[str, ArrayLike],
        exposure: ArrayLike | None = None,
        min_group_size: int = 100,
    ) -> dict[str, dict[str, float]]:
        """Compute per-subgroup Gini and A/E ratio.

        Groups with fewer than ``min_group_size`` observations are excluded and
        logged in ``known_accuracy_limitations`` to satisfy Art 13(3)(b)(v)'s
        requirement that limitations in subgroup performance are disclosed.

        Parameters
        ----------
        y_true:
            Observed outcomes array.
        y_pred:
            Model predictions array.
        groups:
            Dict mapping group label to a boolean or integer mask array of the
            same length as y_true.  E.g. ``{"age_18_25": mask_18_25}``.
        exposure:
            Optional exposure weights.
        min_group_size:
            Minimum observations to report a group (default 100).

        Returns
        -------
        dict[str, dict[str, float]]
            Group label -> ``{"gini": ..., "ae_ratio": ..., "n_obs": ...}``.
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        exp = np.asarray(exposure, dtype=float) if exposure is not None else None

        results: dict[str, dict[str, float]] = {}
        excluded: list[str] = []

        for label, mask_raw in groups.items():
            mask = np.asarray(mask_raw, dtype=bool)
            n = int(mask.sum())
            if n < min_group_size:
                excluded.append(
                    f"Subgroup '{label}' excluded from performance table: "
                    f"only {n} observations (minimum {min_group_size})."
                )
                continue

            yt_g = yt[mask]
            yp_g = yp[mask]
            exp_g = exp[mask] if exp is not None else None

            gini_g = _gini_coefficient(yt_g, yp_g)

            if exp_g is not None:
                total_exp = exp_g.sum()
                ae_g = (yt_g.sum() / yp_g.dot(exp_g)) if total_exp > 0 else float("nan")
            else:
                ae_g = (yt_g.sum() / yp_g.sum()) if yp_g.sum() > 0 else float("nan")

            results[label] = {
                "gini": round(gini_g, 4),
                "ae_ratio": round(ae_g, 4),
                "n_obs": float(n),
            }

        if excluded:
            self.known_accuracy_limitations.extend(excluded)

        self.subgroup_performance.update(results)
        return results

    # ------------------------------------------------------------------ #
    # Gap detection                                                        #
    # ------------------------------------------------------------------ #

    _REQUIRED_FIELDS: list[str] = [
        "provider_name",
        "provider_contact",
        "model_name",
        "model_version",
        "document_date",
        "intended_purpose",
        "output_interpretation_guide",
        "override_procedure",
    ]

    _REQUIRED_NONEMPTY: list[str] = [
        "out_of_scope_uses",
        "known_risks",
        "explanation_tools",
        "input_features",
        "human_oversight_measures",
        "monitoring_metrics",
        "retraining_triggers",
    ]

    def flag_gaps(self) -> list[str]:
        """Return a list of unfilled or insufficient fields.

        The list corresponds roughly to the mandatory content items in
        Article 13(3).  An empty list means no obvious gaps were detected —
        it does not constitute legal confirmation of compliance.

        Returns
        -------
        list[str]
            Each entry is a human-readable description of the gap.
        """
        gaps: list[str] = []

        for f in self._REQUIRED_FIELDS:
            val = getattr(self, f)
            if not val:
                gaps.append(f"Art 13(3): required field '{f}' is empty.")

        for f in self._REQUIRED_NONEMPTY:
            val = getattr(self, f)
            if not val:
                gaps.append(
                    f"Art 13(3): required list '{f}' is empty — "
                    "at least one item is needed."
                )

        if not self.accuracy_metrics:
            gaps.append(
                "Art 13(3)(b)(ii): no accuracy metrics — "
                "call compute_accuracy() or populate accuracy_metrics directly."
            )

        if self.expected_lifetime_months <= 0:
            gaps.append(
                "Art 13(3)(e): expected_lifetime_months must be a positive integer."
            )

        if not self.anomaly_thresholds:
            gaps.append(
                "Art 13(3)(d): anomaly_thresholds is empty — "
                "define at least one metric threshold for human review escalation."
            )

        return gaps

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the document.

        The structure mirrors the Article 13(3) sub-paragraph hierarchy so
        it can be consumed by downstream reporting tools.
        """
        return {
            "article13_transparency_document": {
                "a_provider": {
                    "provider_name": self.provider_name,
                    "provider_contact": self.provider_contact,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "document_date": self.document_date,
                },
                "b_performance": {
                    "i_purpose": {
                        "intended_purpose": self.intended_purpose,
                        "out_of_scope_uses": self.out_of_scope_uses,
                    },
                    "ii_accuracy": {
                        "accuracy_metrics": self.accuracy_metrics,
                        "known_accuracy_limitations": self.known_accuracy_limitations,
                    },
                    "iii_risks": {
                        "known_risks": self.known_risks,
                    },
                    "iv_explainability": {
                        "explanation_tools": self.explanation_tools,
                    },
                    "v_subgroup_performance": self.subgroup_performance,
                    "vi_input_specs": {
                        "input_features": self.input_features,
                    },
                    "vii_output_interpretation": {
                        "output_interpretation_guide": self.output_interpretation_guide,
                    },
                },
                "c_planned_changes": self.planned_changes,
                "d_human_oversight": {
                    "human_oversight_measures": self.human_oversight_measures,
                    "override_procedure": self.override_procedure,
                    "anomaly_thresholds": self.anomaly_thresholds,
                },
                "e_maintenance": {
                    "expected_lifetime_months": self.expected_lifetime_months,
                    "next_retraining_date": self.next_retraining_date,
                    "retraining_triggers": self.retraining_triggers,
                    "monitoring_metrics": self.monitoring_metrics,
                },
            }
        }

    def to_markdown(self) -> str:
        """Render the full Article 13 transparency document as Markdown.

        The output is structured to follow the Article 13(3)(a)–(e) order
        so compliance reviewers can locate each sub-paragraph directly.
        """
        from .renderer import render_article13_markdown

        return render_article13_markdown(self)
