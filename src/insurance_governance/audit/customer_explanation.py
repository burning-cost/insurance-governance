"""PlainLanguageExplainer — Consumer Duty-compliant premium explanations.

FCA PRIN 2A.4 requires that firms communicate pricing outcomes to customers in
plain English. This module converts SHAP feature importances into readable
sentences that explain why a premium is what it is, in terms the customer
understands — not model internals.

The design is deliberately simple. You supply:
1. A mapping from internal feature names to customer-facing labels.
2. A currency symbol (default: GBP).

Then for each prediction event you call ``generate()`` with the entry and the
base premium (the starting point before risk factors are applied). The output
is a paragraph that names the top factors, states whether they increased or
decreased the premium, and gives the approximate pound amount.

Usage::

    from insurance_governance.audit import PlainLanguageExplainer, ExplainabilityAuditEntry

    explainer = PlainLanguageExplainer(
        feature_labels={
            'driver_age': 'your age',
            'ncb_years': 'your no-claims discount',
            'region': 'your postcode area',
            'vehicle_age': 'the age of your vehicle',
        }
    )
    text = explainer.generate(entry, base_premium=350.00)

The output might read:
    "Your premium is £412.50. The main factors affecting your premium were:
    Your no-claims discount reduced your premium by £45.20.
    Your age reduced your premium by £18.30.
    Your postcode area added £95.40 to reflect local claim rates.
    Your vehicle age added £30.60."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .entry import ExplainabilityAuditEntry


@dataclass
class FactorContribution:
    """A single factor's contribution to the premium.

    Used internally by PlainLanguageExplainer to sort and format factors.
    """
    feature_name: str
    label: str
    shap_value: float
    premium_impact: float  # absolute pound impact


class PlainLanguageExplainer:
    """Generate plain English premium explanations for customers.

    Converts SHAP importances from an :class:`~.entry.ExplainabilityAuditEntry`
    into a paragraph a customer can understand. The explanation names the
    top factors, states direction (increased / reduced), and gives approximate
    pound amounts by scaling SHAP values relative to the base premium.

    Parameters
    ----------
    feature_labels:
        Mapping from internal feature names to customer-facing descriptions.
        For example: ``{'ncb_years': 'your no-claims discount'}``. Features
        not in this mapping are excluded from the explanation.
    currency:
        Currency symbol to use in the output. Defaults to ``'GBP'`` which
        renders as ``£``.
    max_factors:
        Maximum number of factors to include in the explanation. The factors
        with the largest absolute impact are shown. Defaults to 5.
    min_impact_pct:
        Minimum impact as a percentage of the base premium to be mentioned.
        Factors below this threshold are omitted as noise. Defaults to 1.0
        (one percent).
    """

    _CURRENCY_SYMBOLS = {"GBP": "£", "EUR": "€", "USD": "$"}

    def __init__(
        self,
        feature_labels: dict[str, str],
        currency: str = "GBP",
        max_factors: int = 5,
        min_impact_pct: float = 1.0,
    ) -> None:
        self._feature_labels = dict(feature_labels)
        self._currency = currency
        self._symbol = self._CURRENCY_SYMBOLS.get(currency, currency)
        self._max_factors = max_factors
        self._min_impact_pct = min_impact_pct

    @property
    def feature_labels(self) -> dict[str, str]:
        """The feature label mapping this explainer was built with."""
        return dict(self._feature_labels)

    def _format_amount(self, amount: float) -> str:
        """Format a pound amount with two decimal places."""
        return f"{self._symbol}{abs(amount):.2f}"

    def _compute_contributions(
        self,
        entry: ExplainabilityAuditEntry,
        base_premium: float,
    ) -> list[FactorContribution]:
        """Convert raw SHAP values to premium impacts.

        SHAP values are on the model's output scale. We scale them so that
        the sum of all SHAP values equals (final_premium - base_premium),
        giving the user an additive breakdown in pounds.

        When ``entry.final_premium`` is available, we anchor the scaling to
        the actual charged premium. Otherwise we use ``entry.prediction``.
        """
        if base_premium <= 0:
            raise ValueError(f"base_premium must be positive; got {base_premium}")

        importances = entry.feature_importances
        if not importances:
            return []

        total_shap = sum(importances.values())
        target_premium = (
            entry.final_premium
            if entry.final_premium is not None
            else entry.prediction
        )
        premium_range = target_premium - base_premium

        # Scale each SHAP value proportionally to the premium range.
        # If total_shap is zero (constant model), distribute nothing.
        scale = (premium_range / total_shap) if abs(total_shap) > 1e-10 else 0.0

        min_abs = base_premium * (self._min_impact_pct / 100.0)

        contributions: list[FactorContribution] = []
        for feature, shap_val in importances.items():
            label = self._feature_labels.get(feature)
            if label is None:
                continue  # not in the customer-facing mapping; skip
            impact = shap_val * scale
            if abs(impact) >= min_abs:
                contributions.append(
                    FactorContribution(
                        feature_name=feature,
                        label=label,
                        shap_value=shap_val,
                        premium_impact=impact,
                    )
                )

        # Sort descending by absolute impact
        contributions.sort(key=lambda c: abs(c.premium_impact), reverse=True)
        return contributions[: self._max_factors]

    def _factor_sentence(self, contrib: FactorContribution) -> str:
        """Render a single factor as a plain English sentence."""
        amount_str = self._format_amount(contrib.premium_impact)
        if contrib.premium_impact > 0:
            return f"{contrib.label.capitalize()} added {amount_str} to your premium."
        else:
            return (
                f"{contrib.label.capitalize()} reduced your premium by {amount_str}."
            )

    def generate(
        self,
        entry: ExplainabilityAuditEntry,
        base_premium: float,
        intro: Optional[str] = None,
    ) -> str:
        """Generate a plain English explanation for a single audit entry.

        Args:
            entry: The prediction event to explain.
            base_premium: The base premium before any risk factors are applied,
                in the same currency as ``final_premium`` or ``prediction``.
                Must be positive.
            intro: Optional custom introductory sentence. If not provided,
                a standard sentence stating the final premium is used.

        Returns:
            A plain English paragraph explaining the premium. Each material
            factor is listed as a separate sentence. If there are no
            explainable factors (either because ``feature_importances`` is
            empty or none pass the ``min_impact_pct`` filter), a fallback
            sentence is returned instead.
        """
        target_premium = (
            entry.final_premium
            if entry.final_premium is not None
            else entry.prediction
        )

        if intro is None:
            intro = (
                f"Your premium is {self._symbol}{target_premium:.2f}. "
                "The main factors affecting your premium were:"
            )

        contributions = self._compute_contributions(entry, base_premium)

        if not contributions:
            return (
                f"{intro} Your premium reflects the overall risk profile of your "
                "policy. For a detailed breakdown, please contact us."
            )

        sentences = [intro]
        for contrib in contributions:
            sentences.append(self._factor_sentence(contrib))

        # If an override was applied, note it at the end.
        if entry.override_applied and entry.override_reason:
            sentences.append(
                "Note: an underwriter reviewed and adjusted your premium. "
                f"Reason: {entry.override_reason}"
            )
        elif entry.decision_basis == "rule_fallback":
            sentences.append(
                "Your premium was also subject to minimum premium rules."
            )

        return " ".join(sentences)

    def generate_bullet_list(
        self,
        entry: ExplainabilityAuditEntry,
        base_premium: float,
    ) -> list[str]:
        """Return the premium explanation as a list of bullet-point strings.

        Useful for rendering in HTML templates or email clients that support
        basic list formatting. Each element is a plain string without bullet
        character — add your own ``'•'`` or ``'- '`` as needed.

        Args:
            entry: The prediction event to explain.
            base_premium: The base premium before risk factors.

        Returns:
            List of plain strings. First element is the summary sentence;
            remaining elements are individual factor sentences.
        """
        target_premium = (
            entry.final_premium
            if entry.final_premium is not None
            else entry.prediction
        )
        summary = f"Your premium: {self._symbol}{target_premium:.2f}"
        contributions = self._compute_contributions(entry, base_premium)
        bullets = [summary]
        for contrib in contributions:
            bullets.append(self._factor_sentence(contrib))
        return bullets
