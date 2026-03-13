"""RiskTierScorer: objective, auditable risk tier assignment for pricing models.

The scorer maps six dimensions onto a 0-100 composite score. The score
determines the tier (1 = Critical/Material, 2 = Significant, 3 = Informational).
Weights and thresholds are configurable but the defaults reflect PRA SS1/23
Principle 1 expectations and UK personal lines calibration.

Why a scorecard rather than a decision tree? A scorecard produces a continuous
score that changes incrementally as circumstances change, and generates a
verbose rationale string. Both properties matter when you're presenting tier
assignments to a Model Risk Committee and want to show your working.

The six dimensions are:
  1. Materiality (GWP impacted) — 25 pts max
  2. Complexity (model architecture and feature count) — 20 pts max
  3. Data quality / external data use — 10 pts max
  4. Validation coverage (has a recent validation been run?) — 10 pts max
  5. Drift history (monitoring trigger history) — 10 pts max
  6. Regulatory exposure — 25 pts max (production status 15 + regulatory use 10)

Total = 100 pts. Tier thresholds:
  Tier 1 (Critical):     60+ pts — Annual review, MRC sign-off
  Tier 2 (Significant):  30-59 pts — 18-month review, Chief Actuary sign-off
  Tier 3 (Informational): < 30 pts — 24-month review, Head of Pricing sign-off

Note: the research (KB 794-795) uses slightly different labels and combines
production status + customer-facing into the regulatory exposure bucket.
The implementation below follows the task specification's 6-dimension structure
while keeping the same GWP thresholds and overall calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Critical",
    2: "High",
    3: "Medium",
    4: "Low",
}

TIER_REVIEW_FREQUENCY = {
    1: "Annual",
    2: "18 months",
    3: "24 months",
    4: "24 months",
}

TIER_SIGN_OFF = {
    1: "Model Risk Committee",
    2: "Chief Actuary",
    3: "Head of Pricing",
    4: "Head of Pricing",
}

# Default thresholds for tier assignment (score >= threshold -> that tier)
DEFAULT_TIER_THRESHOLDS = {
    1: 60,
    2: 30,
    3: 0,
}

# Default weights: max points per dimension (must sum to 100)
DEFAULT_WEIGHTS = {
    "materiality": 25,
    "complexity": 20,
    "data_quality": 10,
    "validation_coverage": 10,
    "drift_history": 10,
    "regulatory_exposure": 25,
}


@dataclass
class DimensionScore:
    """Score and rationale for a single scoring dimension.

    Args:
        name: Dimension name.
        score: Points awarded for this dimension.
        max_score: Maximum possible points for this dimension.
        rationale: Explanation of why this score was awarded.
    """

    name: str
    score: float
    max_score: float
    rationale: str

    @property
    def pct(self) -> float:
        """Score as a percentage of the maximum possible."""
        if self.max_score == 0:
            return 0.0
        return round(self.score / self.max_score * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "max_score": self.max_score,
            "pct": self.pct,
            "rationale": self.rationale,
        }


@dataclass
class TierResult:
    """Result of a risk tier scoring exercise.

    Args:
        tier: Numeric tier (1, 2, or 3). 1 is the highest risk.
        tier_label: Human-readable label (``'Critical'``, ``'High'``, etc.).
        score: Total composite score (0-100).
        dimensions: List of :class:`DimensionScore` objects, one per dimension.
        rationale: Full verbose rationale string for audit purposes.
        review_frequency: Recommended review frequency for this tier.
        sign_off_requirement: Required sign-off level for this tier.
        weights_used: The weight configuration used for this scoring run.
        thresholds_used: The tier threshold configuration used.
    """

    tier: int
    tier_label: str
    score: float
    dimensions: list[DimensionScore]
    rationale: str
    review_frequency: str
    sign_off_requirement: str
    weights_used: dict[str, float] = field(default_factory=dict)
    thresholds_used: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "tier": self.tier,
            "tier_label": self.tier_label,
            "score": self.score,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "rationale": self.rationale,
            "review_frequency": self.review_frequency,
            "sign_off_requirement": self.sign_off_requirement,
            "weights_used": self.weights_used,
            "thresholds_used": {str(k): v for k, v in self.thresholds_used.items()},
        }


class RiskTierScorer:
    """Compute a composite risk tier score for an insurance pricing model.

    The scorer is stateless — call :meth:`score` as many times as you like.
    Weights and thresholds can be overridden at construction time or per-call.

    Default weights sum to 100 and reflect PRA SS1/23 Principle 1 expectations.

    Args:
        weights: Override dimension weights. Must have the same keys as
            :data:`DEFAULT_WEIGHTS`. Values need not sum to 100; they will be
            normalised.
        thresholds: Override tier thresholds (score >= threshold -> tier).

    Examples::

        scorer = RiskTierScorer()
        result = scorer.score(
            gwp_impacted=125_000_000,
            model_complexity='high',
            deployment_status='champion',
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        print(result.tier, result.score, result.rationale)
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[int, float]] = None,
    ) -> None:
        if weights is not None:
            missing = set(DEFAULT_WEIGHTS) - set(weights)
            if missing:
                raise ValueError(
                    f"Custom weights are missing dimensions: {missing}"
                )
            extra = set(weights) - set(DEFAULT_WEIGHTS)
            if extra:
                raise ValueError(
                    f"Custom weights contain unknown dimensions: {extra}"
                )
            self.weights = dict(weights)
        else:
            self.weights = dict(DEFAULT_WEIGHTS)

        if thresholds is not None:
            self.thresholds = dict(thresholds)
        else:
            self.thresholds = dict(DEFAULT_TIER_THRESHOLDS)

        # Normalise weights so they sum to 100
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        scale = 100.0 / total
        self.weights = {k: v * scale for k, v in self.weights.items()}

    def score(
        self,
        gwp_impacted: float,
        model_complexity: str,
        deployment_status: str,
        regulatory_use: bool,
        external_data: bool,
        customer_facing: bool,
        validation_months_ago: Optional[float] = None,
        drift_triggers_last_year: int = 0,
    ) -> TierResult:
        """Score a model and return a :class:`TierResult`.

        Args:
            gwp_impacted: Gross written premium (£) influenced by this model.
            model_complexity: ``'low'``, ``'medium'``, or ``'high'``. Reflects
                model architecture and feature count.
            deployment_status: ``'champion'``, ``'challenger'``, ``'shadow'``,
                ``'development'``, or ``'retired'``.
            regulatory_use: Whether the model feeds Solvency II internal model
                or regulatory capital calculations.
            external_data: Whether the model uses external data sources
                (postcode deprivation, credit reference, telematics).
            customer_facing: Whether the model directly sets prices paid by
                customers.
            validation_months_ago: How many months ago the last independent
                validation was run. ``None`` means never validated (worst score).
            drift_triggers_last_year: Number of monitoring trigger events fired
                in the last 12 months (PSI, A/E, or Gini thresholds breached).

        Returns:
            :class:`TierResult` with composite score, per-dimension breakdown,
            and verbose rationale.
        """
        model_complexity = model_complexity.lower()
        deployment_status = deployment_status.lower()

        if model_complexity not in ("low", "medium", "high"):
            raise ValueError(
                f"model_complexity must be 'low', 'medium', or 'high'; "
                f"got {model_complexity!r}"
            )
        if deployment_status not in (
            "champion", "challenger", "shadow", "development", "retired"
        ):
            raise ValueError(
                f"deployment_status must be one of 'champion', 'challenger', "
                f"'shadow', 'development', 'retired'; got {deployment_status!r}"
            )

        dimensions: list[DimensionScore] = []

        # 1. Materiality (GWP)
        mat = self._score_materiality(gwp_impacted)
        dimensions.append(mat)

        # 2. Complexity
        cplx = self._score_complexity(model_complexity)
        dimensions.append(cplx)

        # 3. Data quality (external data use)
        dq = self._score_data_quality(external_data)
        dimensions.append(dq)

        # 4. Validation coverage
        vc = self._score_validation_coverage(validation_months_ago)
        dimensions.append(vc)

        # 5. Drift history
        dh = self._score_drift_history(drift_triggers_last_year)
        dimensions.append(dh)

        # 6. Regulatory exposure (production status + regulatory use + customer-facing)
        re_ = self._score_regulatory_exposure(
            deployment_status, regulatory_use, customer_facing
        )
        dimensions.append(re_)

        # Composite score: each dimension is scored on its raw max, then
        # scaled by the normalised weight.
        composite = 0.0
        for dim in dimensions:
            composite += (dim.score / dim.max_score) * self.weights[dim.name]
        composite = round(composite, 1)

        tier = self._assign_tier(composite)
        rationale = self._build_rationale(tier, composite, dimensions)

        return TierResult(
            tier=tier,
            tier_label=TIER_LABELS[tier],
            score=composite,
            dimensions=dimensions,
            rationale=rationale,
            review_frequency=TIER_REVIEW_FREQUENCY[tier],
            sign_off_requirement=TIER_SIGN_OFF[tier],
            weights_used=dict(self.weights),
            thresholds_used=dict(self.thresholds),
        )

    # ------------------------------------------------------------------
    # Per-dimension scoring methods
    # ------------------------------------------------------------------

    def _score_materiality(self, gwp: float) -> DimensionScore:
        max_score = 100.0  # raw; will be weighted
        if gwp >= 100_000_000:
            raw = 100.0
            label = f"GWP £{gwp/1_000_000:.0f}m ≥ £100m"
        elif gwp >= 25_000_000:
            raw = 60.0
            label = f"GWP £{gwp/1_000_000:.0f}m (£25m–£100m)"
        elif gwp >= 5_000_000:
            raw = 32.0
            label = f"GWP £{gwp/1_000_000:.1f}m (£5m–£25m)"
        else:
            raw = 12.0
            label = f"GWP £{gwp/1_000_000:.2f}m (<£5m)"
        return DimensionScore(
            name="materiality",
            score=raw,
            max_score=max_score,
            rationale=label,
        )

    def _score_complexity(self, complexity: str) -> DimensionScore:
        mapping = {"high": 100.0, "medium": 60.0, "low": 25.0}
        raw = mapping[complexity]
        labels = {
            "high": "High complexity (GBM/ensemble/NN, >50 features)",
            "medium": "Medium complexity (GLM with interactions, regularised, shallow tree)",
            "low": "Low complexity (simple GLM, lookup table, linear model)",
        }
        return DimensionScore(
            name="complexity",
            score=raw,
            max_score=100.0,
            rationale=labels[complexity],
        )

    def _score_data_quality(self, external_data: bool) -> DimensionScore:
        raw = 100.0 if external_data else 0.0
        rationale = (
            "Uses external data (postcode deprivation / credit reference / telematics)"
            if external_data
            else "No external data sources"
        )
        return DimensionScore(
            name="data_quality",
            score=raw,
            max_score=100.0,
            rationale=rationale,
        )

    def _score_validation_coverage(
        self, months_ago: Optional[float]
    ) -> DimensionScore:
        if months_ago is None:
            raw = 100.0
            rationale = "Never independently validated"
        elif months_ago <= 6:
            raw = 0.0
            rationale = f"Validated {months_ago:.0f} months ago (within 6 months)"
        elif months_ago <= 12:
            raw = 20.0
            rationale = f"Validated {months_ago:.0f} months ago (6-12 months)"
        elif months_ago <= 18:
            raw = 50.0
            rationale = f"Validated {months_ago:.0f} months ago (12-18 months)"
        elif months_ago <= 24:
            raw = 75.0
            rationale = f"Validated {months_ago:.0f} months ago (18-24 months)"
        else:
            raw = 100.0
            rationale = f"Validated {months_ago:.0f} months ago (overdue)"
        return DimensionScore(
            name="validation_coverage",
            score=raw,
            max_score=100.0,
            rationale=rationale,
        )

    def _score_drift_history(self, triggers: int) -> DimensionScore:
        if triggers == 0:
            raw = 0.0
            rationale = "No monitoring triggers fired in last 12 months"
        elif triggers == 1:
            raw = 33.0
            rationale = "1 monitoring trigger in last 12 months"
        elif triggers <= 3:
            raw = 67.0
            rationale = f"{triggers} monitoring triggers in last 12 months"
        else:
            raw = 100.0
            rationale = f"{triggers} monitoring triggers in last 12 months (elevated drift)"
        return DimensionScore(
            name="drift_history",
            score=raw,
            max_score=100.0,
            rationale=rationale,
        )

    def _score_regulatory_exposure(
        self,
        deployment_status: str,
        regulatory_use: bool,
        customer_facing: bool,
    ) -> DimensionScore:
        # Production status sub-score (max 40/100)
        status_scores = {
            "champion": 40.0,
            "challenger": 20.0,
            "shadow": 15.0,
            "development": 0.0,
            "retired": 0.0,
        }
        status_raw = status_scores[deployment_status]
        status_label = f"deployment_status={deployment_status}"

        # Regulatory use sub-score (max 30/100)
        reg_raw = 30.0 if regulatory_use else 0.0
        reg_label = (
            "regulatory use (Solvency II / regulatory capital)" if regulatory_use
            else "no regulatory use"
        )

        # Customer-facing sub-score (max 30/100)
        cf_raw = 30.0 if customer_facing else 0.0
        cf_label = "directly customer-facing pricing" if customer_facing else "not directly customer-facing"

        total_raw = status_raw + reg_raw + cf_raw
        rationale = f"{status_label}, {reg_label}, {cf_label}"

        return DimensionScore(
            name="regulatory_exposure",
            score=total_raw,
            max_score=100.0,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Tier assignment and rationale
    # ------------------------------------------------------------------

    def _assign_tier(self, score: float) -> int:
        """Map a composite score to a tier.

        Iterates tier thresholds in descending order (highest risk first).
        Returns 3 as the floor.
        """
        for tier in sorted(self.thresholds.keys()):
            if score >= self.thresholds[tier]:
                return tier
        return max(self.thresholds.keys())

    def _build_rationale(
        self,
        tier: int,
        score: float,
        dimensions: list[DimensionScore],
    ) -> str:
        """Build the verbose rationale string for audit purposes."""
        label = TIER_LABELS[tier]
        threshold = self.thresholds.get(tier, 0)
        lines = [
            f"Tier {tier} ({label}) assigned: composite score {score:.1f}/100 "
            f"(threshold ≥ {threshold})",
            "",
            "Dimension breakdown:",
        ]
        for dim in dimensions:
            contribution = (dim.score / dim.max_score) * self.weights[dim.name]
            lines.append(
                f"  {dim.name}: {contribution:.1f}pts "
                f"({dim.score:.0f}/{dim.max_score:.0f} × weight {self.weights[dim.name]:.1f}) "
                f"— {dim.rationale}"
            )
        lines.append("")
        lines.append(
            f"Review frequency: {TIER_REVIEW_FREQUENCY[tier]}. "
            f"Sign-off required: {TIER_SIGN_OFF[tier]}."
        )
        return "\n".join(lines)
