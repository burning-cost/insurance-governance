"""EU AI Act scope and risk classification for insurance pricing models.

Implements the two-stage test required before compliance obligations attach:

1. Is the system an "AI system" per Article 3(1) of Regulation (EU) 2024/1689?
2. If so, does it fall under Annex III point 5(c) (high-risk, insurance)?

Key regulatory reference points:

- Article 3(1) definition of AI system: machine-based system designed to operate
  with varying levels of autonomy that infers how to generate outputs that can
  influence real or virtual environments.
- Annex III 5(c): systems used to evaluate the creditworthiness of natural persons
  or establish their credit score (EIOPA opinion extends this to life/health
  insurance pricing that uses personal data for risk differentiation).
- EC Guidelines C/2025/3554 paragraph 42: traditional statistical methods
  (linear/logistic regression, GLMs) without learning components are not AI
  systems under the Act's definition — they are "optimisation" tools.
- Motor and property pricing are NOT listed in Annex III; they remain outside
  the high-risk classification regardless of method.

Conservative classification principle applied: where a model uses ML (gradient
boosting, neural networks, random forests, etc.) and touches life or health
insurance, the classifier returns POTENTIALLY_HIGH_RISK even if the operator
believes it falls outside. The burden of proof is on the provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskClassification(str, Enum):
    """Classification outcome under Annex III of Regulation (EU) 2024/1689."""

    HIGH_RISK = "high_risk"
    """System is in scope of Annex III 5(c). Full compliance obligations apply."""

    POTENTIALLY_HIGH_RISK = "potentially_high_risk"
    """System may be in scope. Operator should seek legal review before deploying."""

    NOT_HIGH_RISK = "not_high_risk"
    """System does not meet Annex III 5(c) criteria (e.g. motor/property pricing)."""

    OUT_OF_SCOPE = "out_of_scope"
    """System is not an AI system per Article 3(1) (e.g. a pure GLM without
    learning components per EC Guidelines C/2025/3554 §42)."""


class ModelType(str, Enum):
    """Broad taxonomy of modelling approaches for scope determination."""

    GLM = "glm"
    """Generalised linear model. Likely out of scope per C/2025/3554 §42."""

    GAM = "gam"
    """Generalised additive model. Borderline — no consensus yet; treat conservatively."""

    GRADIENT_BOOSTING = "gradient_boosting"
    """XGBoost, LightGBM, CatBoost etc. In scope as AI system."""

    NEURAL_NETWORK = "neural_network"
    """Deep learning models. In scope as AI system."""

    RANDOM_FOREST = "random_forest"
    """Ensemble of decision trees with randomisation. In scope as AI system."""

    DECISION_TREE = "decision_tree"
    """Single decision tree. Borderline — simple trees may fall outside definition."""

    REGULARISED_REGRESSION = "regularised_regression"
    """Lasso, Ridge, Elastic Net. Treated as AI system due to learning component."""

    OTHER_ML = "other_ml"
    """Any other machine-learning approach. Conservative: treat as AI system."""

    RULE_BASED = "rule_based"
    """Deterministic rule engine with no learning. Out of scope."""

    UNKNOWN = "unknown"
    """Model type not specified. Conservative: assume AI system."""


# Lines of business that fall under Annex III 5(c) per EIOPA opinion
_HIGH_RISK_LOB: frozenset[str] = frozenset(
    {
        "life",
        "health",
        "critical_illness",
        "income_protection",
        "long_term_care",
        "private_medical_insurance",
        "pmi",
        "term_life",
        "whole_of_life",
        "annuity",
    }
)

# Model types that are AI systems per Article 3(1)
_AI_SYSTEM_MODEL_TYPES: frozenset[ModelType] = frozenset(
    {
        ModelType.GRADIENT_BOOSTING,
        ModelType.NEURAL_NETWORK,
        ModelType.RANDOM_FOREST,
        ModelType.REGULARISED_REGRESSION,
        ModelType.OTHER_ML,
        ModelType.UNKNOWN,
        ModelType.GAM,  # conservative
    }
)

# Model types that are definitively out of scope per C/2025/3554 §42
_OUT_OF_SCOPE_MODEL_TYPES: frozenset[ModelType] = frozenset(
    {
        ModelType.GLM,
        ModelType.RULE_BASED,
    }
)


@dataclass
class ClassificationResult:
    """Output of ``AIActClassifier.classify()``.

    All reasoning is captured in ``rationale`` so operators can attach this
    directly to their technical documentation (Annex IV).
    """

    is_ai_system: bool
    """Whether the system meets the Article 3(1) definition."""

    risk_classification: RiskClassification
    """Annex III risk tier."""

    requires_conformity_assessment: bool
    """Whether Article 43 conformity assessment obligations apply."""

    assessment_route: str
    """'internal_control' (Annex VI self-assessment) or 'notified_body' or 'n/a'."""

    rationale: list[str] = field(default_factory=list)
    """Ordered list of regulatory reasoning steps that produced this result."""

    warnings: list[str] = field(default_factory=list)
    """Items requiring attention before a formal legal opinion can be issued."""


class AIActClassifier:
    """Classify an insurance pricing model under Regulation (EU) 2024/1689.

    This is a decision-support tool, not legal advice. The outputs provide
    a starting point for an operator's own assessment under Article 6 and
    Annex III. Where ``risk_classification`` is POTENTIALLY_HIGH_RISK, the
    operator must seek qualified legal review before deploying the system.

    Examples
    --------
    Motor pricing with XGBoost — not high-risk:

    >>> clf = AIActClassifier()
    >>> result = clf.classify(
    ...     model_type="gradient_boosting",
    ...     line_of_business="motor",
    ...     uses_personal_data=True,
    ...     automated_decision=True,
    ... )
    >>> result.risk_classification
    <RiskClassification.NOT_HIGH_RISK: 'not_high_risk'>

    Life insurance pricing with gradient boosting — high-risk:

    >>> result = clf.classify(
    ...     model_type="gradient_boosting",
    ...     line_of_business="life",
    ...     uses_personal_data=True,
    ...     automated_decision=True,
    ... )
    >>> result.risk_classification
    <RiskClassification.HIGH_RISK: 'high_risk'>
    """

    def classify(
        self,
        model_type: str,
        line_of_business: str,
        uses_personal_data: bool,
        automated_decision: bool,
    ) -> ClassificationResult:
        """Run the two-stage Article 6 / Annex III classification test.

        Parameters
        ----------
        model_type:
            One of the ``ModelType`` enum values (case-insensitive string).
            Use ``"glm"`` for traditional actuarial frequency/severity models,
            ``"gradient_boosting"`` for XGBoost/LightGBM etc.
        line_of_business:
            Short string identifying the line. Values like ``"life"``,
            ``"health"``, ``"motor"``, ``"property"`` are recognised.
            Unrecognised values are treated conservatively.
        uses_personal_data:
            Whether the model ingests personal data about the policyholder
            (name, age, health, behaviour etc.).
        automated_decision:
            Whether the model output is used to make or materially influence
            a decision about an individual without mandatory human review.
        """
        rationale: list[str] = []
        warnings: list[str] = []

        # --- Normalise inputs ---
        try:
            mt = ModelType(model_type.lower())
        except ValueError:
            mt = ModelType.UNKNOWN
            warnings.append(
                f"Unrecognised model_type '{model_type}'. "
                "Defaulting to ModelType.UNKNOWN — treated as AI system."
            )

        lob_normalised = line_of_business.lower().strip().replace(" ", "_")

        # --- Stage 1: Is this an AI system per Article 3(1)? ---
        if mt in _OUT_OF_SCOPE_MODEL_TYPES:
            rationale.append(
                f"Model type '{mt.value}' is a traditional statistical/optimisation "
                "method. Per EC Guidelines C/2025/3554 §42, traditional statistical "
                "approaches (including GLMs) without learning components are not AI "
                "systems under Article 3(1)."
            )
            return ClassificationResult(
                is_ai_system=False,
                risk_classification=RiskClassification.OUT_OF_SCOPE,
                requires_conformity_assessment=False,
                assessment_route="n/a",
                rationale=rationale,
                warnings=[
                    "Verify that the model has no embedded learning components "
                    "(e.g. auto-selected interaction terms, boosted residuals). "
                    "If it does, reclassify as 'other_ml'."
                ],
            )

        if mt in _AI_SYSTEM_MODEL_TYPES:
            rationale.append(
                f"Model type '{mt.value}' uses machine-learning inference. "
                "This meets the Article 3(1) definition of an AI system."
            )
        else:
            # decision_tree — borderline
            rationale.append(
                f"Model type '{mt.value}' is borderline under Article 3(1). "
                "Simple decision trees may be treated as rule-based systems, "
                "but any tree with learned thresholds is likely an AI system. "
                "Conservative classification applied: treated as AI system."
            )
            warnings.append(
                "Decision tree classification is contested. Obtain legal opinion "
                "on Article 3(1) applicability for your specific implementation."
            )

        # --- Stage 2: Annex III 5(c) high-risk determination ---
        is_high_risk_lob = lob_normalised in _HIGH_RISK_LOB

        if not is_high_risk_lob:
            # Unrecognised LOB — warn but treat as not in Annex III
            if lob_normalised not in {
                "motor", "property", "home", "commercial_property",
                "pet", "travel", "liability", "cyber",
            }:
                warnings.append(
                    f"Line of business '{line_of_business}' is not in the recognised "
                    "set. If it involves individual life/health risk assessment, "
                    "reclassify as a high-risk LOB."
                )
            rationale.append(
                f"Line of business '{line_of_business}' is not listed in Annex III "
                "point 5(c) (which covers life/health insurance risk assessment). "
                "Motor and property pricing are explicitly outside the Annex III "
                "high-risk categories."
            )
            return ClassificationResult(
                is_ai_system=True,
                risk_classification=RiskClassification.NOT_HIGH_RISK,
                requires_conformity_assessment=False,
                assessment_route="n/a",
                rationale=rationale,
                warnings=warnings,
            )

        # Life/health LOB — check personal data and automated decision conditions
        rationale.append(
            f"Line of business '{line_of_business}' falls within the EIOPA-extended "
            "scope of Annex III 5(c): AI systems used to evaluate individual risk "
            "for life/health insurance purposes."
        )

        if not uses_personal_data:
            rationale.append(
                "Model does not use personal data. Annex III 5(c) targets systems "
                "that evaluate individual natural persons. A purely aggregate/index "
                "model is arguably outside scope."
            )
            warnings.append(
                "Confirm that no personal data enters the model at any stage, "
                "including upstream feature engineering."
            )
            return ClassificationResult(
                is_ai_system=True,
                risk_classification=RiskClassification.POTENTIALLY_HIGH_RISK,
                requires_conformity_assessment=True,
                assessment_route="internal_control",
                rationale=rationale,
                warnings=warnings,
            )

        if not automated_decision:
            rationale.append(
                "System output is subject to mandatory human review before any "
                "individual decision is made. Article 14 human oversight obligations "
                "remain, but the intensity of Annex III obligations may be reduced."
            )
            warnings.append(
                "Ensure human oversight measures are formally documented per "
                "Article 14(4). If human reviewers routinely accept model output "
                "without genuine review, the system may be de facto automated."
            )
            return ClassificationResult(
                is_ai_system=True,
                risk_classification=RiskClassification.POTENTIALLY_HIGH_RISK,
                requires_conformity_assessment=True,
                assessment_route="internal_control",
                rationale=rationale,
                warnings=warnings,
            )

        # All conditions met: definite high-risk
        rationale.append(
            "System is an AI system (Article 3(1)), operates in a high-risk sector "
            "(Annex III 5(c)), uses personal data to evaluate natural persons, and "
            "drives automated individual decisions. Full Article 9–15 obligations apply."
        )
        rationale.append(
            "Conformity assessment route: Annex VI internal control self-assessment "
            "(Article 43(2)). Notified body involvement is not required for insurance "
            "pricing systems unless they also perform a safety function."
        )

        return ClassificationResult(
            is_ai_system=True,
            risk_classification=RiskClassification.HIGH_RISK,
            requires_conformity_assessment=True,
            assessment_route="internal_control",
            rationale=rationale,
            warnings=warnings,
        )
