"""EU AI Act compliance module for insurance pricing models.

Provides tooling for Regulation (EU) 2024/1689 ('the EU AI Act') obligations
that apply to insurance pricing systems classified as high-risk under Annex III
point 5(c):

- ``AIActClassifier`` — two-stage Article 6 / Annex III scope test to determine
  whether a model is an AI system and whether it falls in the high-risk category.
- ``Article13Document`` — structured dataclass mapping to Article 13(3)(a)–(e)
  mandatory transparency document content, with accuracy computation and gap
  detection.
- ``ConformityAssessment`` — seven-step Annex VI internal control self-assessment
  pack, with automated checks against an attached Article 13 document.
- ``render_article13_markdown``, ``render_conformity_markdown``,
  ``article13_to_html`` — rendering utilities.

Quick start::

    from insurance_governance.euaia import (
        AIActClassifier,
        Article13Document,
        ConformityAssessment,
        ClassificationResult,
        RiskClassification,
        StepStatus,
    )

    clf = AIActClassifier()
    result = clf.classify(
        model_type="gradient_boosting",
        line_of_business="life",
        uses_personal_data=True,
        automated_decision=True,
    )
    # result.risk_classification == RiskClassification.HIGH_RISK

Regulatory scope note:
    Motor and property insurance pricing are NOT high-risk under Annex III.
    Life and health insurance using ML with personal data IS high-risk per the
    EIOPA opinion on Annex III 5(c). GLMs may be outside scope per EC Guidelines
    C/2025/3554 §42 but operators should verify this with legal counsel.
"""

from .article13 import Article13Document
from .classifier import AIActClassifier, ClassificationResult, RiskClassification, ModelType
from .conformity import ConformityAssessment, AssessmentStep, StepStatus
from .renderer import render_article13_markdown, render_conformity_markdown, article13_to_html

__all__ = [
    "Article13Document",
    "AIActClassifier",
    "ClassificationResult",
    "RiskClassification",
    "ModelType",
    "ConformityAssessment",
    "AssessmentStep",
    "StepStatus",
    "render_article13_markdown",
    "render_conformity_markdown",
    "article13_to_html",
]
