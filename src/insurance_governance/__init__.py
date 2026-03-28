"""insurance-governance: unified model governance for UK insurance pricing.

Merges model validation (insurance-validation), model risk management
(insurance-mrm), and EU AI Act compliance (euaia) into a single package.

Three subpackages, one install:

- ``insurance_governance.validation`` — statistical validation tests, Gini,
  PSI, discrimination checks, HTML report generation.
- ``insurance_governance.mrm`` — ModelCard, ModelInventory, RiskTierScorer,
  GovernanceReport for Model Risk Committee packs.
- ``insurance_governance.euaia`` — Article 13 transparency documents,
  Annex VI conformity assessment, and Annex III scope classification.

Most commonly used classes are re-exported at the top level for convenience::

    from insurance_governance import (
        # Validation
        ModelValidationReport,
        ValidationModelCard,
        PerformanceReport,
        # MRM
        MRMModelCard,
        ModelInventory,
        RiskTierScorer,
        GovernanceReport,
        # EU AI Act
        AIActClassifier,
        Article13Document,
        ConformityAssessment,
    )
"""

# Validation subpackage — re-export with alias to avoid ModelCard clash
from .validation import (
    ModelValidationReport,
    DataQualityReport,
    PerformanceReport,
    DiscriminationReport,
    StabilityReport,
    ReportGenerator,
    TestResult,
    TestCategory,
    Severity,
    RAGStatus,
)
from .validation import ModelCard as ValidationModelCard

# MRM subpackage — re-export with alias to avoid ModelCard clash
from .mrm import (
    Assumption,
    Limitation,
    DimensionScore,
    ModelInventory,
    RiskTierScorer,
    TierResult,
    GovernanceReport,
)
from .mrm import ModelCard as MRMModelCard

# EU AI Act subpackage
from .euaia import (
    AIActClassifier,
    Article13Document,
    ClassificationResult,
    ConformityAssessment,
    AssessmentStep,
    RiskClassification,
    ModelType,
    StepStatus,
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-governance")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # Validation
    "ModelValidationReport",
    "ValidationModelCard",
    "DataQualityReport",
    "PerformanceReport",
    "DiscriminationReport",
    "StabilityReport",
    "ReportGenerator",
    "TestResult",
    "TestCategory",
    "Severity",
    "RAGStatus",
    # MRM
    "MRMModelCard",
    "Assumption",
    "Limitation",
    "DimensionScore",
    "ModelInventory",
    "RiskTierScorer",
    "TierResult",
    "GovernanceReport",
    # EU AI Act
    "AIActClassifier",
    "Article13Document",
    "ClassificationResult",
    "ConformityAssessment",
    "AssessmentStep",
    "RiskClassification",
    "ModelType",
    "StepStatus",
    "__version__",
]
