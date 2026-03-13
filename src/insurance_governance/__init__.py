"""insurance-governance: unified model governance for UK insurance pricing.

Merges PRA SS1/23 model validation (insurance-validation) and model risk
management (insurance-mrm) into a single package.

Two subpackages, one install:

- ``insurance_governance.validation`` — statistical validation tests, Gini,
  PSI, discrimination checks, HTML report generation.
- ``insurance_governance.mrm`` — ModelCard, ModelInventory, RiskTierScorer,
  GovernanceReport for Model Risk Committee packs.

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

__version__ = "0.1.0"

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
    "__version__",
]
