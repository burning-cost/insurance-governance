"""insurance_governance.validation: model validation for UK GI pricing models.

Structured to meet the evidence requirements of FCA Consumer Duty (PRIN 2A) and TR24/2,
and to support PRA SoP3/24 annual attestation (IMOR). The validation suite follows
SS1/23 best practice by analogy — SS1/23 is a banking supervisory statement and does not
apply directly to Solvency II insurers, but its model governance principles are widely
referenced in UK insurer MRM frameworks.

Quick start (high-level API)
-----------------------------
    import numpy as np
    from insurance_governance.validation import ModelValidationReport, ModelCard

    card = ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area", "vehicle_group"],
        limitations=["No telematics data", "Limited fleet exposure"],
        owner="Pricing Team",
    )

    report = ModelValidationReport(
        model_card=card,
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_val=exposure_val,
        y_train=y_train,
        y_pred_train=y_pred_train,
    )

    report.generate("validation_report.html")
    report.to_json("validation_report.json")
"""
from .data_quality import DataQualityReport
from .discrimination import DiscriminationReport
from .model_card import ModelCard
from .performance import PerformanceReport
from .report import ReportGenerator
from .results import RAGStatus, Severity, TestCategory, TestResult
from .stability import StabilityReport
from .validation_report import ModelValidationReport

__all__ = [
    # High-level facade
    "ModelValidationReport",
    # Lower-level components
    "ModelCard",
    "DataQualityReport",
    "PerformanceReport",
    "DiscriminationReport",
    "StabilityReport",
    "ReportGenerator",
    # Result types
    "TestResult",
    "TestCategory",
    "Severity",
    "RAGStatus",
]
