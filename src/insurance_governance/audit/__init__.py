"""insurance_governance.audit: explainability audit trail for UK insurance pricing models.

Provides tamper-evident recording of model prediction events, SHAP-based
feature explanations, plain English customer-facing summaries, and periodic
audit reports for regulatory submission.

Typical workflow::

    from insurance_governance.audit import (
        ExplainabilityAuditEntry,
        ExplainabilityAuditLog,
        SHAPExplainer,
        PlainLanguageExplainer,
        AuditSummaryReport,
    )

    # 1. Create an entry for each prediction
    entry = ExplainabilityAuditEntry(
        model_id='motor-freq-v3',
        model_version='3.1.0',
        input_features={'driver_age': 32, 'ncb_years': 5, 'region': 'SE'},
        feature_importances={'driver_age': -0.12, 'ncb_years': -0.31, 'region': 0.08},
        prediction=412.50,
        final_premium=412.50,
        decision_basis='model_output',
    )

    # 2. Append to an append-only log
    log = ExplainabilityAuditLog('audit.jsonl', 'motor-freq-v3', '3.1.0')
    log.append(entry)

    # 3. Generate a customer explanation
    explainer = PlainLanguageExplainer(
        feature_labels={
            'driver_age': 'your age',
            'ncb_years': 'your no-claims discount',
            'region': 'your postcode area',
        }
    )
    text = explainer.generate(entry, base_premium=350.00)

    # 4. Generate a periodic audit report
    report = AuditSummaryReport(log, period='2025-Q4')
    report.save_html('audit_report.html')
"""

from .customer_explanation import PlainLanguageExplainer
from .entry import ExplainabilityAuditEntry
from .log import ExplainabilityAuditLog
from .report import AuditSummaryReport
from .shap_explainer import SHAPExplainer

__all__ = [
    "AuditSummaryReport",
    "ExplainabilityAuditEntry",
    "ExplainabilityAuditLog",
    "PlainLanguageExplainer",
    "SHAPExplainer",
]
