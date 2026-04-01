"""
OutcomeResult and OutcomeSuite — structured result types for Consumer Duty outcome testing.

Every test in this subpackage returns an OutcomeResult. The design mirrors
TestResult in the validation subpackage: a consistent return type regardless
of which metric produced it, so downstream reporting is uniform.

OutcomeSuite is a thin container for a collection of results with convenience
properties for pass/fail counts and severity breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insurance_governance.validation.results import RAGStatus, Severity


def _compute_outcome_rag(results: list["OutcomeResult"]) -> RAGStatus:
    """
    Derive overall RAG from a list of OutcomeResults.

    RED  — any CRITICAL failure.
    AMBER — any WARNING failure, no CRITICAL.
    GREEN — all pass, or only INFO failures.
    """
    has_critical = any(
        not r.passed and r.severity == Severity.CRITICAL for r in results
    )
    has_warning = any(
        not r.passed and r.severity == Severity.WARNING for r in results
    )
    if has_critical:
        return RAGStatus.RED
    if has_warning:
        return RAGStatus.AMBER
    return RAGStatus.GREEN


@dataclass
class OutcomeResult:
    """
    Structured result from a single Consumer Duty outcome test.

    Attributes
    ----------
    outcome:
        Which FCA Consumer Duty outcome this relates to.
        One of ``'price_value'``, ``'claims'``, ``'support'``.
    test_name:
        Short identifier, e.g. ``'fair_value_ratio'`` or ``'decline_rate_disparity'``.
    passed:
        True if the metric is within the acceptable threshold.
    metric_value:
        Primary numeric output. None for qualitative checks.
    threshold:
        The threshold the metric was tested against. None if no numeric threshold.
    period:
        The reporting period this result covers, e.g. ``'2025-Q4'``.
    segment:
        Customer segment label if this result is segment-specific. None for portfolio-wide.
    details:
        Human-readable explanation of what was tested and what the result means.
        Written for a board reviewer who was not present when the data was analysed.
    severity:
        Severity of a failure. Use CRITICAL for findings that require immediate
        action before the next pricing cycle. WARNING for findings that need
        remediation within the review period. INFO for informational results.
    corrective_actions:
        List of concrete actions to take if this test fails. Empty for passing results.
    extra:
        Additional structured data not included in summary tables. Use for
        per-segment breakdowns, distribution summaries, or raw counts.
    """

    outcome: str
    test_name: str
    passed: bool
    metric_value: float | None = None
    threshold: float | None = None
    period: str = ""
    segment: str | None = None
    details: str = ""
    severity: Severity = Severity.INFO
    corrective_actions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    _VALID_OUTCOMES = frozenset({"price_value", "claims", "support"})

    def __post_init__(self) -> None:
        if self.outcome not in self._VALID_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {sorted(self._VALID_OUTCOMES)}, "
                f"got {self.outcome!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON export."""
        return {
            "outcome": self.outcome,
            "test_name": self.test_name,
            "passed": self.passed,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "period": self.period,
            "segment": self.segment,
            "details": self.details,
            "severity": self.severity.value,
            "corrective_actions": self.corrective_actions,
            "extra": self.extra,
        }


@dataclass
class OutcomeSuite:
    """
    Container for a collection of OutcomeResults with convenience properties.

    Parameters
    ----------
    results:
        The outcome test results.
    period:
        The reporting period these results cover.
    """

    results: list[OutcomeResult]
    period: str = ""

    @property
    def passed(self) -> list[OutcomeResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[OutcomeResult]:
        return [r for r in self.results if not r.passed]

    @property
    def critical_failures(self) -> list[OutcomeResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == Severity.CRITICAL
        ]

    @property
    def warning_failures(self) -> list[OutcomeResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == Severity.WARNING
        ]

    @property
    def rag_status(self) -> RAGStatus:
        return _compute_outcome_rag(self.results)

    def by_outcome(self, outcome: str) -> list[OutcomeResult]:
        """Filter results for a specific outcome area."""
        return [r for r in self.results if r.outcome == outcome]

    def vulnerable_segment_results(self) -> list[OutcomeResult]:
        """Return results tagged with a segment — caller marks vulnerable segments."""
        return [r for r in self.results if r.segment is not None]

    def summary(self) -> dict[str, Any]:
        """Return a plain-dict summary suitable for report headers."""
        return {
            "period": self.period,
            "total": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "critical": len(self.critical_failures),
            "warnings": len(self.warning_failures),
            "rag_status": self.rag_status.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }
