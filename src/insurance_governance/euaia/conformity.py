"""Conformity assessment for high-risk AI systems: Annex VI internal control.

Regulation (EU) 2024/1689 Article 43(2) specifies that providers of high-risk
AI systems listed in Annex III (other than those covered by harmonised standards)
shall follow a conformity assessment procedure based on internal control as
described in Annex VI.

Annex VI sets out seven steps that the provider must document and retain as part
of their quality management system (QMS). This module provides a structured
representation of each step, automated checks where possible, and Markdown
rendering for the conformity assessment pack.

The seven Annex VI steps:
    1. Risk classification result (Article 6 + Annex III)
    2. Quality management system verification (Article 17)
    3. Technical documentation completeness (Annex IV)
    4. Risk management system (Article 9)
    5. Human oversight design (Article 14)
    6. Accuracy, robustness and cybersecurity (Article 15)
    7. EU Declaration of Conformity (Article 47)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    """Completion status of a single Annex VI step."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class AssessmentStep:
    """One step in the Annex VI self-assessment.

    Parameters
    ----------
    step_number:
        Integer 1–7 corresponding to the Annex VI sequence.
    title:
        Short title used in headings.
    regulatory_reference:
        The article or annex that this step implements.
    status:
        Whether the step has been completed, is incomplete, or does not apply.
    evidence:
        Free-text description of the evidence or artefact that satisfies the step.
    findings:
        List of specific findings, deficiencies, or observations for this step.
    """

    step_number: int
    title: str
    regulatory_reference: str
    status: StepStatus = StepStatus.INCOMPLETE
    evidence: str = ""
    findings: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Return True if the step is marked complete or not applicable."""
        return self.status in (StepStatus.COMPLETE, StepStatus.NOT_APPLICABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "regulatory_reference": self.regulatory_reference,
            "status": self.status.value,
            "evidence": self.evidence,
            "findings": self.findings,
        }


def _default_steps() -> list[AssessmentStep]:
    """Initialise the seven Annex VI steps with their regulatory references."""
    return [
        AssessmentStep(
            step_number=1,
            title="Risk Classification",
            regulatory_reference="Article 6 + Annex III",
        ),
        AssessmentStep(
            step_number=2,
            title="Quality Management System",
            regulatory_reference="Article 17",
        ),
        AssessmentStep(
            step_number=3,
            title="Technical Documentation",
            regulatory_reference="Annex IV",
        ),
        AssessmentStep(
            step_number=4,
            title="Risk Management System",
            regulatory_reference="Article 9",
        ),
        AssessmentStep(
            step_number=5,
            title="Human Oversight Design",
            regulatory_reference="Article 14",
        ),
        AssessmentStep(
            step_number=6,
            title="Accuracy, Robustness and Cybersecurity",
            regulatory_reference="Article 15",
        ),
        AssessmentStep(
            step_number=7,
            title="EU Declaration of Conformity",
            regulatory_reference="Article 47",
        ),
    ]


@dataclass
class ConformityAssessment:
    """Annex VI internal control conformity assessment pack.

    Create an instance, populate each step's ``evidence`` and ``findings``
    via ``get_step()``, then call ``run_all()`` to execute automated checks
    and ``flag_incomplete()`` to identify what still needs human input.

    Examples
    --------
    Basic usage:

    >>> ca = ConformityAssessment(
    ...     model_name="Life Propensity Model v2",
    ...     assessor_name="Model Governance Team",
    ...     assessment_date="2025-11-01",
    ... )
    >>> step1 = ca.get_step(1)
    >>> step1.evidence = "Classification performed using AIActClassifier: HIGH_RISK."
    >>> step1.status = StepStatus.COMPLETE
    >>> gaps = ca.flag_incomplete()
    """

    model_name: str = ""
    assessor_name: str = ""
    assessment_date: str = ""
    steps: list[AssessmentStep] = field(default_factory=_default_steps)

    # Optional: link to Article13Document for automated step checks
    _article13_doc: Any = field(default=None, repr=False, compare=False)

    def get_step(self, step_number: int) -> AssessmentStep:
        """Retrieve an assessment step by its number (1–7).

        Parameters
        ----------
        step_number:
            Integer between 1 and 7 inclusive.

        Raises
        ------
        ValueError
            If the step number is out of range.
        """
        if not 1 <= step_number <= 7:
            raise ValueError(f"step_number must be between 1 and 7, got {step_number}.")
        return self.steps[step_number - 1]

    def attach_article13(self, doc: Any) -> None:
        """Attach an ``Article13Document`` to enable automated checks.

        When a document is attached, ``run_all()`` will use its content to
        auto-populate findings for steps 3, 5, and 6.

        Parameters
        ----------
        doc:
            An ``Article13Document`` instance.
        """
        self._article13_doc = doc

    def run_all(self) -> dict[str, list[str]]:
        """Execute automated checks and return a dict of step findings.

        Automated checks:
        - Step 3 (Technical documentation): verifies Article13Document gaps.
        - Step 5 (Human oversight): checks override_procedure and anomaly_thresholds.
        - Step 6 (Accuracy/robustness): checks accuracy_metrics are populated.

        Steps 1, 2, 4, and 7 require manual evidence entry and are only checked
        for completeness (evidence field non-empty).

        Returns
        -------
        dict[str, list[str]]
            Maps step title to list of auto-generated findings.
        """
        all_findings: dict[str, list[str]] = {}

        if self._article13_doc is not None:
            doc = self._article13_doc

            # Step 3: Technical documentation completeness
            s3 = self.get_step(3)
            gaps = doc.flag_gaps()
            if gaps:
                s3.findings.extend(gaps)
                s3.status = StepStatus.INCOMPLETE
            else:
                if not s3.evidence:
                    s3.evidence = (
                        "Article 13 document populated; no mandatory field gaps detected."
                    )
                if s3.status == StepStatus.INCOMPLETE:
                    s3.status = StepStatus.COMPLETE
            all_findings[s3.title] = list(s3.findings)

            # Step 5: Human oversight
            s5 = self.get_step(5)
            oversight_findings: list[str] = []
            if not doc.human_oversight_measures:
                oversight_findings.append(
                    "No human oversight measures documented (Article 14(4)(a))."
                )
            if not doc.override_procedure:
                oversight_findings.append(
                    "Override procedure not documented (Article 14(4)(b))."
                )
            if not doc.anomaly_thresholds:
                oversight_findings.append(
                    "No anomaly thresholds defined for escalation triggers "
                    "(Article 14(4)(c))."
                )
            s5.findings.extend(oversight_findings)
            if not oversight_findings and s5.status == StepStatus.INCOMPLETE:
                if s5.evidence:
                    s5.status = StepStatus.COMPLETE
            elif oversight_findings:
                s5.status = StepStatus.INCOMPLETE
            all_findings[s5.title] = list(s5.findings)

            # Step 6: Accuracy and robustness
            s6 = self.get_step(6)
            accuracy_findings: list[str] = []
            if not doc.accuracy_metrics:
                accuracy_findings.append(
                    "No accuracy metrics recorded — required by Article 15(1)."
                )
            if not doc.monitoring_metrics:
                accuracy_findings.append(
                    "No monitoring metrics defined — required for ongoing robustness "
                    "under Article 15(3)."
                )
            s6.findings.extend(accuracy_findings)
            if not accuracy_findings and s6.status == StepStatus.INCOMPLETE:
                if s6.evidence:
                    s6.status = StepStatus.COMPLETE
            elif accuracy_findings:
                s6.status = StepStatus.INCOMPLETE
            all_findings[s6.title] = list(s6.findings)

        # Check manual steps for evidence presence
        for step_num in (1, 2, 4, 7):
            step = self.get_step(step_num)
            if step.status == StepStatus.INCOMPLETE and not step.evidence:
                step.findings.append(
                    f"No evidence recorded for step {step_num} — manual input required."
                )
            all_findings[step.title] = list(step.findings)

        return all_findings

    def flag_incomplete(self) -> list[str]:
        """Return a list of steps that require further action.

        Returns
        -------
        list[str]
            Each entry names the incomplete step and its regulatory reference.
        """
        return [
            f"Step {s.step_number} ({s.title}, {s.regulatory_reference}): incomplete."
            for s in self.steps
            if s.status == StepStatus.INCOMPLETE
        ]

    def overall_status(self) -> StepStatus:
        """Return the overall assessment status.

        Returns COMPLETE only if all seven steps are complete or not applicable.
        Returns INCOMPLETE if any step is incomplete.
        """
        if all(s.is_complete() for s in self.steps):
            return StepStatus.COMPLETE
        return StepStatus.INCOMPLETE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the assessment."""
        return {
            "conformity_assessment": {
                "model_name": self.model_name,
                "assessor_name": self.assessor_name,
                "assessment_date": self.assessment_date,
                "overall_status": self.overall_status().value,
                "steps": [s.to_dict() for s in self.steps],
            }
        }

    def to_markdown(self) -> str:
        """Render the conformity assessment pack as Markdown."""
        from .renderer import render_conformity_markdown

        return render_conformity_markdown(self)
