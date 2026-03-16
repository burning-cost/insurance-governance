"""
Model card schema for UK insurance pricing models.

The model card is the anchor document for the validation report. It
captures what the model is, who built it, what it is allowed to do, and
what its known limitations are. This aligns with SS1/23 Principle 3 (as
applied by analogy to insurance MRM) and the FCA's requirement for
Consumer Duty fair value documentation.

Usage
-----
    from insurance_governance.validation import ModelCard

    card = ModelCard(
        model_name="Motor Third-Party Property Damage Frequency",
        version="2.1.0",
        purpose="Estimate expected claim frequency for private motor policies",
        intended_use="Underwriting pricing, not claims reserving",
        developer="Pricing Team",
        development_date="2024-09-01",
        limitations="Out-of-sample performance degrades for vehicles >10 years old",
        materiality_tier=2,
        approved_by=["Chief Actuary", "Model Risk Committee"],
        variables=["vehicle_age", "driver_age", "annual_mileage", "region"],
        target_variable="claim_count",
        model_type="GLM",
        distribution_family="Poisson",
    )

New simplified API (also accepted)
-----------------------------------
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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional, Union


# ---------------------------------------------------------------------------
# Validation error — raised on bad construction.  Replaces pydantic.ValidationError
# so calling code that catches ValidationError from this module still works.
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a ModelCard is constructed with invalid field values."""


# ---------------------------------------------------------------------------
# Valid literals for model_type
# ---------------------------------------------------------------------------

_VALID_MODEL_TYPES = frozenset({"GLM", "GBM", "GAM", "Neural Network", "Ensemble", "Other"})


@dataclass
class ModelCard:
    """
    Structured metadata for an insurance pricing model.

    All fields are validated on construction. Required fields must be
    non-empty strings or non-empty lists. This forces the team to
    document their model before running validation — not as an afterthought.

    Supports two field naming conventions:
    - Legacy: model_name, developer, variables, approved_by
    - Simplified: name, owner, features (with automatic mapping)

    Raises
    ------
    ValidationError
        If any required field is empty, out of range, or otherwise invalid.
    """

    # Primary name field (legacy)
    model_name: Optional[str] = field(
        default=None,
        metadata={"description": "Full descriptive name of the model, e.g. 'Motor TPPD Frequency v2.1'"},
    )
    # Simplified API alias
    name: Optional[str] = field(
        default=None,
        metadata={"description": "Model name (simplified API alias for model_name)."},
    )
    version: str = field(
        default="",
        metadata={"description": "Version string, e.g. '2.1.0'"},
    )
    purpose: str = field(
        default="",
        metadata={"description": "One or two sentences stating what the model does."},
    )
    intended_use: Optional[str] = field(
        default=None,
        metadata={"description": "Scope of permitted use."},
    )
    # Legacy developer field
    developer: Optional[str] = field(
        default=None,
        metadata={"description": "Name of team or individual who built the model."},
    )
    # Simplified API alias
    owner: Optional[str] = field(
        default=None,
        metadata={"description": "Model owner / developer (simplified API alias for developer)."},
    )
    development_date: Optional[Union[date, str]] = field(
        default=None,
        metadata={"description": "Date the model was signed off for production use."},
    )
    # Limitations: accepts string or list
    limitations: Optional[Union[str, list]] = field(
        default=None,
        metadata={"description": "Known limitations, failure modes, or out-of-scope populations."},
    )
    materiality_tier: Optional[int] = field(
        default=None,
        metadata={"description": "Model risk tier per internal classification (1=highest risk, 3=lowest)."},
    )
    approved_by: Optional[list] = field(
        default=None,
        metadata={"description": "List of named approvers with title."},
    )
    # Legacy variables field
    variables: Optional[list] = field(
        default=None,
        metadata={"description": "List of model input variables (features) used in production."},
    )
    # Simplified API alias
    features: Optional[list] = field(
        default=None,
        metadata={"description": "Feature list (simplified API alias for variables)."},
    )
    # Legacy target_variable field
    target_variable: Optional[str] = field(
        default=None,
        metadata={"description": "Name of the response variable, e.g. 'claim_count'."},
    )
    # Simplified API alias
    target: Optional[str] = field(
        default=None,
        metadata={"description": "Target variable (simplified API alias for target_variable)."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"description": "High-level model family: GLM, GBM, GAM, Neural Network, Ensemble, Other."},
    )
    distribution_family: Optional[str] = field(
        default=None,
        metadata={"description": "Statistical distribution assumed for the response, e.g. 'Poisson'."},
    )
    # Simplified API: methodology replaces distribution_family when provided
    methodology: Optional[str] = field(
        default=None,
        metadata={"description": "Model methodology description (simplified API)."},
    )

    # Optional but recommended fields
    validation_date: Optional[Union[date, str]] = field(
        default=None,
        metadata={"description": "Date of this validation run."},
    )
    validator_name: Optional[str] = field(
        default=None,
        metadata={"description": "Name of the independent validator."},
    )
    model_description: Optional[str] = field(
        default=None,
        metadata={"description": "Extended description for the report narrative."},
    )
    alternatives_considered: Optional[str] = field(
        default=None,
        metadata={"description": "Alternative approaches evaluated during development."},
    )
    monitoring_frequency: Optional[str] = field(
        default=None,
        metadata={"description": "How often ongoing model monitoring is performed, e.g. 'Quarterly'."},
    )
    outstanding_issues: Optional[list] = field(
        default=None,
        metadata={"description": "Known outstanding issues requiring resolution."},
    )
    monitoring_owner: Optional[str] = field(
        default=None,
        metadata={"description": "Named owner responsible for ongoing model monitoring."},
    )
    monitoring_triggers: Optional[dict] = field(
        default=None,
        metadata={"description": "Metric names and threshold values that trigger a model review."},
    )

    def __post_init__(self) -> None:
        self._normalise_aliases()
        self._validate()

    # ------------------------------------------------------------------
    # Internal: alias resolution and validation
    # ------------------------------------------------------------------

    def _normalise_aliases(self) -> None:
        """Resolve simplified API aliases to legacy field names."""
        # model_name / name
        if self.model_name is None and self.name is not None:
            self.model_name = self.name

        # developer / owner
        if self.developer is None and self.owner is not None:
            self.developer = self.owner

        # variables / features
        if self.variables is None and self.features is not None:
            self.variables = self.features

        # target_variable / target
        if self.target_variable is None and self.target is not None:
            self.target_variable = self.target

        # distribution_family / methodology
        if self.distribution_family is None and self.methodology is not None:
            self.distribution_family = self.methodology

        # Normalise limitations list -> string
        if isinstance(self.limitations, list):
            self.limitations = "; ".join(self.limitations)

        # Coerce date strings to date objects
        if isinstance(self.development_date, str) and self.development_date:
            try:
                self.development_date = date.fromisoformat(self.development_date)
            except ValueError as exc:
                raise ValidationError(
                    f"development_date must be an ISO date string (YYYY-MM-DD), got {self.development_date!r}"
                ) from exc

        if isinstance(self.validation_date, str) and self.validation_date:
            try:
                self.validation_date = date.fromisoformat(self.validation_date)
            except ValueError as exc:
                raise ValidationError(
                    f"validation_date must be an ISO date string (YYYY-MM-DD), got {self.validation_date!r}"
                ) from exc

    def _validate(self) -> None:
        """Run all validation checks and raise ValidationError on failure."""
        if self.model_name is None:
            raise ValidationError("Either 'model_name' or 'name' must be provided.")
        if not self.model_name.strip():
            raise ValidationError("model_name must be a non-empty string.")

        if len(self.version) < 1:
            raise ValidationError("version must be a non-empty string.")

        if len(self.purpose) < 5:
            raise ValidationError(
                f"purpose must be at least 5 characters, got {len(self.purpose)}."
            )

        if self.materiality_tier is not None:
            if self.materiality_tier < 1 or self.materiality_tier > 3:
                raise ValidationError(
                    f"materiality_tier must be between 1 and 3, got {self.materiality_tier}."
                )

        if self.model_type is not None and self.model_type not in _VALID_MODEL_TYPES:
            raise ValidationError(
                f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}, got {self.model_type!r}."
            )

        if self.approved_by is not None:
            for entry in self.approved_by:
                if not str(entry).strip():
                    raise ValidationError("Each entry in approved_by must be a non-empty string.")

        if self.variables is not None:
            for var in self.variables:
                if not str(var).strip():
                    raise ValidationError("Each variable name must be a non-empty string.")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_effective_model_name(self) -> str:
        return self.model_name or self.name or "Unknown"

    def get_effective_developer(self) -> str:
        return self.developer or self.owner or "Not specified"

    def get_effective_variables(self) -> list:
        return self.variables or self.features or []

    def get_effective_target(self) -> str:
        return self.target_variable or self.target or "Not specified"

    def get_effective_distribution(self) -> str:
        return self.distribution_family or self.methodology or "Not specified"

    def get_effective_limitations(self) -> str:
        if isinstance(self.limitations, list):
            return "; ".join(self.limitations)
        return self.limitations or "None documented"

    def summary(self) -> dict:
        """Return a flat dict suitable for the report summary table."""
        return {
            "Model name": self.get_effective_model_name(),
            "Version": self.version,
            "Model type": self.model_type or "Not specified",
            "Distribution": self.get_effective_distribution(),
            "Developer": self.get_effective_developer(),
            "Development date": str(self.development_date) if self.development_date else "Not specified",
            "Materiality tier": self.materiality_tier if self.materiality_tier is not None else "Not specified",
            "Approved by": ", ".join(self.approved_by) if self.approved_by else "Pending",
            "Target variable": self.get_effective_target(),
            "Number of variables": len(self.get_effective_variables()),
        }
