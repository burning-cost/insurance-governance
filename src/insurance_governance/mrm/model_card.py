"""ModelCard and Assumption dataclasses for insurance pricing model governance.

These are the structured metadata containers for a pricing model. The intent is
to capture everything a Model Risk Committee needs to classify a model, assign
a validation schedule, and understand what the model is for and what it is not
for.

Design note: we use dataclasses with default_factory rather than Pydantic to
keep this library dependency-free. All fields are plain Python types so the
card serialises cleanly to JSON with no Pydantic machinery required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Valid enumerations (stored as strings — avoids enum import overhead)
# ---------------------------------------------------------------------------

MODEL_CLASSES = frozenset({"pricing", "reserving", "capital", "underwriting", "claims"})
DISTRIBUTION_FAMILIES = frozenset({
    "Poisson", "Gamma", "Tweedie", "Gaussian", "Binomial", "LogNormal",
    "NegativeBinomial", "InverseGaussian", "other",
})
CHAMPION_STATUSES = frozenset({"champion", "challenger", "shadow", "retired", "development"})
RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})


def _today_iso() -> str:
    return date.today().isoformat()


@dataclass
class Assumption:
    """A single material assumption in a pricing model.

    Each assumption should have an explicit risk rating and a documented
    mitigation. Listing assumptions without risk ratings is compliance theatre.

    Args:
        description: Plain-language description of the assumption.
        risk: Risk level if the assumption is violated — ``'LOW'``,
            ``'MEDIUM'``, or ``'HIGH'``.
        mitigation: What monitoring or control reduces the risk. Can be
            empty for LOW-risk assumptions but should not be for MEDIUM/HIGH.
        rationale: Optional explanation of why this risk level was assigned.
    """

    description: str
    risk: str = "LOW"
    mitigation: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.risk not in RISK_LEVELS:
            raise ValueError(
                f"risk must be one of {sorted(RISK_LEVELS)}, got {self.risk!r}"
            )

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Assumption":
        """Deserialise from a plain dict."""
        return cls(
            description=d["description"],
            risk=d.get("risk", "LOW"),
            mitigation=d.get("mitigation", ""),
            rationale=d.get("rationale", ""),
        )


@dataclass
class Limitation:
    """A known limitation of the model with impact assessment.

    Args:
        description: What the limitation is.
        impact: What happens when the limitation is breached.
        population_at_risk: Which policyholders or segments are most affected.
        monitoring_flag: Whether this limitation triggers a specific monitoring
            check.
    """

    description: str
    impact: str = ""
    population_at_risk: str = ""
    monitoring_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Limitation":
        """Deserialise from a plain dict."""
        return cls(
            description=d["description"],
            impact=d.get("impact", ""),
            population_at_risk=d.get("population_at_risk", ""),
            monitoring_flag=d.get("monitoring_flag", False),
        )


@dataclass
class ModelCard:
    """Structured metadata for an insurance pricing model.

    This is the governance record for a model. It covers identity,
    classification, intended use, assumptions, limitations, sign-off, and
    monitoring plan — the fields a Model Risk Committee and a PRA supervisor
    would expect to see.

    Mandatory fields are those without a default. Everything else has a
    sensible default (empty string, empty list, or None) so you can build
    up a card incrementally.

    Args:
        model_id: Unique identifier — used as the primary key in the inventory.
            Use a stable slug like ``'motor-freq-tppd-v2'``, not a UUID.
        model_name: Full human-readable name.
        version: Semantic version string (``'2.1.0'``).
        model_class: One of ``'pricing'``, ``'reserving'``, ``'capital'``,
            ``'underwriting'``, ``'claims'``.
        intended_use: Explicit permitted-use statement. Be specific.
        not_intended_for: Explicit list of prohibited uses.
        target_variable: What the model predicts (e.g. ``'claim_count'``).
        distribution_family: Statistical distribution assumed for the target.
        model_type: Architecture description (e.g. ``'GLM'``, ``'CatBoost'``).
        rating_factors: List of features used in the model.
        training_data_period: Tuple of ISO date strings ``(start, end)``.
        development_date: ISO date string when the model was finalised.
        developer: Team or individual who built the model.
        champion_challenger_status: One of ``'champion'``, ``'challenger'``,
            ``'shadow'``, ``'retired'``, ``'development'``.
        assumptions: List of :class:`Assumption` objects.
        limitations: List of :class:`Limitation` objects. You can also pass
            plain strings — they will be converted to ``Limitation`` objects.
        outstanding_issues: Free-text issues not yet resolved.
        portfolio_scope: Which products or entities this model applies to.
        geographic_scope: Geographical coverage.
        customer_facing: Whether the model directly sets customer premiums.
        regulatory_use: Whether used in Solvency II or regulatory capital.
        gwp_impacted: Gross written premium (£) on which this model has
            material influence.
        materiality_tier: Risk tier (1, 2, or 3). Set by
            :class:`~insurance_mrm.scorer.RiskTierScorer` — do not assert
            without calculating.
        tier_rationale: Verbose rationale for the tier assignment.
        approved_by: List of approvers (name and/or role).
        approval_date: ISO date string of most recent approval.
        approval_conditions: Any conditions attached to the approval.
        next_review_date: ISO date string when revalidation is due.
        monitoring_owner: Named individual responsible for ongoing monitoring.
        monitoring_frequency: How often monitoring runs (``'Quarterly'``, etc.).
        monitoring_triggers: Dict mapping metric names to threshold values that
            trigger an ad-hoc review.
        trigger_actions: Dict mapping trigger conditions to required actions.
        last_monitoring_run: ISO date string of the most recent monitoring run.
        last_validation_run: ISO date string of the most recent validation.
        last_validation_run_id: UUID from the insurance-validation JSON output.
        overall_rag: Overall RAG status from most recent validation
            (``'GREEN'``, ``'AMBER'``, ``'RED'``).
        created_at: ISO datetime when this card was created (auto-set).
        updated_at: ISO datetime when this card was last modified (auto-set on
            ``to_dict()``).
    """

    # --- Identity (required) ---
    model_id: str
    model_name: str
    version: str

    # --- Classification ---
    model_class: str = "pricing"
    intended_use: str = ""
    not_intended_for: list[str] = field(default_factory=list)
    target_variable: str = ""
    distribution_family: str = ""
    model_type: str = ""
    rating_factors: list[str] = field(default_factory=list)
    training_data_period: tuple[str, str] = ("", "")
    development_date: str = ""
    developer: str = ""
    champion_challenger_status: str = "development"

    # --- Assumptions & limitations ---
    assumptions: list[Assumption] = field(default_factory=list)
    limitations: list[Any] = field(default_factory=list)  # Limitation or str
    outstanding_issues: list[str] = field(default_factory=list)

    # --- Scope ---
    portfolio_scope: str = ""
    geographic_scope: str = ""
    customer_facing: bool = True
    regulatory_use: bool = False
    gwp_impacted: float = 0.0

    # --- Risk tier ---
    materiality_tier: Optional[int] = None
    tier_rationale: str = ""

    # --- Governance ---
    approved_by: list[str] = field(default_factory=list)
    approval_date: str = ""
    approval_conditions: str = ""
    next_review_date: str = ""

    # --- Monitoring ---
    monitoring_owner: str = ""
    monitoring_frequency: str = "Quarterly"
    monitoring_triggers: dict[str, float] = field(default_factory=dict)
    trigger_actions: dict[str, str] = field(default_factory=dict)
    last_monitoring_run: str = ""
    last_validation_run: str = ""
    last_validation_run_id: str = ""
    overall_rag: str = ""

    # --- Audit ---
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")
        if self.model_class not in MODEL_CLASSES:
            raise ValueError(
                f"model_class must be one of {sorted(MODEL_CLASSES)}, "
                f"got {self.model_class!r}"
            )
        if self.champion_challenger_status not in CHAMPION_STATUSES:
            raise ValueError(
                f"champion_challenger_status must be one of "
                f"{sorted(CHAMPION_STATUSES)}, "
                f"got {self.champion_challenger_status!r}"
            )
        if self.materiality_tier is not None and self.materiality_tier not in (1, 2, 3):
            raise ValueError("materiality_tier must be 1, 2, or 3")
        if self.overall_rag and self.overall_rag not in ("GREEN", "AMBER", "RED"):
            raise ValueError(
                "overall_rag must be 'GREEN', 'AMBER', or 'RED'"
            )
        # Coerce plain-string limitations to Limitation objects
        normalised = []
        for lim in self.limitations:
            if isinstance(lim, str):
                normalised.append(Limitation(description=lim))
            elif isinstance(lim, dict):
                normalised.append(Limitation.from_dict(lim))
            elif isinstance(lim, Limitation):
                normalised.append(lim)
            else:
                raise TypeError(
                    f"limitations entries must be str, dict, or Limitation; "
                    f"got {type(lim)}"
                )
        self.limitations = normalised
        # Coerce plain-dict assumptions
        normalised_assumptions = []
        for assumption in self.assumptions:
            if isinstance(assumption, dict):
                normalised_assumptions.append(Assumption.from_dict(assumption))
            elif isinstance(assumption, Assumption):
                normalised_assumptions.append(assumption)
            else:
                raise TypeError(
                    f"assumptions entries must be dict or Assumption; "
                    f"got {type(assumption)}"
                )
        self.assumptions = normalised_assumptions

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def high_risk_assumptions(self) -> list[Assumption]:
        """Return assumptions rated HIGH."""
        return [a for a in self.assumptions if a.risk == "HIGH"]

    @property
    def medium_risk_assumptions(self) -> list[Assumption]:
        """Return assumptions rated MEDIUM."""
        return [a for a in self.assumptions if a.risk == "MEDIUM"]

    @property
    def is_approved(self) -> bool:
        """True if the card has at least one approver and an approval date."""
        return bool(self.approved_by and self.approval_date)

    def assumption_summary(self) -> dict[str, int]:
        """Count assumptions by risk level."""
        counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for a in self.assumptions:
            counts[a.risk] = counts.get(a.risk, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the card to a plain dict suitable for JSON encoding.

        All nested objects (Assumption, Limitation) are also serialised to
        dicts. The ``updated_at`` field is refreshed to the current UTC time.
        """
        self.updated_at = datetime.now(timezone.utc).isoformat()
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_class": self.model_class,
            "intended_use": self.intended_use,
            "not_intended_for": self.not_intended_for,
            "target_variable": self.target_variable,
            "distribution_family": self.distribution_family,
            "model_type": self.model_type,
            "rating_factors": self.rating_factors,
            "training_data_period": list(self.training_data_period),
            "development_date": self.development_date,
            "developer": self.developer,
            "champion_challenger_status": self.champion_challenger_status,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "limitations": [lim.to_dict() for lim in self.limitations],
            "outstanding_issues": self.outstanding_issues,
            "portfolio_scope": self.portfolio_scope,
            "geographic_scope": self.geographic_scope,
            "customer_facing": self.customer_facing,
            "regulatory_use": self.regulatory_use,
            "gwp_impacted": self.gwp_impacted,
            "materiality_tier": self.materiality_tier,
            "tier_rationale": self.tier_rationale,
            "approved_by": self.approved_by,
            "approval_date": self.approval_date,
            "approval_conditions": self.approval_conditions,
            "next_review_date": self.next_review_date,
            "monitoring_owner": self.monitoring_owner,
            "monitoring_frequency": self.monitoring_frequency,
            "monitoring_triggers": self.monitoring_triggers,
            "trigger_actions": self.trigger_actions,
            "last_monitoring_run": self.last_monitoring_run,
            "last_validation_run": self.last_validation_run,
            "last_validation_run_id": self.last_validation_run_id,
            "overall_rag": self.overall_rag,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelCard":
        """Deserialise from a plain dict (e.g., loaded from JSON)."""
        assumptions = [
            Assumption.from_dict(a) if isinstance(a, dict) else a
            for a in d.get("assumptions", [])
        ]
        limitations = [
            Limitation.from_dict(lim) if isinstance(lim, dict) else lim
            for lim in d.get("limitations", [])
        ]
        training_period = d.get("training_data_period", ("", ""))
        if isinstance(training_period, list):
            training_period = tuple(training_period)

        return cls(
            model_id=d["model_id"],
            model_name=d["model_name"],
            version=d["version"],
            model_class=d.get("model_class", "pricing"),
            intended_use=d.get("intended_use", ""),
            not_intended_for=d.get("not_intended_for", []),
            target_variable=d.get("target_variable", ""),
            distribution_family=d.get("distribution_family", ""),
            model_type=d.get("model_type", ""),
            rating_factors=d.get("rating_factors", []),
            training_data_period=training_period,
            development_date=d.get("development_date", ""),
            developer=d.get("developer", ""),
            champion_challenger_status=d.get("champion_challenger_status", "development"),
            assumptions=assumptions,
            limitations=limitations,
            outstanding_issues=d.get("outstanding_issues", []),
            portfolio_scope=d.get("portfolio_scope", ""),
            geographic_scope=d.get("geographic_scope", ""),
            customer_facing=d.get("customer_facing", True),
            regulatory_use=d.get("regulatory_use", False),
            gwp_impacted=d.get("gwp_impacted", 0.0),
            materiality_tier=d.get("materiality_tier"),
            tier_rationale=d.get("tier_rationale", ""),
            approved_by=d.get("approved_by", []),
            approval_date=d.get("approval_date", ""),
            approval_conditions=d.get("approval_conditions", ""),
            next_review_date=d.get("next_review_date", ""),
            monitoring_owner=d.get("monitoring_owner", ""),
            monitoring_frequency=d.get("monitoring_frequency", "Quarterly"),
            monitoring_triggers=d.get("monitoring_triggers", {}),
            trigger_actions=d.get("trigger_actions", {}),
            last_monitoring_run=d.get("last_monitoring_run", ""),
            last_validation_run=d.get("last_validation_run", ""),
            last_validation_run_id=d.get("last_validation_run_id", ""),
            overall_rag=d.get("overall_rag", ""),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_json(cls, s: str) -> "ModelCard":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))
