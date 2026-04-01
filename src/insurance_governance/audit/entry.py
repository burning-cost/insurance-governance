"""ExplainabilityAuditEntry — immutable record of a single model prediction event.

Every time a pricing model makes a decision that affects a customer premium, we
need a tamper-evident record of what happened: what the model saw, what it said,
whether a human overrode it, and who signed off. This module provides that record.

Design notes:
- Dataclass, not Pydantic. The rest of this package is dependency-light.
- SHA-256 hash over the serialised entry for immutability verification. The hash
  excludes the ``entry_hash`` field itself (obviously), and is computed over a
  canonical JSON representation to avoid platform-specific encoding differences.
- ``decision_basis`` is an explicit controlled vocabulary. "model_output" means
  the premium was taken directly from the model. "human_override" means a
  reviewer changed the output. "rule_fallback" means a deterministic rule took
  over (e.g. minimum premium, underwriting exclusion).
- ``reviewer_id`` should reference the SM&CR Controlled Function holder
  responsible for the decision, not just a username.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


DECISION_BASIS_VALUES = frozenset({"model_output", "human_override", "rule_fallback"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class ExplainabilityAuditEntry:
    """A single model prediction event, immutably recorded for regulatory audit.

    This record captures everything needed to reconstruct the reasoning behind
    a pricing decision: the inputs the model saw, the SHAP attributions, the
    raw model output, the final charged premium (which may differ due to
    rounding or commercial loading), and any human review or override.

    The ``entry_hash`` is a SHA-256 digest of the canonical serialisation of
    all other fields. Call :meth:`verify_integrity` after loading from storage
    to confirm the record has not been tampered with.

    Args:
        entry_id: UUID4 string identifying this record. Auto-generated if not
            supplied.
        model_id: Foreign key to the MRMModelCard for the model that produced
            this prediction.
        model_version: Semantic version of the model at prediction time.
        timestamp_utc: ISO 8601 UTC timestamp of the prediction event. Defaults
            to the current time.
        session_id: Optional identifier for the containing batch run or API
            request. Useful for grouping entries from a pricing refresh.
        input_features: Dict of raw feature values passed to the model. Keys
            should match the model's ``rating_factors``.
        feature_importances: Dict of signed SHAP values keyed by feature name.
            Positive means the feature increased the prediction; negative means
            it decreased it.
        prediction: Raw model output (e.g. log-scale linear predictor before
            applying the base rate, or the final relativised premium before
            any commercial loading).
        final_premium: The premium actually charged to the customer. May differ
            from ``prediction`` due to rounding, minimum premium rules, or
            commercial loading. Nullable if the policy was declined.
        human_reviewed: Whether a human reviewer examined this decision.
        reviewer_id: The SM&CR Controlled Function holder who reviewed this
            decision. Should be a stable identifier, not just a display name.
        override_applied: Whether the human reviewer changed the model output.
        override_reason: Free-text reason for the override. Required when
            ``override_applied`` is True.
        decision_basis: How the final premium was determined. One of
            ``'model_output'``, ``'human_override'``, or ``'rule_fallback'``.
        entry_hash: SHA-256 hash of the serialised entry, excluding this field.
            Auto-computed on creation; do not set manually.
    """

    # Identity
    model_id: str
    model_version: str

    # Decision content
    input_features: dict[str, Any]
    feature_importances: dict[str, float]
    prediction: float

    # Optional fields with defaults
    entry_id: str = field(default_factory=_new_uuid)
    timestamp_utc: str = field(default_factory=_utc_now_iso)
    session_id: Optional[str] = None
    final_premium: Optional[float] = None
    human_reviewed: bool = False
    reviewer_id: Optional[str] = None
    override_applied: bool = False
    override_reason: Optional[str] = None
    decision_basis: str = "model_output"

    # Computed last
    entry_hash: str = field(default="", init=True)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")
        if self.decision_basis not in DECISION_BASIS_VALUES:
            raise ValueError(
                f"decision_basis must be one of {sorted(DECISION_BASIS_VALUES)}, "
                f"got {self.decision_basis!r}"
            )
        if self.override_applied and not self.override_reason:
            raise ValueError(
                "override_reason must be provided when override_applied is True"
            )
        # Compute hash if not already set (e.g. when constructing fresh)
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _canonical_dict(self) -> dict[str, Any]:
        """Return a dict of all fields except entry_hash, with stable ordering."""
        return {
            "entry_id": self.entry_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp_utc": self.timestamp_utc,
            "session_id": self.session_id,
            "input_features": self.input_features,
            "feature_importances": self.feature_importances,
            "prediction": self.prediction,
            "final_premium": self.final_premium,
            "human_reviewed": self.human_reviewed,
            "reviewer_id": self.reviewer_id,
            "override_applied": self.override_applied,
            "override_reason": self.override_reason,
            "decision_basis": self.decision_basis,
        }

    def _compute_hash(self) -> str:
        """Compute SHA-256 over the canonical JSON representation of this entry."""
        canonical = json.dumps(
            self._canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Check that the entry has not been modified since it was created.

        Returns True if the stored hash matches a freshly-computed hash over
        the current field values. Returns False otherwise.

        A False result indicates the record has been tampered with or was
        corrupted in storage.
        """
        return self.entry_hash == self._compute_hash()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding.

        The output includes all fields including ``entry_hash``.
        """
        d = self._canonical_dict()
        d["entry_hash"] = self.entry_hash
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExplainabilityAuditEntry":
        """Deserialise from a plain dict (e.g., loaded from a JSONL log).

        The stored hash is preserved; call :meth:`verify_integrity` separately
        if you want to check the record has not been tampered with.
        """
        entry = cls(
            entry_id=d.get("entry_id", str(uuid.uuid4())),
            model_id=d["model_id"],
            model_version=d["model_version"],
            timestamp_utc=d.get("timestamp_utc", _utc_now_iso()),
            session_id=d.get("session_id"),
            input_features=d.get("input_features", {}),
            feature_importances=d.get("feature_importances", {}),
            prediction=float(d["prediction"]),
            final_premium=d.get("final_premium"),
            human_reviewed=d.get("human_reviewed", False),
            reviewer_id=d.get("reviewer_id"),
            override_applied=d.get("override_applied", False),
            override_reason=d.get("override_reason"),
            decision_basis=d.get("decision_basis", "model_output"),
            entry_hash=d.get("entry_hash", ""),
        )
        # Restore stored hash rather than recomputing, so verify_integrity works
        if d.get("entry_hash"):
            object.__setattr__(entry, "entry_hash", d["entry_hash"])
        return entry
