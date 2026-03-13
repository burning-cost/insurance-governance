"""ModelInventory: persistent JSON-file registry of insurance pricing models.

The inventory is the source of truth for which models exist, what tier they're
in, when they were last validated, and when the next review is due. It writes
to a plain JSON file so there's no database to set up.

Design note: we deliberately chose a JSON file backend rather than SQLite.
SQLite is fine but introduces a C extension dependency and the concurrency
semantics are harder to explain to a pricing actuary. A JSON file is readable
in any text editor and auditable in git. The trade-off is that it doesn't scale
past a few thousand models — but a mid-tier insurer has maybe 50-100 production
models, so this is the right trade-off.

Thread safety: the inventory uses a simple load-modify-save pattern. It is not
thread-safe. For production use with concurrent writers, put the inventory file
on a git-backed store and treat updates as commits.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from .model_card import ModelCard
from .scorer import TierResult


# ---------------------------------------------------------------------------
# Registry entry structure (stored in the JSON file)
# ---------------------------------------------------------------------------

class _Encoder(json.JSONEncoder):
    """JSON encoder that handles date/datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _load_registry(path: str) -> dict[str, Any]:
    """Load the registry JSON file. Returns empty structure if file absent."""
    if not os.path.exists(path):
        return {"models": {}, "events": []}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Ensure the expected top-level keys exist (backwards compat)
    data.setdefault("models", {})
    data.setdefault("events", [])
    return data


def _save_registry(path: str, data: dict[str, Any]) -> None:
    """Write registry to disk atomically (write to tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, cls=_Encoder)
    os.replace(tmp, path)


class ModelInventory:
    """Persistent registry of pricing model governance records.

    Each entry combines a :class:`~insurance_mrm.model_card.ModelCard` with
    its :class:`~insurance_mrm.scorer.TierResult` and a lightweight record of
    validation history.

    The inventory file is a JSON document. You can put it in version control.

    Args:
        path: File path for the JSON registry. Created on first
            :meth:`register` call if it doesn't exist.

    Examples::

        inventory = ModelInventory('mrm_registry.json')
        inventory.register(card, tier_result)
        df = inventory.list()                 # dict list of all models
        due = inventory.due_for_review(30)    # models due in 30 days
    """

    def __init__(self, path: str) -> None:
        self.path = path

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(
        self, card: ModelCard, tier: Optional[TierResult] = None
    ) -> str:
        """Add or update a model in the inventory.

        If a model with the same ``model_id`` already exists, it is updated
        (not duplicated). Returns the ``model_id``.

        Args:
            card: The :class:`~insurance_mrm.model_card.ModelCard` to register.
            tier: Optional :class:`~insurance_mrm.scorer.TierResult`. If
                provided, the card's ``materiality_tier`` and ``tier_rationale``
                are updated before saving.
        """
        data = _load_registry(self.path)

        if tier is not None:
            card.materiality_tier = tier.tier
            card.tier_rationale = tier.rationale

        entry: dict[str, Any] = {
            "card": card.to_dict(),
            "tier_result": tier.to_dict() if tier is not None else None,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "validation_history": data["models"].get(
                card.model_id, {}
            ).get("validation_history", []),
        }
        data["models"][card.model_id] = entry
        _save_registry(self.path, data)
        return card.model_id

    def update_validation(
        self,
        model_id: str,
        validation_date: str,
        overall_rag: str,
        next_review_date: str,
        run_id: str = "",
        notes: str = "",
    ) -> None:
        """Record the outcome of a validation run for a model.

        Updates the card's ``last_validation_run``, ``overall_rag``, and
        ``next_review_date`` fields, and appends an entry to the validation
        history.

        Args:
            model_id: The model to update.
            validation_date: ISO date string of the validation.
            overall_rag: ``'GREEN'``, ``'AMBER'``, or ``'RED'``.
            next_review_date: ISO date string when revalidation is due.
            run_id: UUID from the insurance-validation JSON output.
            notes: Free-text notes on this validation.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        if overall_rag not in ("GREEN", "AMBER", "RED"):
            raise ValueError(
                f"overall_rag must be 'GREEN', 'AMBER', or 'RED'; "
                f"got {overall_rag!r}"
            )
        data = _load_registry(self.path)
        if model_id not in data["models"]:
            raise KeyError(
                f"Model '{model_id}' not found in inventory at {self.path!r}"
            )
        entry = data["models"][model_id]
        entry["card"]["last_validation_run"] = validation_date
        entry["card"]["last_validation_run_id"] = run_id
        entry["card"]["overall_rag"] = overall_rag
        entry["card"]["next_review_date"] = next_review_date
        entry["card"]["updated_at"] = datetime.now(timezone.utc).isoformat()

        history_record = {
            "validation_date": validation_date,
            "overall_rag": overall_rag,
            "next_review_date": next_review_date,
            "run_id": run_id,
            "notes": notes,
        }
        entry.setdefault("validation_history", []).append(history_record)

        _save_registry(self.path, data)

    def update_status(
        self,
        model_id: str,
        champion_challenger_status: str,
    ) -> None:
        """Update the deployment status of a model.

        Args:
            model_id: The model to update.
            champion_challenger_status: New status — ``'champion'``,
                ``'challenger'``, ``'shadow'``, ``'retired'``, or
                ``'development'``.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        from .model_card import CHAMPION_STATUSES
        if champion_challenger_status not in CHAMPION_STATUSES:
            raise ValueError(
                f"champion_challenger_status must be one of "
                f"{sorted(CHAMPION_STATUSES)}; got {champion_challenger_status!r}"
            )
        data = _load_registry(self.path)
        if model_id not in data["models"]:
            raise KeyError(
                f"Model '{model_id}' not found in inventory at {self.path!r}"
            )
        data["models"][model_id]["card"]["champion_challenger_status"] = (
            champion_challenger_status
        )
        data["models"][model_id]["card"]["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        _save_registry(self.path, data)

    def log_event(
        self,
        model_id: str,
        event_type: str,
        description: str,
        triggered_by: str = "",
    ) -> None:
        """Append an event to the audit log.

        Args:
            model_id: The model this event relates to.
            event_type: Short event type string, e.g.
                ``'monitoring_trigger'``, ``'status_change'``,
                ``'ad_hoc_review'``.
            description: Free-text description of the event.
            triggered_by: What caused the event (system name, process, person).

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        data = _load_registry(self.path)
        if model_id not in data["models"]:
            raise KeyError(
                f"Model '{model_id}' not found in inventory at {self.path!r}"
            )
        event = {
            "model_id": model_id,
            "event_type": event_type,
            "description": description,
            "triggered_by": triggered_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data.setdefault("events", []).append(event)
        _save_registry(self.path, data)

    def remove(self, model_id: str) -> None:
        """Remove a model from the inventory.

        Args:
            model_id: The model to remove.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        data = _load_registry(self.path)
        if model_id not in data["models"]:
            raise KeyError(
                f"Model '{model_id}' not found in inventory at {self.path!r}"
            )
        del data["models"][model_id]
        _save_registry(self.path, data)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> dict[str, Any]:
        """Retrieve the full inventory entry for a model.

        Args:
            model_id: Model to retrieve.

        Returns:
            Dict with keys ``card``, ``tier_result``, ``registered_at``,
            ``validation_history``.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        data = _load_registry(self.path)
        if model_id not in data["models"]:
            raise KeyError(
                f"Model '{model_id}' not found in inventory at {self.path!r}"
            )
        return deepcopy(data["models"][model_id])

    def get_card(self, model_id: str) -> ModelCard:
        """Retrieve and deserialise the ModelCard for a model.

        Args:
            model_id: Model to retrieve.

        Returns:
            Deserialised :class:`~insurance_mrm.model_card.ModelCard`.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        entry = self.get(model_id)
        return ModelCard.from_dict(entry["card"])

    def list(
        self,
        status: Optional[str] = None,
        tier: Optional[int] = None,
        owner: Optional[str] = None,
        model_class: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all models in the inventory with summary fields.

        Returns a list of dicts, each containing the key summary fields for
        a model. This is intentionally a flat structure (no nested Assumption
        objects) to make it easy to tabulate or convert to a DataFrame.

        Args:
            status: Filter by ``champion_challenger_status``.
            tier: Filter by ``materiality_tier``.
            owner: Filter by ``monitoring_owner`` (substring match).
            model_class: Filter by ``model_class``.

        Returns:
            List of summary dicts, sorted by ``materiality_tier`` ascending
            then ``model_name`` ascending.
        """
        data = _load_registry(self.path)
        rows: list[dict[str, Any]] = []
        for model_id, entry in data["models"].items():
            card = entry["card"]

            if status and card.get("champion_challenger_status") != status:
                continue
            if tier is not None and card.get("materiality_tier") != tier:
                continue
            if owner and owner.lower() not in card.get("monitoring_owner", "").lower():
                continue
            if model_class and card.get("model_class") != model_class:
                continue

            rows.append({
                "model_id": card.get("model_id"),
                "model_name": card.get("model_name"),
                "version": card.get("version"),
                "model_class": card.get("model_class"),
                "champion_challenger_status": card.get("champion_challenger_status"),
                "materiality_tier": card.get("materiality_tier"),
                "tier_label": _tier_label(card.get("materiality_tier")),
                "gwp_impacted": card.get("gwp_impacted", 0.0),
                "overall_rag": card.get("overall_rag", ""),
                "next_review_date": card.get("next_review_date", ""),
                "last_validation_run": card.get("last_validation_run", ""),
                "monitoring_owner": card.get("monitoring_owner", ""),
                "developer": card.get("developer", ""),
                "registered_at": entry.get("registered_at", ""),
            })

        rows.sort(
            key=lambda r: (
                r["materiality_tier"] if r["materiality_tier"] is not None else 99,
                r["model_name"] or "",
            )
        )
        return rows

    def due_for_review(
        self, within_days: int = 60
    ) -> list[dict[str, Any]]:
        """Return models whose next review is within *within_days* days.

        Includes overdue models (past their review date).

        Args:
            within_days: Lookahead window in days.

        Returns:
            List of summary dicts from :meth:`list`, filtered and sorted by
            ``next_review_date`` ascending.
        """
        cutoff = date.today() + timedelta(days=within_days)
        rows = self.list()
        due = []
        for row in rows:
            nrd = row.get("next_review_date", "")
            if not nrd:
                continue
            try:
                review_date = date.fromisoformat(nrd[:10])
            except ValueError:
                continue
            if review_date <= cutoff:
                due.append(row)
        due.sort(key=lambda r: r.get("next_review_date", ""))
        return due

    def overdue(self) -> list[dict[str, Any]]:
        """Return models that are past their next review date.

        Returns:
            List of summary dicts, sorted by ``next_review_date`` ascending
            (most overdue first).
        """
        today = date.today().isoformat()[:10]
        rows = self.list()
        result = []
        for row in rows:
            nrd = row.get("next_review_date", "")
            if nrd and nrd[:10] < today:
                result.append(row)
        result.sort(key=lambda r: r.get("next_review_date", ""))
        return result

    def validation_history(self, model_id: str) -> list[dict[str, Any]]:
        """Return the validation history for a model.

        Args:
            model_id: The model to query.

        Returns:
            List of validation history records, most recent last.

        Raises:
            KeyError: If ``model_id`` is not in the inventory.
        """
        entry = self.get(model_id)
        return entry.get("validation_history", [])

    def events(
        self,
        model_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return audit log events.

        Args:
            model_id: If provided, filter to events for this model.
            event_type: If provided, filter to this event type.

        Returns:
            List of event dicts, oldest first.
        """
        data = _load_registry(self.path)
        result = []
        for evt in data.get("events", []):
            if model_id and evt.get("model_id") != model_id:
                continue
            if event_type and evt.get("event_type") != event_type:
                continue
            result.append(dict(evt))
        return result

    def summary(self) -> dict[str, Any]:
        """Return high-level inventory statistics.

        Returns:
            Dict with total model count, count by tier, count by status,
            count by RAG, number overdue for review.
        """
        rows = self.list()
        by_tier: dict[Any, int] = {}
        by_status: dict[str, int] = {}
        by_rag: dict[str, int] = {}
        for row in rows:
            t = row.get("materiality_tier")
            by_tier[t] = by_tier.get(t, 0) + 1
            s = row.get("champion_challenger_status", "")
            by_status[s] = by_status.get(s, 0) + 1
            rag = row.get("overall_rag", "") or "Not assessed"
            by_rag[rag] = by_rag.get(rag, 0) + 1
        return {
            "total_models": len(rows),
            "by_tier": by_tier,
            "by_status": by_status,
            "by_rag": by_rag,
            "overdue_count": len(self.overdue()),
        }


def _tier_label(tier: Optional[int]) -> str:
    from .scorer import TIER_LABELS
    if tier is None:
        return "Not assessed"
    return TIER_LABELS.get(tier, "Unknown")
