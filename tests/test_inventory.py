"""Tests for ModelInventory."""

import json
import os
import pytest
from datetime import date, timedelta

from insurance_governance.mrm.model_card import Assumption, ModelCard
from insurance_governance.mrm.inventory import ModelInventory
from insurance_governance.mrm.scorer import RiskTierScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path):
    return str(tmp_path / "registry.json")


@pytest.fixture
def scorer():
    return RiskTierScorer()


@pytest.fixture
def card_motor():
    return ModelCard(
        model_id="motor-freq-v2",
        model_name="Motor TPPD Frequency",
        version="2.1.0",
        model_class="pricing",
        intended_use="Frequency pricing for private motor",
        champion_challenger_status="champion",
        gwp_impacted=125_000_000,
        approved_by=["Chief Actuary"],
        approval_date="2024-10-15",
        next_review_date="2028-10-15",
        monitoring_owner="Sarah Ahmed",
        overall_rag="GREEN",
    )


@pytest.fixture
def card_home():
    return ModelCard(
        model_id="home-buildings-v1",
        model_name="Home Buildings Severity",
        version="1.0.0",
        model_class="pricing",
        champion_challenger_status="champion",
        gwp_impacted=45_000_000,
        next_review_date=(date.today() + timedelta(days=20)).isoformat(),
        monitoring_owner="James Brown",
    )


@pytest.fixture
def card_dev():
    return ModelCard(
        model_id="fleet-research-v0",
        model_name="Fleet Research Model",
        version="0.1.0",
        model_class="pricing",
        champion_challenger_status="development",
        gwp_impacted=0.0,
        next_review_date=(date.today() - timedelta(days=10)).isoformat(),
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestModelInventoryCRUD:
    def test_register_creates_file(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        assert os.path.exists(tmp_registry)

    def test_register_returns_model_id(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        result = inv.register(card_motor)
        assert result == "motor-freq-v2"

    def test_register_and_get(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        entry = inv.get("motor-freq-v2")
        assert entry["card"]["model_id"] == "motor-freq-v2"

    def test_get_missing_raises(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        with pytest.raises(KeyError, match="motor-freq-v2"):
            inv.get("motor-freq-v2")

    def test_get_card(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        card = inv.get_card("motor-freq-v2")
        assert isinstance(card, ModelCard)
        assert card.model_id == "motor-freq-v2"
        assert card.monitoring_owner == "Sarah Ahmed"

    def test_register_with_tier(self, tmp_registry, card_motor, scorer):
        tier = scorer.score(
            gwp_impacted=125_000_000,
            model_complexity="high",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor, tier)
        entry = inv.get("motor-freq-v2")
        assert entry["tier_result"] is not None
        assert entry["tier_result"]["tier"] == tier.tier
        # Card should have materiality_tier set
        assert entry["card"]["materiality_tier"] == tier.tier

    def test_register_update_existing(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        # Update the card and re-register
        card_motor.version = "2.2.0"
        inv.register(card_motor)
        entry = inv.get("motor-freq-v2")
        assert entry["card"]["version"] == "2.2.0"
        # Should not create duplicate
        rows = inv.list()
        motor_rows = [r for r in rows if r["model_id"] == "motor-freq-v2"]
        assert len(motor_rows) == 1

    def test_remove(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.remove("motor-freq-v2")
        rows = inv.list()
        assert len(rows) == 0

    def test_remove_missing_raises(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        with pytest.raises(KeyError):
            inv.remove("nonexistent")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestModelInventoryFiltering:
    def setup_method(self):
        pass

    def test_list_all(self, tmp_registry, card_motor, card_home, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.register(card_home)
        inv.register(card_dev)
        rows = inv.list()
        assert len(rows) == 3

    def test_filter_by_status(self, tmp_registry, card_motor, card_home, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.register(card_home)
        inv.register(card_dev)
        rows = inv.list(status="champion")
        assert len(rows) == 2
        assert all(r["champion_challenger_status"] == "champion" for r in rows)

    def test_filter_by_tier(self, tmp_registry, card_motor, scorer):
        inv = ModelInventory(tmp_registry)
        tier = scorer.score(
            gwp_impacted=125_000_000,
            model_complexity="high",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        inv.register(card_motor, tier)
        rows = inv.list(tier=tier.tier)
        assert len(rows) == 1

    def test_filter_by_owner(self, tmp_registry, card_motor, card_home):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.register(card_home)
        rows = inv.list(owner="Sarah")
        assert len(rows) == 1
        assert rows[0]["monitoring_owner"] == "Sarah Ahmed"

    def test_filter_by_model_class(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        rows = inv.list(model_class="pricing")
        assert len(rows) == 1
        rows_empty = inv.list(model_class="reserving")
        assert len(rows_empty) == 0

    def test_list_sorted_by_tier_then_name(self, tmp_registry, scorer):
        inv = ModelInventory(tmp_registry)
        card_a = ModelCard(
            model_id="model-a", model_name="AAA", version="1.0.0",
            materiality_tier=2
        )
        card_b = ModelCard(
            model_id="model-b", model_name="BBB", version="1.0.0",
            materiality_tier=1
        )
        card_c = ModelCard(
            model_id="model-c", model_name="CCC", version="1.0.0",
            materiality_tier=2
        )
        inv.register(card_a)
        inv.register(card_b)
        inv.register(card_c)
        rows = inv.list()
        assert rows[0]["model_id"] == "model-b"  # tier 1 first
        # tier 2 models sorted by name
        tier2_names = [r["model_name"] for r in rows if r["materiality_tier"] == 2]
        assert tier2_names == sorted(tier2_names)

    def test_empty_inventory_list(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        rows = inv.list()
        assert rows == []


# ---------------------------------------------------------------------------
# Due for review
# ---------------------------------------------------------------------------

class TestDueForReview:
    def test_due_within_window(self, tmp_registry, card_home):
        inv = ModelInventory(tmp_registry)
        inv.register(card_home)  # next_review_date = today + 20 days
        due = inv.due_for_review(within_days=30)
        assert any(r["model_id"] == "home-buildings-v1" for r in due)

    def test_not_due_outside_window(self, tmp_registry, card_motor):
        # next_review_date = 2028-10-15 (far future)
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        due = inv.due_for_review(within_days=30)
        motor_due = [r for r in due if r["model_id"] == "motor-freq-v2"]
        assert len(motor_due) == 0

    def test_overdue_appears_in_due(self, tmp_registry, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_dev)  # next_review_date = today - 10 days
        due = inv.due_for_review(within_days=0)
        assert any(r["model_id"] == "fleet-research-v0" for r in due)

    def test_overdue_method(self, tmp_registry, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_dev)
        overdue = inv.overdue()
        assert any(r["model_id"] == "fleet-research-v0" for r in overdue)

    def test_no_review_date_excluded(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        card = ModelCard(
            model_id="no-date",
            model_name="No Date",
            version="1.0.0",
        )
        inv.register(card)
        due = inv.due_for_review(within_days=9999)
        assert not any(r["model_id"] == "no-date" for r in due)


# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------

class TestUpdateOperations:
    def test_update_validation(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.update_validation(
            model_id="motor-freq-v2",
            validation_date="2025-10-01",
            overall_rag="GREEN",
            next_review_date="2026-10-01",
            run_id="abc-123",
            notes="All sections passed",
        )
        card = inv.get_card("motor-freq-v2")
        assert card.last_validation_run == "2025-10-01"
        assert card.overall_rag == "GREEN"
        assert card.next_review_date == "2026-10-01"
        assert card.last_validation_run_id == "abc-123"

    def test_update_validation_history_appended(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.update_validation(
            "motor-freq-v2", "2025-10-01", "GREEN", "2026-10-01"
        )
        inv.update_validation(
            "motor-freq-v2", "2024-10-01", "AMBER", "2025-10-01"
        )
        history = inv.validation_history("motor-freq-v2")
        assert len(history) == 2

    def test_update_validation_invalid_rag_raises(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        with pytest.raises(ValueError, match="overall_rag must be"):
            inv.update_validation(
                "motor-freq-v2", "2025-10-01", "YELLOW", "2026-10-01"
            )

    def test_update_validation_missing_model_raises(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        with pytest.raises(KeyError):
            inv.update_validation(
                "nonexistent", "2025-10-01", "GREEN", "2026-10-01"
            )

    def test_update_status(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.update_status("motor-freq-v2", "retired")
        card = inv.get_card("motor-freq-v2")
        assert card.champion_challenger_status == "retired"

    def test_update_status_invalid_raises(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        with pytest.raises(ValueError, match="champion_challenger_status"):
            inv.update_status("motor-freq-v2", "live")

    def test_update_status_missing_model_raises(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        with pytest.raises(KeyError):
            inv.update_status("nonexistent", "retired")


# ---------------------------------------------------------------------------
# Events / audit log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_log_event(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.log_event(
            model_id="motor-freq-v2",
            event_type="monitoring_trigger",
            description="PSI exceeded 0.25 on driver_age",
            triggered_by="insurance-monitoring Q3 2025",
        )
        events = inv.events(model_id="motor-freq-v2")
        assert len(events) == 1
        assert events[0]["event_type"] == "monitoring_trigger"

    def test_log_event_missing_model_raises(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        with pytest.raises(KeyError):
            inv.log_event("nonexistent", "test", "description")

    def test_events_filter_by_type(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.log_event("motor-freq-v2", "monitoring_trigger", "PSI alert")
        inv.log_event("motor-freq-v2", "status_change", "Promoted to champion")
        events = inv.events(event_type="monitoring_trigger")
        assert len(events) == 1

    def test_events_filter_by_model(self, tmp_registry, card_motor, card_home):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.register(card_home)
        inv.log_event("motor-freq-v2", "test", "Motor event")
        inv.log_event("home-buildings-v1", "test", "Home event")
        motor_events = inv.events(model_id="motor-freq-v2")
        assert len(motor_events) == 1
        assert motor_events[0]["description"] == "Motor event"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestInventorySummary:
    def test_summary_empty(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        s = inv.summary()
        assert s["total_models"] == 0

    def test_summary_counts(self, tmp_registry, card_motor, card_home, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        inv.register(card_home)
        inv.register(card_dev)
        s = inv.summary()
        assert s["total_models"] == 3
        assert s["by_status"]["champion"] == 2
        assert s["by_status"]["development"] == 1

    def test_summary_overdue_count(self, tmp_registry, card_dev):
        inv = ModelInventory(tmp_registry)
        inv.register(card_dev)
        s = inv.summary()
        assert s["overdue_count"] >= 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestInventoryPersistence:
    def test_data_persists_across_instances(self, tmp_registry, card_motor):
        inv1 = ModelInventory(tmp_registry)
        inv1.register(card_motor)

        inv2 = ModelInventory(tmp_registry)
        rows = inv2.list()
        assert len(rows) == 1
        assert rows[0]["model_id"] == "motor-freq-v2"

    def test_registry_file_is_valid_json(self, tmp_registry, card_motor):
        inv = ModelInventory(tmp_registry)
        inv.register(card_motor)
        with open(tmp_registry) as f:
            data = json.load(f)
        assert "models" in data
        assert "motor-freq-v2" in data["models"]

    def test_no_file_returns_empty_list(self, tmp_registry):
        inv = ModelInventory(tmp_registry)
        rows = inv.list()
        assert rows == []
