"""Tests for ModelCard, Assumption, and Limitation."""

import json
import pytest
from insurance_governance.mrm.model_card import (
    Assumption,
    Limitation,
    ModelCard,
    CHAMPION_STATUSES,
    MODEL_CLASSES,
)


# ---------------------------------------------------------------------------
# Assumption tests
# ---------------------------------------------------------------------------

class TestAssumption:
    def test_basic_creation(self):
        a = Assumption(description="Region as proxy for road density")
        assert a.description == "Region as proxy for road density"
        assert a.risk == "LOW"
        assert a.mitigation == ""
        assert a.rationale == ""

    def test_all_risk_levels(self):
        for level in ("LOW", "MEDIUM", "HIGH"):
            a = Assumption(description="test", risk=level)
            assert a.risk == level

    def test_invalid_risk_level(self):
        with pytest.raises(ValueError, match="risk must be one of"):
            Assumption(description="test", risk="CRITICAL")

    def test_to_dict(self):
        a = Assumption(
            description="Stationarity", risk="MEDIUM", mitigation="Quarterly A/E"
        )
        d = a.to_dict()
        assert d["description"] == "Stationarity"
        assert d["risk"] == "MEDIUM"
        assert d["mitigation"] == "Quarterly A/E"

    def test_from_dict_roundtrip(self):
        a = Assumption(
            description="Stationarity",
            risk="HIGH",
            mitigation="Monthly monitoring",
            rationale="Key driver",
        )
        d = a.to_dict()
        a2 = Assumption.from_dict(d)
        assert a2.description == a.description
        assert a2.risk == a.risk
        assert a2.mitigation == a.mitigation
        assert a2.rationale == a.rationale

    def test_from_dict_defaults(self):
        a = Assumption.from_dict({"description": "test"})
        assert a.risk == "LOW"
        assert a.mitigation == ""


# ---------------------------------------------------------------------------
# Limitation tests
# ---------------------------------------------------------------------------

class TestLimitation:
    def test_basic_creation(self):
        lim = Limitation(description="Thin data for vehicles > 10 years")
        assert lim.description == "Thin data for vehicles > 10 years"
        assert lim.impact == ""
        assert lim.population_at_risk == ""
        assert lim.monitoring_flag is False

    def test_all_fields(self):
        lim = Limitation(
            description="No telematics",
            impact="Mileage self-reported; accuracy unknown",
            population_at_risk="High-mileage drivers",
            monitoring_flag=True,
        )
        assert lim.monitoring_flag is True
        assert lim.population_at_risk == "High-mileage drivers"

    def test_to_dict(self):
        lim = Limitation(description="test", impact="some impact")
        d = lim.to_dict()
        assert d["description"] == "test"
        assert d["impact"] == "some impact"

    def test_from_dict_roundtrip(self):
        lim = Limitation(
            description="Thin data",
            impact="High variance",
            population_at_risk="Old vehicles",
            monitoring_flag=True,
        )
        lim2 = Limitation.from_dict(lim.to_dict())
        assert lim2.description == lim.description
        assert lim2.monitoring_flag == lim.monitoring_flag


# ---------------------------------------------------------------------------
# ModelCard tests
# ---------------------------------------------------------------------------

class TestModelCardCreation:
    def test_minimal_creation(self):
        card = ModelCard(
            model_id="motor-freq-v1",
            model_name="Motor Frequency",
            version="1.0.0",
        )
        assert card.model_id == "motor-freq-v1"
        assert card.model_class == "pricing"
        assert card.champion_challenger_status == "development"
        assert card.customer_facing is True
        assert card.regulatory_use is False
        assert card.gwp_impacted == 0.0
        assert card.materiality_tier is None

    def test_full_creation(self):
        card = ModelCard(
            model_id="motor-freq-v2",
            model_name="Motor TPPD Frequency",
            version="2.1.0",
            model_class="pricing",
            intended_use="Frequency pricing for private motor",
            not_intended_for=["Commercial motor", "Fleet"],
            target_variable="claim_count",
            distribution_family="Poisson",
            model_type="CatBoost",
            rating_factors=["driver_age", "vehicle_age", "region"],
            training_data_period=("2019-01-01", "2023-12-31"),
            developer="Pricing Team",
            champion_challenger_status="champion",
            assumptions=[
                Assumption(description="Stationarity", risk="MEDIUM")
            ],
            limitations=["Thin data for old vehicles"],
            outstanding_issues=["VIF check pending"],
            gwp_impacted=125_000_000,
            customer_facing=True,
            regulatory_use=False,
            approved_by=["Chief Actuary"],
            approval_date="2024-10-15",
            next_review_date="2025-10-15",
            monitoring_owner="Sarah Ahmed",
            monitoring_triggers={"psi_score": 0.25},
        )
        assert card.model_id == "motor-freq-v2"
        assert len(card.assumptions) == 1
        assert len(card.limitations) == 1
        assert isinstance(card.limitations[0], Limitation)

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            ModelCard(model_id="", model_name="Test", version="1.0.0")

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ModelCard(model_id="test", model_name="", version="1.0.0")

    def test_empty_version_raises(self):
        with pytest.raises(ValueError, match="version cannot be empty"):
            ModelCard(model_id="test", model_name="Test", version="")

    def test_invalid_model_class_raises(self):
        with pytest.raises(ValueError, match="model_class must be one of"):
            ModelCard(
                model_id="test",
                model_name="Test",
                version="1.0.0",
                model_class="fraud",
            )

    def test_valid_model_classes(self):
        for cls in MODEL_CLASSES:
            card = ModelCard(
                model_id=f"test-{cls}",
                model_name="Test",
                version="1.0.0",
                model_class=cls,
            )
            assert card.model_class == cls

    def test_invalid_champion_status_raises(self):
        with pytest.raises(ValueError, match="champion_challenger_status"):
            ModelCard(
                model_id="test",
                model_name="Test",
                version="1.0.0",
                champion_challenger_status="live",
            )

    def test_valid_champion_statuses(self):
        for status in CHAMPION_STATUSES:
            card = ModelCard(
                model_id=f"test-{status}",
                model_name="Test",
                version="1.0.0",
                champion_challenger_status=status,
            )
            assert card.champion_challenger_status == status

    def test_invalid_materiality_tier_raises(self):
        with pytest.raises(ValueError, match="materiality_tier must be 1, 2, or 3"):
            ModelCard(
                model_id="test",
                model_name="Test",
                version="1.0.0",
                materiality_tier=5,
            )

    def test_valid_materiality_tiers(self):
        for tier in (1, 2, 3):
            card = ModelCard(
                model_id=f"test-{tier}",
                model_name="Test",
                version="1.0.0",
                materiality_tier=tier,
            )
            assert card.materiality_tier == tier

    def test_invalid_rag_raises(self):
        with pytest.raises(ValueError, match="overall_rag must be"):
            ModelCard(
                model_id="test",
                model_name="Test",
                version="1.0.0",
                overall_rag="YELLOW",
            )

    def test_string_limitations_coerced(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
            limitations=["Plain string limitation"],
        )
        assert isinstance(card.limitations[0], Limitation)
        assert card.limitations[0].description == "Plain string limitation"

    def test_dict_limitations_coerced(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
            limitations=[{"description": "Test limitation", "impact": "Some impact"}],
        )
        assert isinstance(card.limitations[0], Limitation)
        assert card.limitations[0].impact == "Some impact"

    def test_invalid_limitation_type_raises(self):
        with pytest.raises(TypeError, match="limitations entries"):
            ModelCard(
                model_id="test",
                model_name="Test",
                version="1.0.0",
                limitations=[123],
            )

    def test_dict_assumptions_coerced(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
            assumptions=[{"description": "test assumption", "risk": "HIGH"}],
        )
        assert isinstance(card.assumptions[0], Assumption)
        assert card.assumptions[0].risk == "HIGH"


class TestModelCardProperties:
    def setup_method(self):
        self.card = ModelCard(
            model_id="motor-freq-v2",
            model_name="Motor Frequency",
            version="2.0.0",
            assumptions=[
                Assumption(description="A1", risk="LOW"),
                Assumption(description="A2", risk="MEDIUM"),
                Assumption(description="A3", risk="HIGH"),
                Assumption(description="A4", risk="HIGH"),
            ],
            approved_by=["Chief Actuary"],
            approval_date="2024-10-01",
        )

    def test_high_risk_assumptions(self):
        high = self.card.high_risk_assumptions
        assert len(high) == 2
        assert all(a.risk == "HIGH" for a in high)

    def test_medium_risk_assumptions(self):
        med = self.card.medium_risk_assumptions
        assert len(med) == 1
        assert med[0].risk == "MEDIUM"

    def test_is_approved_true(self):
        assert self.card.is_approved is True

    def test_is_approved_false_no_approvers(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
        )
        assert card.is_approved is False

    def test_is_approved_false_no_date(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
            approved_by=["Someone"],
        )
        assert card.is_approved is False

    def test_assumption_summary(self):
        summary = self.card.assumption_summary()
        assert summary["LOW"] == 1
        assert summary["MEDIUM"] == 1
        assert summary["HIGH"] == 2


class TestModelCardSerialisation:
    def setup_method(self):
        self.card = ModelCard(
            model_id="motor-freq-v2",
            model_name="Motor TPPD Frequency",
            version="2.1.0",
            model_class="pricing",
            intended_use="Frequency pricing for private motor",
            not_intended_for=["Commercial motor", "Fleet"],
            target_variable="claim_count",
            distribution_family="Poisson",
            model_type="CatBoost",
            rating_factors=["driver_age", "region"],
            training_data_period=("2019-01-01", "2023-12-31"),
            developer="Pricing Team",
            champion_challenger_status="champion",
            assumptions=[
                Assumption(description="Stationarity", risk="MEDIUM", mitigation="Quarterly A/E"),
            ],
            limitations=["Thin data for old vehicles"],
            outstanding_issues=["VIF check pending"],
            gwp_impacted=125_000_000,
            customer_facing=True,
            regulatory_use=False,
            approved_by=["Chief Actuary"],
            approval_date="2024-10-15",
            approval_conditions="Subject to quarterly monitoring",
            next_review_date="2025-10-15",
            monitoring_owner="Sarah Ahmed",
            monitoring_frequency="Quarterly",
            monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
            materiality_tier=1,
            tier_rationale="Tier 1: high GWP",
            overall_rag="GREEN",
        )

    def test_to_dict_has_expected_keys(self):
        d = self.card.to_dict()
        expected_keys = [
            "model_id", "model_name", "version", "model_class",
            "assumptions", "limitations", "approved_by", "created_at",
        ]
        for key in expected_keys:
            assert key in d

    def test_to_dict_assumptions_are_dicts(self):
        d = self.card.to_dict()
        assert isinstance(d["assumptions"][0], dict)
        assert d["assumptions"][0]["risk"] == "MEDIUM"

    def test_to_dict_limitations_are_dicts(self):
        d = self.card.to_dict()
        assert isinstance(d["limitations"][0], dict)
        assert d["limitations"][0]["description"] == "Thin data for old vehicles"

    def test_to_dict_training_period_is_list(self):
        d = self.card.to_dict()
        assert isinstance(d["training_data_period"], list)

    def test_to_json_is_valid_json(self):
        j = self.card.to_json()
        loaded = json.loads(j)
        assert loaded["model_id"] == "motor-freq-v2"

    def test_from_dict_roundtrip(self):
        d = self.card.to_dict()
        card2 = ModelCard.from_dict(d)
        assert card2.model_id == self.card.model_id
        assert card2.version == self.card.version
        assert len(card2.assumptions) == len(self.card.assumptions)
        assert card2.assumptions[0].risk == "MEDIUM"
        assert len(card2.limitations) == len(self.card.limitations)
        assert card2.limitations[0].description == "Thin data for old vehicles"

    def test_from_json_roundtrip(self):
        j = self.card.to_json()
        card2 = ModelCard.from_json(j)
        assert card2.model_id == self.card.model_id
        assert card2.materiality_tier == 1
        assert card2.overall_rag == "GREEN"

    def test_from_dict_with_list_training_period(self):
        d = self.card.to_dict()
        d["training_data_period"] = ["2019-01-01", "2023-12-31"]
        card2 = ModelCard.from_dict(d)
        assert card2.training_data_period == ("2019-01-01", "2023-12-31")

    def test_empty_card_roundtrip(self):
        card = ModelCard(
            model_id="minimal",
            model_name="Minimal Card",
            version="0.1.0",
        )
        d = card.to_dict()
        card2 = ModelCard.from_dict(d)
        assert card2.model_id == "minimal"
        assert card2.assumptions == []
        assert card2.limitations == []
