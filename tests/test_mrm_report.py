"""Tests for GovernanceReport."""

import json
import os
import pytest

from insurance_governance.mrm.model_card import Assumption, Limitation, ModelCard
from insurance_governance.mrm.scorer import RiskTierScorer
from insurance_governance.mrm.report import GovernanceReport, _rag_badge, _tier_badge, _e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return RiskTierScorer()


@pytest.fixture
def full_card():
    return ModelCard(
        model_id="motor-freq-v2",
        model_name="Motor TPPD Frequency",
        version="2.1.0",
        model_class="pricing",
        intended_use="Frequency pricing for private motor. Not for commercial use.",
        not_intended_for=["Commercial motor", "Fleet", "Reserving"],
        target_variable="claim_count",
        distribution_family="Poisson",
        model_type="CatBoost",
        rating_factors=["driver_age", "vehicle_age", "region"],
        training_data_period=("2019-01-01", "2023-12-31"),
        developer="Pricing Team",
        champion_challenger_status="champion",
        assumptions=[
            Assumption(
                description="Claim frequency stationarity since 2022",
                risk="MEDIUM",
                mitigation="Quarterly A/E monitoring",
            ),
            Assumption(
                description="Region as proxy for road density",
                risk="LOW",
                mitigation="Annual ABI cross-check",
            ),
            Assumption(
                description="No structural break from EVs",
                risk="HIGH",
                mitigation="Segment by vehicle type quarterly",
            ),
        ],
        limitations=[
            Limitation(
                description="Thin data for vehicles > 10 years",
                impact="High variance in predictions",
                population_at_risk="Older vehicles",
            ),
        ],
        outstanding_issues=["VIF check on driver_age pending"],
        gwp_impacted=125_000_000,
        customer_facing=True,
        regulatory_use=False,
        approved_by=["Chief Actuary", "Model Risk Committee"],
        approval_date="2024-10-15",
        approval_conditions="Subject to quarterly A/E monitoring",
        next_review_date="2025-10-15",
        monitoring_owner="Sarah Ahmed",
        monitoring_frequency="Quarterly",
        monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
        last_monitoring_run="2025-09-30",
        last_validation_run="2024-10-01",
        last_validation_run_id="abc-123-def",
        overall_rag="GREEN",
        materiality_tier=1,
        tier_rationale="Tier 1: GWP £125m, high complexity",
    )


@pytest.fixture
def minimal_card():
    return ModelCard(
        model_id="minimal-model",
        model_name="Minimal Test Model",
        version="0.1.0",
    )


@pytest.fixture
def tier_result(scorer, full_card):
    return scorer.score(
        gwp_impacted=125_000_000,
        model_complexity="high",
        deployment_status="champion",
        regulatory_use=False,
        external_data=False,
        customer_facing=True,
    )


@pytest.fixture
def validation_results():
    return {
        "overall_rag": "GREEN",
        "run_id": "run-abc-123",
        "run_date": "2024-10-01",
        "gini": 0.42,
        "ae_ratio": 1.01,
        "psi_score": 0.07,
        "hl_p_value": 0.12,
        "section_results": [
            {"section": "Data Quality", "status": "GREEN", "notes": "All checks passed"},
            {"section": "Discrimination", "status": "GREEN", "notes": "Gini 0.42"},
            {"section": "Calibration", "status": "AMBER", "notes": "H-L p-value borderline"},
        ],
    }


@pytest.fixture
def monitoring_results():
    return {
        "period": "2025-Q3",
        "ae_ratio": 1.02,
        "psi_score": 0.06,
        "gini": 0.41,
        "recommendation": "Continue",
        "triggered_alerts": [],
    }


# ---------------------------------------------------------------------------
# to_dict tests
# ---------------------------------------------------------------------------

class TestGovernanceReportToDict:
    def test_has_expected_top_level_keys(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        expected_keys = [
            "report_date", "model_identity", "risk_tier",
            "validation_summary", "monitoring_summary",
            "assumptions_register", "outstanding_issues",
            "governance", "recommendations",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_model_identity_fields(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        identity = d["model_identity"]
        assert identity["model_id"] == "motor-freq-v2"
        assert identity["model_name"] == "Motor TPPD Frequency"
        assert identity["version"] == "2.1.0"
        assert identity["gwp_impacted"] == 125_000_000
        assert identity["customer_facing"] is True
        assert "Commercial motor" in identity["not_intended_for"]

    def test_risk_tier_fields_with_tier_result(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        tier = d["risk_tier"]
        assert tier["tier"] == tier_result.tier
        assert tier["score"] == tier_result.score
        assert len(tier["dimensions"]) == 6

    def test_risk_tier_falls_back_to_card(self, full_card):
        """If no TierResult, use the card's materiality_tier."""
        report = GovernanceReport(card=full_card)
        d = report.to_dict()
        assert d["risk_tier"]["tier"] == 1
        assert d["risk_tier"]["rationale"] == "Tier 1: GWP £125m, high complexity"

    def test_validation_summary_with_results(self, full_card, tier_result, validation_results):
        report = GovernanceReport(
            card=full_card, tier=tier_result, validation_results=validation_results
        )
        d = report.to_dict()
        val = d["validation_summary"]
        assert val["overall_rag"] == "GREEN"
        assert val["key_metrics"]["gini"] == 0.42
        assert val["key_metrics"]["ae_ratio"] == 1.01
        assert len(val["section_results"]) == 3

    def test_validation_summary_without_results(self, full_card):
        """Falls back to card's overall_rag."""
        report = GovernanceReport(card=full_card)
        d = report.to_dict()
        assert d["validation_summary"]["overall_rag"] == "GREEN"

    def test_monitoring_summary_with_results(
        self, full_card, tier_result, monitoring_results
    ):
        report = GovernanceReport(
            card=full_card, tier=tier_result, monitoring_results=monitoring_results
        )
        d = report.to_dict()
        mon = d["monitoring_summary"]
        assert mon["period"] == "2025-Q3"
        assert mon["ae_ratio"] == 1.02
        assert mon["recommendation"] == "Continue"

    def test_assumptions_register(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        ar = d["assumptions_register"]
        assert ar["total"] == 3
        assert ar["high_risk"] == 1
        assert ar["medium_risk"] == 1
        assert len(ar["assumptions"]) == 3

    def test_governance_fields(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        gov = d["governance"]
        assert "Chief Actuary" in gov["approved_by"]
        assert gov["approval_date"] == "2024-10-15"
        assert gov["next_review_date"] == "2025-10-15"
        assert "quarterly" in gov["approval_conditions"].lower()

    def test_outstanding_issues(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        assert len(d["outstanding_issues"]) == 1
        assert "VIF check" in d["outstanding_issues"][0]

    def test_recommendations_not_empty(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        d = report.to_dict()
        assert len(d["recommendations"]) >= 1

    def test_minimal_card_no_errors(self, minimal_card):
        report = GovernanceReport(card=minimal_card)
        d = report.to_dict()
        assert d["model_identity"]["model_id"] == "minimal-model"
        assert d["risk_tier"]["tier"] is None

    def test_custom_report_date(self, full_card):
        report = GovernanceReport(card=full_card, report_date="2025-11-01")
        d = report.to_dict()
        assert d["report_date"] == "2025-11-01"


# ---------------------------------------------------------------------------
# Recommendations logic
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_no_approval_triggers_recommendation(self):
        card = ModelCard(
            model_id="unapproved",
            model_name="Unapproved Model",
            version="1.0.0",
        )
        report = GovernanceReport(card=card)
        recs = report._build_recommendations()
        assert any("not been formally approved" in r for r in recs)

    def test_no_review_date_triggers_recommendation(self):
        card = ModelCard(
            model_id="no-review",
            model_name="No Review",
            version="1.0.0",
        )
        report = GovernanceReport(card=card)
        recs = report._build_recommendations()
        assert any("No next review date" in r for r in recs)

    def test_red_rag_triggers_recommendation(self):
        card = ModelCard(
            model_id="red-model",
            model_name="Red Model",
            version="1.0.0",
            overall_rag="RED",
        )
        report = GovernanceReport(card=card)
        recs = report._build_recommendations()
        assert any("RED" in r for r in recs)

    def test_amber_rag_triggers_recommendation(self):
        card = ModelCard(
            model_id="amber-model",
            model_name="Amber Model",
            version="1.0.0",
            overall_rag="AMBER",
        )
        report = GovernanceReport(card=card)
        recs = report._build_recommendations()
        assert any("AMBER" in r for r in recs)

    def test_high_risk_assumption_triggers_recommendation(self):
        card = ModelCard(
            model_id="risky",
            model_name="Risky Model",
            version="1.0.0",
            assumptions=[Assumption(description="test", risk="HIGH")],
        )
        report = GovernanceReport(card=card)
        recs = report._build_recommendations()
        assert any("HIGH risk" in r for r in recs)

    def test_monitoring_alerts_trigger_recommendation(self):
        card = ModelCard(
            model_id="alerted",
            model_name="Alert Model",
            version="1.0.0",
        )
        report = GovernanceReport(
            card=card,
            monitoring_results={"triggered_alerts": ["PSI exceeded 0.25"]},
        )
        recs = report._build_recommendations()
        assert any("monitoring alert" in r for r in recs)

    def test_clean_card_no_action_recommendations(self, full_card, tier_result):
        # A fully documented card with GREEN validation and no alerts
        report = GovernanceReport(
            card=full_card,
            tier=tier_result,
            validation_results={"overall_rag": "GREEN"},
            monitoring_results={"triggered_alerts": [], "recommendation": "Continue"},
        )
        recs = report._build_recommendations()
        # Should still check outstanding issues and missing validation run
        assert isinstance(recs, list)
        assert len(recs) >= 1


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

class TestGovernanceReportHTML:
    def test_to_html_returns_string(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert isinstance(html, str)

    def test_html_contains_model_name(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert "Motor TPPD Frequency" in html

    def test_html_contains_model_id(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert "motor-freq-v2" in html

    def test_html_contains_rag_badge(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert "GREEN" in html

    def test_html_contains_gini_metric(
        self, full_card, tier_result, validation_results
    ):
        report = GovernanceReport(
            card=full_card, tier=tier_result, validation_results=validation_results
        )
        html = report.to_html()
        assert "0.420" in html

    def test_html_contains_section_results(
        self, full_card, tier_result, validation_results
    ):
        report = GovernanceReport(
            card=full_card, tier=tier_result, validation_results=validation_results
        )
        html = report.to_html()
        assert "Data Quality" in html

    def test_html_is_valid_html_structure(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body" in html
        assert "</body>" in html

    def test_html_no_cdn_references(self, full_card, tier_result):
        """Verify no external CDN dependencies."""
        report = GovernanceReport(card=full_card, tier=tier_result)
        html = report.to_html()
        assert "cdn.jsdelivr.net" not in html
        assert "cdnjs.cloudflare.com" not in html
        assert "unpkg.com" not in html

    def test_html_with_minimal_card(self, minimal_card):
        """Should not raise even with empty card."""
        report = GovernanceReport(card=minimal_card)
        html = report.to_html()
        assert "Minimal Test Model" in html

    def test_save_html(self, full_card, tier_result, tmp_path):
        report = GovernanceReport(card=full_card, tier=tier_result)
        path = str(tmp_path / "report.html")
        report.save_html(path)
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read()
        assert "Motor TPPD Frequency" in content

    def test_save_json(self, full_card, tier_result, tmp_path):
        report = GovernanceReport(card=full_card, tier=tier_result)
        path = str(tmp_path / "report.json")
        report.save_json(path)
        assert os.path.exists(path)
        with open(path, "r") as f:
            data = json.load(f)
        assert data["model_identity"]["model_id"] == "motor-freq-v2"

    def test_to_json_is_valid(self, full_card, tier_result):
        report = GovernanceReport(card=full_card, tier=tier_result)
        j = report.to_json()
        data = json.loads(j)
        assert data["model_identity"]["model_id"] == "motor-freq-v2"


# ---------------------------------------------------------------------------
# HTML helper functions
# ---------------------------------------------------------------------------

class TestHTMLHelpers:
    def test_rag_badge_green(self):
        badge = _rag_badge("GREEN")
        assert "GREEN" in badge
        assert "span" in badge

    def test_rag_badge_unknown(self):
        badge = _rag_badge("Not assessed")
        assert "Not assessed" in badge

    def test_tier_badge_none(self):
        badge = _tier_badge(None)
        assert "Not assessed" in badge

    def test_tier_badge_tier1(self):
        badge = _tier_badge(1)
        assert "Tier 1" in badge
        assert "Critical" in badge

    def test_e_escapes_html(self):
        assert _e("<script>") == "&lt;script&gt;"
        assert _e("a & b") == "a &amp; b"
        assert _e('"quoted"') == "&quot;quoted&quot;"

    def test_e_handles_non_string(self):
        assert _e(42) == "42"
        assert _e(3.14) == "3.14"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_validation_results_rag_overrides_card_rag(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
            overall_rag="AMBER",
        )
        report = GovernanceReport(
            card=card,
            validation_results={"overall_rag": "GREEN"},
        )
        assert report._overall_rag == "GREEN"

    def test_monitoring_results_triggered_alerts_in_html(self):
        card = ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0.0",
        )
        report = GovernanceReport(
            card=card,
            monitoring_results={
                "triggered_alerts": ["PSI exceeded 0.25 on vehicle_age"]
            },
        )
        html = report.to_html()
        assert "PSI exceeded" in html

    def test_empty_assumptions_and_limitations(self, minimal_card):
        report = GovernanceReport(card=minimal_card)
        d = report.to_dict()
        assert d["assumptions_register"]["total"] == 0
        assert d["limitations"] == []

    def test_large_gwp_formatted_in_html(self, full_card):
        report = GovernanceReport(card=full_card)
        html = report.to_html()
        # £125m should be formatted correctly
        assert "£125.0m" in html or "£125m" in html

    def test_xss_prevention_in_model_name(self):
        card = ModelCard(
            model_id="test",
            model_name="<script>alert('xss')</script>",
            version="1.0.0",
        )
        report = GovernanceReport(card=card)
        html = report.to_html()
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
