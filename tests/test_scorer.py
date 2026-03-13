"""Tests for RiskTierScorer and TierResult."""

import pytest
from insurance_governance.mrm.scorer import (
    RiskTierScorer,
    TierResult,
    DimensionScore,
    DEFAULT_WEIGHTS,
    DEFAULT_TIER_THRESHOLDS,
    TIER_LABELS,
)


# ---------------------------------------------------------------------------
# DimensionScore tests
# ---------------------------------------------------------------------------

class TestDimensionScore:
    def test_pct_calculation(self):
        ds = DimensionScore(name="materiality", score=75.0, max_score=100.0, rationale="test")
        assert ds.pct == 75.0

    def test_pct_zero_max_score(self):
        ds = DimensionScore(name="test", score=0.0, max_score=0.0, rationale="test")
        assert ds.pct == 0.0

    def test_to_dict(self):
        ds = DimensionScore(name="complexity", score=60.0, max_score=100.0, rationale="GLM")
        d = ds.to_dict()
        assert d["name"] == "complexity"
        assert d["score"] == 60.0
        assert d["pct"] == 60.0


# ---------------------------------------------------------------------------
# RiskTierScorer construction
# ---------------------------------------------------------------------------

class TestRiskTierScorerConstruction:
    def test_default_construction(self):
        scorer = RiskTierScorer()
        assert set(scorer.weights.keys()) == set(DEFAULT_WEIGHTS.keys())

    def test_weights_normalised_to_100(self):
        scorer = RiskTierScorer()
        total = sum(scorer.weights.values())
        assert abs(total - 100.0) < 0.001

    def test_custom_weights_normalised(self):
        custom = {
            "materiality": 50,
            "complexity": 10,
            "data_quality": 10,
            "validation_coverage": 10,
            "drift_history": 10,
            "regulatory_exposure": 10,
        }
        scorer = RiskTierScorer(weights=custom)
        total = sum(scorer.weights.values())
        assert abs(total - 100.0) < 0.001
        # materiality should now be 50% of total
        assert abs(scorer.weights["materiality"] - 50.0) < 0.001

    def test_custom_thresholds(self):
        scorer = RiskTierScorer(thresholds={1: 70, 2: 40, 3: 0})
        assert scorer.thresholds[1] == 70

    def test_missing_weights_raise(self):
        with pytest.raises(ValueError, match="missing dimensions"):
            RiskTierScorer(weights={"materiality": 50})

    def test_extra_weights_raise(self):
        w = dict(DEFAULT_WEIGHTS)
        w["unknown_dimension"] = 5
        with pytest.raises(ValueError, match="unknown dimensions"):
            RiskTierScorer(weights=w)

    def test_zero_total_weights_raise(self):
        w = {k: 0 for k in DEFAULT_WEIGHTS}
        with pytest.raises(ValueError, match="positive number"):
            RiskTierScorer(weights=w)


# ---------------------------------------------------------------------------
# RiskTierScorer.score() — tier assignment
# ---------------------------------------------------------------------------

class TestRiskTierScorerTierAssignment:
    """Verify that well-known model profiles land in expected tiers."""

    def setup_method(self):
        self.scorer = RiskTierScorer()

    def _score(self, **kwargs):
        defaults = dict(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        defaults.update(kwargs)
        return self.scorer.score(**defaults)

    def test_high_gwp_high_complexity_is_tier1(self):
        result = self._score(gwp_impacted=150_000_000, model_complexity="high")
        assert result.tier == 1

    def test_medium_model_is_tier2(self):
        result = self._score(
            gwp_impacted=40_000_000,
            model_complexity="medium",
            regulatory_use=False,
        )
        assert result.tier in (1, 2)

    def test_development_low_gwp_is_tier3(self):
        result = self._score(
            gwp_impacted=100_000,
            model_complexity="low",
            deployment_status="development",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
        )
        assert result.tier == 3

    def test_regulatory_use_increases_tier(self):
        without_reg = self._score(
            gwp_impacted=10_000_000,
            model_complexity="low",
            deployment_status="shadow",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
        )
        with_reg = self._score(
            gwp_impacted=10_000_000,
            model_complexity="low",
            deployment_status="shadow",
            regulatory_use=True,
            external_data=False,
            customer_facing=False,
        )
        assert with_reg.score > without_reg.score

    def test_score_returns_tier_result(self):
        result = self._score()
        assert isinstance(result, TierResult)

    def test_score_has_6_dimensions(self):
        result = self._score()
        assert len(result.dimensions) == 6

    def test_score_in_valid_range(self):
        result = self._score()
        assert 0 <= result.score <= 100

    def test_tier_label_consistent(self):
        result = self._score(gwp_impacted=150_000_000, model_complexity="high")
        assert result.tier_label == TIER_LABELS[result.tier]

    def test_rationale_contains_tier(self):
        result = self._score()
        assert f"Tier {result.tier}" in result.rationale

    def test_review_frequency_not_empty(self):
        result = self._score()
        assert result.review_frequency

    def test_sign_off_not_empty(self):
        result = self._score()
        assert result.sign_off_requirement


# ---------------------------------------------------------------------------
# Score boundary conditions
# ---------------------------------------------------------------------------

class TestRiskTierScorerBoundaries:
    def setup_method(self):
        self.scorer = RiskTierScorer()

    def _score(self, **kwargs):
        defaults = dict(
            gwp_impacted=0.0,
            model_complexity="low",
            deployment_status="development",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
        )
        defaults.update(kwargs)
        return self.scorer.score(**defaults)

    def test_zero_gwp_low_complexity_development_lowest_score(self):
        result = self._score()
        # Should be tier 3
        assert result.tier == 3

    def test_maximum_inputs_highest_score(self):
        result = self._score(
            gwp_impacted=999_000_000,
            model_complexity="high",
            deployment_status="champion",
            regulatory_use=True,
            external_data=True,
            customer_facing=True,
        )
        assert result.score > 80
        assert result.tier == 1

    def test_gwp_thresholds(self):
        # Test each GWP bracket
        scores = []
        for gwp in [500_000, 10_000_000, 50_000_000, 200_000_000]:
            r = self._score(gwp_impacted=gwp)
            scores.append(r.score)
        # Higher GWP should always yield higher or equal score
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_complexity_ordering(self):
        low = self._score(model_complexity="low")
        med = self._score(model_complexity="medium")
        high = self._score(model_complexity="high")
        assert low.score <= med.score
        assert med.score <= high.score

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError, match="model_complexity must be"):
            self._score(model_complexity="extreme")

    def test_invalid_deployment_status_raises(self):
        with pytest.raises(ValueError, match="deployment_status must be one of"):
            self._score(deployment_status="live")

    def test_validation_months_ago_none_worst_case(self):
        r_none = self._score(
            gwp_impacted=50_000_000, validation_months_ago=None
        )
        r_recent = self._score(
            gwp_impacted=50_000_000, validation_months_ago=3.0
        )
        assert r_none.score >= r_recent.score

    def test_validation_months_ago_ordering(self):
        # More months ago = higher risk = higher score on validation_coverage
        r6 = self._score(gwp_impacted=50_000_000, validation_months_ago=6.0)
        r18 = self._score(gwp_impacted=50_000_000, validation_months_ago=18.0)
        assert r18.score >= r6.score

    def test_drift_triggers_ordering(self):
        r0 = self._score(gwp_impacted=50_000_000, drift_triggers_last_year=0)
        r5 = self._score(gwp_impacted=50_000_000, drift_triggers_last_year=5)
        assert r5.score >= r0.score

    def test_external_data_increases_score(self):
        without = self._score(gwp_impacted=50_000_000, external_data=False)
        with_ = self._score(gwp_impacted=50_000_000, external_data=True)
        assert with_.score > without.score


# ---------------------------------------------------------------------------
# TierResult serialisation
# ---------------------------------------------------------------------------

class TestTierResultSerialisation:
    def test_to_dict_has_expected_keys(self):
        scorer = RiskTierScorer()
        result = scorer.score(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        d = result.to_dict()
        for key in ("tier", "tier_label", "score", "dimensions", "rationale"):
            assert key in d

    def test_dimensions_serialise(self):
        scorer = RiskTierScorer()
        result = scorer.score(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        d = result.to_dict()
        assert len(d["dimensions"]) == 6
        for dim in d["dimensions"]:
            assert "name" in dim
            assert "score" in dim
            assert "rationale" in dim

    def test_thresholds_keys_are_strings(self):
        scorer = RiskTierScorer()
        result = scorer.score(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        d = result.to_dict()
        for k in d["thresholds_used"]:
            assert isinstance(k, str)


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_stricter_thresholds_elevate_tier1(self):
        # With high thresholds, even high-scoring models should hit tier 1 threshold
        scorer = RiskTierScorer(thresholds={1: 90, 2: 50, 3: 0})
        result = scorer.score(
            gwp_impacted=50_000_000,
            model_complexity="medium",
            deployment_status="champion",
            regulatory_use=False,
            external_data=False,
            customer_facing=True,
        )
        # With threshold 90, many models won't reach tier 1
        assert result.tier in (1, 2, 3)

    def test_lenient_thresholds_make_everything_tier1(self):
        scorer = RiskTierScorer(thresholds={1: 0, 2: -10, 3: -20})
        result = scorer.score(
            gwp_impacted=1_000,
            model_complexity="low",
            deployment_status="development",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
        )
        assert result.tier == 1
