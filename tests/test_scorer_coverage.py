"""
Supplemental tests for RiskTierScorer to cover remaining scoring branches.

The existing test_scorer.py covers the main paths. This file targets specific
branches that were uncovered: validation coverage month bands and drift history
trigger counts.
"""

from __future__ import annotations

import pytest

from insurance_governance.mrm.scorer import RiskTierScorer


@pytest.fixture
def scorer() -> RiskTierScorer:
    return RiskTierScorer()


def _base_kwargs(**overrides) -> dict:
    """Return a minimal set of kwargs that yield a deterministic baseline score."""
    base = dict(
        gwp_impacted=50_000_000,
        model_complexity="medium",
        deployment_status="champion",
        regulatory_use=False,
        external_data=False,
        customer_facing=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Validation coverage month bands
# ---------------------------------------------------------------------------


class TestValidationCoverageBands:
    """Each time band in _score_validation_coverage should be exercised."""

    def test_validated_within_6_months_lowest_score(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=3.0))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        # Within 6 months: raw = 0.0
        assert vc.score == 0.0
        assert "within 6 months" in vc.rationale

    def test_validated_between_6_and_12_months(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=9.0))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc.score == 20.0
        assert "6-12 months" in vc.rationale

    def test_validated_between_12_and_18_months(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=15.0))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc.score == 50.0
        assert "12-18 months" in vc.rationale

    def test_validated_between_18_and_24_months(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=21.0))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc.score == 75.0
        assert "18-24 months" in vc.rationale

    def test_validated_over_24_months_ago(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=30.0))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc.score == 100.0
        assert "overdue" in vc.rationale

    def test_never_validated_worst_score(self, scorer):
        result = scorer.score(**_base_kwargs(validation_months_ago=None))
        vc = next(d for d in result.dimensions if d.name == "validation_coverage")
        assert vc.score == 100.0
        assert "Never" in vc.rationale

    def test_validation_score_monotonically_increases_with_age(self, scorer):
        """Older validations should produce higher (worse) scores."""
        months = [3.0, 9.0, 15.0, 21.0, 30.0]
        scores = []
        for m in months:
            r = scorer.score(**_base_kwargs(validation_months_ago=m))
            vc = next(d for d in r.dimensions if d.name == "validation_coverage")
            scores.append(vc.score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


# ---------------------------------------------------------------------------
# Drift history trigger bands
# ---------------------------------------------------------------------------


class TestDriftHistoryBands:
    """Each trigger count band in _score_drift_history should be exercised."""

    def test_zero_triggers_lowest_drift_score(self, scorer):
        result = scorer.score(**_base_kwargs(drift_triggers_last_year=0))
        dh = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh.score == 0.0
        assert "No monitoring triggers" in dh.rationale

    def test_one_trigger(self, scorer):
        result = scorer.score(**_base_kwargs(drift_triggers_last_year=1))
        dh = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh.score == 33.0
        assert "1 monitoring trigger" in dh.rationale

    def test_two_triggers(self, scorer):
        result = scorer.score(**_base_kwargs(drift_triggers_last_year=2))
        dh = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh.score == 67.0
        assert "2 monitoring triggers" in dh.rationale

    def test_three_triggers(self, scorer):
        result = scorer.score(**_base_kwargs(drift_triggers_last_year=3))
        dh = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh.score == 67.0
        assert "3 monitoring triggers" in dh.rationale

    def test_four_or_more_triggers_highest_score(self, scorer):
        result = scorer.score(**_base_kwargs(drift_triggers_last_year=5))
        dh = next(d for d in result.dimensions if d.name == "drift_history")
        assert dh.score == 100.0
        assert "elevated drift" in dh.rationale

    def test_drift_score_monotonically_increases(self, scorer):
        """More triggers should produce equal or higher drift scores."""
        trigger_counts = [0, 1, 2, 4]
        scores = []
        for t in trigger_counts:
            r = scorer.score(**_base_kwargs(drift_triggers_last_year=t))
            dh = next(d for d in r.dimensions if d.name == "drift_history")
            scores.append(dh.score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


# ---------------------------------------------------------------------------
# _assign_tier fallback (score below all thresholds)
# ---------------------------------------------------------------------------


class TestTierAssignmentFallback:
    def test_score_below_all_thresholds_returns_max_tier(self):
        """With non-standard thresholds where no tier is reached, should return
        the tier with the highest key (least urgent)."""
        # Set thresholds requiring very high score
        scorer = RiskTierScorer(thresholds={1: 200, 2: 150, 3: 100})
        # Minimum possible score (low complexity, development, no external data,
        # no regulatory use, not customer-facing, recently validated, no drift)
        result = scorer.score(
            gwp_impacted=0.0,
            model_complexity="low",
            deployment_status="development",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
            validation_months_ago=1.0,
            drift_triggers_last_year=0,
        )
        # No threshold is met, so should return max(thresholds.keys()) = 3
        assert result.tier == 3


# ---------------------------------------------------------------------------
# Deployment status sub-scores
# ---------------------------------------------------------------------------


class TestDeploymentStatusSubScores:
    """Verify each deployment status maps to the correct sub-score."""

    def test_challenger_status(self, scorer):
        result = scorer.score(**_base_kwargs(deployment_status="challenger"))
        re = next(d for d in result.dimensions if d.name == "regulatory_exposure")
        # challenger = 20 pts out of regulatory_exposure sub-score
        assert "challenger" in re.rationale

    def test_shadow_status(self, scorer):
        result = scorer.score(**_base_kwargs(deployment_status="shadow"))
        re = next(d for d in result.dimensions if d.name == "regulatory_exposure")
        assert "shadow" in re.rationale

    def test_development_status(self, scorer):
        result = scorer.score(**_base_kwargs(deployment_status="development"))
        re = next(d for d in result.dimensions if d.name == "regulatory_exposure")
        assert "development" in re.rationale

    def test_retired_status(self, scorer):
        result = scorer.score(**_base_kwargs(deployment_status="retired"))
        re = next(d for d in result.dimensions if d.name == "regulatory_exposure")
        assert "retired" in re.rationale

    def test_champion_has_higher_re_than_shadow(self, scorer):
        r_champ = scorer.score(**_base_kwargs(deployment_status="champion"))
        r_shadow = scorer.score(**_base_kwargs(deployment_status="shadow"))
        re_champ = next(d for d in r_champ.dimensions if d.name == "regulatory_exposure")
        re_shadow = next(d for d in r_shadow.dimensions if d.name == "regulatory_exposure")
        assert re_champ.score > re_shadow.score
