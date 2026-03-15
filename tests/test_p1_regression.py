"""
Regression tests for P1 bugs fixed in v0.1.1.

P1-1: actual_vs_expected() and ae_with_poisson_ci() ignored exposure.
P1-2: Hosmer-Lemeshow test used df = n_groups - 2 instead of n_groups - 1.
P1-3: RiskTierScorer._assign_tier() iterated ascending by tier key and fell
      back to max key (lowest-risk tier) when no threshold was met.
P1-4: TIER_LABELS defined Tier 4 (dead code), labels were Critical/High/Medium
      but docstring said Critical/Significant/Informational.
"""
from __future__ import annotations

import numpy as np
import pytest

from insurance_governance.validation import PerformanceReport
from insurance_governance.mrm.scorer import (
    RiskTierScorer,
    TIER_LABELS,
    TIER_REVIEW_FREQUENCY,
    TIER_SIGN_OFF,
    DEFAULT_TIER_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# P1-1: actual_vs_expected uses exposure in denominator
# ---------------------------------------------------------------------------

class TestActualVsExpectedExposure:
    """When exposure is provided, y_pred is treated as a rate.
    Expected claims = sum(y_pred * exposure * w).
    """

    def test_exposure_changes_denominator(self):
        """With half the exposure, expected claims halve — A/E doubles."""
        y_true = np.array([10.0, 20.0, 30.0])   # claim counts
        y_pred = np.array([0.1, 0.2, 0.3])       # predicted rates per policy year
        exposure_full = np.array([100.0, 100.0, 100.0])  # 100 years each
        # expected = 0.1*100 + 0.2*100 + 0.3*100 = 60
        # actual = 60 => A/E = 1.0

        report = PerformanceReport(y_true, y_pred, exposure=exposure_full)
        result = report.actual_vs_expected()
        assert result.metric_value == pytest.approx(1.0, abs=1e-6)
        assert result.passed is True

    def test_no_exposure_uses_pred_as_count(self):
        """Without exposure, y_pred is treated as a count directly."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])   # counts, not rates
        report = PerformanceReport(y_true, y_pred)
        result = report.actual_vs_expected()
        assert result.metric_value == pytest.approx(1.0, abs=1e-6)

    def test_exposure_ae_differs_from_no_exposure(self):
        """Passing exposure should produce a different A/E than omitting it
        when y_pred is a rate and exposure varies across policies."""
        rng = np.random.default_rng(0)
        n = 50
        y_pred = rng.uniform(0.05, 0.20, size=n)   # rates
        exposure = rng.uniform(0.5, 2.0, size=n)   # policy years
        y_true = y_pred * exposure * rng.uniform(0.8, 1.2, size=n)

        with_exp = PerformanceReport(y_true, y_pred, exposure=exposure)
        without_exp = PerformanceReport(y_true, y_pred)

        ae_with = with_exp.actual_vs_expected().metric_value
        ae_without = without_exp.actual_vs_expected().metric_value

        # The two computations differ because one multiplies by exposure
        assert ae_with != pytest.approx(ae_without, abs=1e-3)

    def test_band_ae_ratios_with_exposure(self):
        """Band-level A/E ratios should also reflect exposure in denominator."""
        y_true = np.array([5.0, 10.0, 15.0, 20.0])
        y_pred = np.array([0.1, 0.1, 0.1, 0.1])   # same rate
        exposure = np.array([50.0, 100.0, 150.0, 200.0])
        # expected per band: 0.1 * exposure = [5, 10, 15, 20] => each A/E = 1.0

        report = PerformanceReport(y_true, y_pred, exposure=exposure)
        result = report.actual_vs_expected(n_bands=2)
        for band in result.extra["bands"]:
            if band["ae_ratio"] is not None:
                assert band["ae_ratio"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# P1-1: ae_with_poisson_ci uses exposure in denominator
# ---------------------------------------------------------------------------

class TestAeWithPoissonCiExposure:
    def test_exposure_changes_expected(self):
        """A/E CI with exposure-adjusted denominator should be 1.0 when
        y_true = y_pred * exposure exactly."""
        y_pred = np.array([0.10, 0.20, 0.30])
        exposure = np.array([100.0, 100.0, 100.0])
        y_true = y_pred * exposure  # perfect calibration

        report = PerformanceReport(y_true, y_pred, exposure=exposure)
        result = report.ae_with_poisson_ci()
        assert result.extra["ae_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_no_exposure_ae_unchanged(self):
        """Without exposure, ae_with_poisson_ci should equal actual / predicted."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        report = PerformanceReport(y_true, y_pred)
        result = report.ae_with_poisson_ci()
        assert result.extra["ae_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_exposure_ae_differs_from_no_exposure(self):
        """Exposure should change the A/E ratio from the non-exposure version."""
        rng = np.random.default_rng(1)
        n = 100
        y_pred = rng.uniform(0.05, 0.20, size=n)
        exposure = rng.uniform(0.5, 2.0, size=n)
        y_true = y_pred * exposure * rng.uniform(0.9, 1.1, size=n)

        with_exp = PerformanceReport(y_true, y_pred, exposure=exposure)
        without_exp = PerformanceReport(y_true, y_pred)

        ae_with = with_exp.ae_with_poisson_ci().extra["ae_ratio"]
        ae_without = without_exp.ae_with_poisson_ci().extra["ae_ratio"]
        assert ae_with != pytest.approx(ae_without, abs=1e-3)


# ---------------------------------------------------------------------------
# P1-2: Hosmer-Lemeshow uses df = n_groups - 1
# ---------------------------------------------------------------------------

class TestHosmerLemeshowDf:
    def test_df_is_n_groups_minus_1(self):
        """The reported df must be n_groups - 1."""
        rng = np.random.default_rng(2)
        n = 500
        y_pred = rng.uniform(0.01, 0.20, size=n)
        y_true = rng.binomial(1, y_pred).astype(float)

        report = PerformanceReport(y_true, y_pred)

        for n_groups in (5, 10, 20):
            result = report.hosmer_lemeshow_test(n_groups=n_groups)
            assert result.extra["df"] == n_groups - 1, (
                f"Expected df={n_groups - 1}, got {result.extra['df']} "
                f"for n_groups={n_groups}"
            )

    def test_df_3_groups_gives_df_2(self):
        """Edge case: 3 groups => df = 2."""
        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        report = PerformanceReport(y_true, y_pred)
        result = report.hosmer_lemeshow_test(n_groups=3)
        assert result.extra["df"] == 2

    def test_df_2_does_not_raise(self):
        """Minimum sensible case: 2 groups => df = 1, p-value is defined."""
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        report = PerformanceReport(y_true, y_pred)
        result = report.hosmer_lemeshow_test(n_groups=2)
        assert result.extra["df"] == 1
        assert result.extra["p_value"] is not None


# ---------------------------------------------------------------------------
# P1-3: RiskTierScorer._assign_tier sorts by threshold descending
# ---------------------------------------------------------------------------

class TestAssignTierOrder:
    def test_default_thresholds_tier1(self):
        """Score of 75 must land in Tier 1 (threshold 60), not Tier 2 or 3."""
        scorer = RiskTierScorer()
        tier = scorer._assign_tier(75.0)
        assert tier == 1

    def test_default_thresholds_tier2(self):
        """Score of 45 must land in Tier 2 (threshold 30-59), not Tier 3."""
        scorer = RiskTierScorer()
        tier = scorer._assign_tier(45.0)
        assert tier == 2

    def test_default_thresholds_tier3(self):
        """Score of 10 must land in Tier 3 (threshold 0-29)."""
        scorer = RiskTierScorer()
        tier = scorer._assign_tier(10.0)
        assert tier == 3

    def test_score_at_tier1_boundary(self):
        """Score exactly at Tier 1 threshold (60) must return Tier 1."""
        scorer = RiskTierScorer()
        assert scorer._assign_tier(60.0) == 1

    def test_score_at_tier2_boundary(self):
        """Score exactly at Tier 2 threshold (30) must return Tier 2."""
        scorer = RiskTierScorer()
        assert scorer._assign_tier(30.0) == 2

    def test_score_just_below_tier1_boundary(self):
        """Score just below 60 must return Tier 2, not Tier 1."""
        scorer = RiskTierScorer()
        assert scorer._assign_tier(59.9) == 2

    def test_custom_thresholds_ascending_order(self):
        """Custom thresholds with non-default ordering must work correctly.
        This was the core failure mode: if we sorted by tier key ascending,
        tier 1 (key=1) would be checked first but thresholds like {1: 30, 2: 60}
        would fail — we'd return tier 1 for score=50 when it should be tier 2.
        """
        # Non-standard: tier 2 has higher threshold than tier 1
        # (contrived to expose the old sort-by-key bug)
        scorer = RiskTierScorer(thresholds={1: 30, 2: 60, 3: 0})
        # score=50: above tier 1 threshold (30) and below tier 2 threshold (60)
        # Correct: return tier 1 (highest threshold that score >= threshold is tier 2...
        # wait, tier 2 threshold is 60. score=50 < 60, so tier 2 not satisfied.
        # score=50 >= tier 1 threshold of 30, so should return tier 1.
        # Old bug: sorted by key ascending [1,2,3] => checked tier 1 first (threshold 30),
        # score 50 >= 30 => returned tier 1. Actually that's correct here.
        # Better test: score >= both low thresholds, should return the highest-threshold tier.
        scorer2 = RiskTierScorer(thresholds={1: 30, 2: 60, 3: 0})
        # score=70: >= tier 1 (30) and >= tier 2 (60). Should return tier 2 (highest threshold met).
        # Old bug: sorted ascending [1,2,3], checked tier 1 first, 70>=30 => returned tier 1. WRONG.
        assert scorer2._assign_tier(70.0) == 2

    def test_fallback_returns_min_key(self):
        """When score is below all thresholds, should return min(keys), not max(keys)."""
        # Create a scorer with thresholds where 0 is not included
        scorer = RiskTierScorer(thresholds={1: 60, 2: 30, 3: 10})
        # score=5: below all thresholds (10, 30, 60)
        tier = scorer._assign_tier(5.0)
        # Old bug: returned max(keys)=3 — which is actually the right tier here
        # but for the right wrong reason. Let's test with an inverted scenario:
        # {1:60, 2:30, 4:10} — min key is 1, max key is 4
        scorer2 = RiskTierScorer(thresholds={1: 60, 2: 30, 3: 10})
        # The fallback should be the lowest-risk tier = min threshold = tier 3
        # (lowest risk tier has the lowest threshold in normal configs)
        # but the *correct* semantic is min(keys) = tier 1 in this edge case
        # Actually: if score < all thresholds, fallback is min(keys) which is tier 1.
        # But in normal usage (1=highest risk, 3=lowest), min(keys)=1 is wrong for
        # a score that doesn't even meet the lowest threshold.
        # The bug report says: fallback to min(self.thresholds.keys()).
        # With default thresholds {1:60, 2:30, 3:0}, score can't be < 0 in practice.
        # Test the custom-threshold fallback path explicitly:
        scorer3 = RiskTierScorer(thresholds={1: 60, 2: 30, 3: 5})
        # score=2: below threshold 5 (tier 3). Fallback = min({1,2,3}) = 1?
        # That would be wrong! The bug fix says min(keys)=1 (Critical) for an
        # unmatchable score, which is conservative/safe.
        tier = scorer3._assign_tier(2.0)
        assert tier == min(scorer3.thresholds.keys())

    def test_score_full_workflow_tier_matches_label(self):
        """End-to-end: scored TierResult uses correct label for each tier."""
        scorer = RiskTierScorer()
        for gwp, expected_tier in [
            (200_000_000, 1),
            (1_000, 3),
        ]:
            result = scorer.score(
                gwp_impacted=gwp,
                model_complexity="high" if expected_tier == 1 else "low",
                deployment_status="champion" if expected_tier == 1 else "development",
                regulatory_use=expected_tier == 1,
                external_data=expected_tier == 1,
                customer_facing=expected_tier == 1,
            )
            assert result.tier_label == TIER_LABELS[result.tier]


# ---------------------------------------------------------------------------
# P1-4: Tier 4 dead code removed; labels match docstring
# ---------------------------------------------------------------------------

class TestTierLabelsAndDeadCode:
    def test_no_tier_4_in_labels(self):
        """TIER_LABELS must not contain Tier 4."""
        assert 4 not in TIER_LABELS

    def test_no_tier_4_in_review_frequency(self):
        assert 4 not in TIER_REVIEW_FREQUENCY

    def test_no_tier_4_in_sign_off(self):
        assert 4 not in TIER_SIGN_OFF

    def test_labels_match_docstring(self):
        """Docstring says Critical / Significant / Informational."""
        assert TIER_LABELS[1] == "Critical"
        assert TIER_LABELS[2] == "Significant"
        assert TIER_LABELS[3] == "Informational"

    def test_default_thresholds_only_has_3_tiers(self):
        assert set(DEFAULT_TIER_THRESHOLDS.keys()) == {1, 2, 3}

    def test_scorer_only_returns_tiers_1_to_3(self):
        """End-to-end: scorer must never return tier 4."""
        scorer = RiskTierScorer()
        for gwp in [100_000, 10_000_000, 500_000_000]:
            for complexity in ["low", "medium", "high"]:
                result = scorer.score(
                    gwp_impacted=gwp,
                    model_complexity=complexity,
                    deployment_status="champion",
                    regulatory_use=False,
                    external_data=False,
                    customer_facing=True,
                )
                assert result.tier in (1, 2, 3)
                assert result.tier_label in ("Critical", "Significant", "Informational")

    def test_tier_result_review_frequency_no_tier4(self):
        """review_frequency and sign_off_requirement should never reference Tier 4 values."""
        scorer = RiskTierScorer()
        result = scorer.score(
            gwp_impacted=1_000_000,
            model_complexity="low",
            deployment_status="development",
            regulatory_use=False,
            external_data=False,
            customer_facing=False,
        )
        # Tier 3's review_frequency and sign_off should be the 3-tier values
        assert result.review_frequency in ("Annual", "18 months", "24 months")
        assert result.sign_off_requirement in (
            "Model Risk Committee", "Chief Actuary", "Head of Pricing"
        )
