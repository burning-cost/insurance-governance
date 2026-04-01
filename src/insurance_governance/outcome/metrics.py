"""
PriceValueMetrics and ClaimsMetrics — Consumer Duty outcome tests.

These classes take plain numpy arrays (or polars Series castable to numpy)
and return OutcomeResult objects. No polars DataFrames at the metric level:
that coupling belongs in the framework facade, which handles segmentation
and column extraction.

Design decisions:
- We accept arrays not DataFrames so each metric function is testable in
  isolation without a full policy dataset.
- Thresholds have documented sources: GIPP guidance, FCA fair value
  frameworks, Lloyd's market oversight standards.
- All ratio-based tests use max/min median across segments, not pairwise,
  to avoid the multiple-comparisons problem in large segment matrices.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from insurance_governance.validation.results import Severity

from .results import OutcomeResult


class PriceValueMetrics:
    """
    Price and value outcome tests for FCA Consumer Duty (PRIN 2A.3).

    All methods return OutcomeResult objects.
    """

    # ------------------------------------------------------------------
    # Fair value ratio
    # ------------------------------------------------------------------

    @staticmethod
    def fair_value_ratio(
        premiums: Sequence[float],
        claims_paid: Sequence[float],
        expenses: Sequence[float],
        period: str,
        segment: str | None = None,
        threshold: float = 0.70,
    ) -> OutcomeResult:
        """
        Test whether the portfolio claims ratio meets a fair value threshold.

        The claims ratio is defined as total claims paid divided by total
        premiums. The FCA's fair value framework does not mandate a specific
        ratio, but a ratio below 0.70 is a common trigger for a fair value
        review in personal lines. Expenses are passed for transparency in the
        details string but do not alter the pass/fail logic — the intent is
        that this threshold represents the pure claims element.

        Parameters
        ----------
        premiums:
            Gross written premiums for the period.
        claims_paid:
            Total claims paid (settled) for the period, matched to the same
            policy cohort.
        expenses:
            Total expenses attributed to the product.
        period:
            Reporting period label, e.g. ``'2025-Q4'``.
        segment:
            Optional segment label for segment-level reporting.
        threshold:
            Minimum acceptable claims ratio. Default 0.70.

        Returns
        -------
        OutcomeResult
        """
        p = np.asarray(premiums, dtype=float)
        c = np.asarray(claims_paid, dtype=float)
        e = np.asarray(expenses, dtype=float)

        total_premium = float(p.sum())
        total_claims = float(c.sum())
        total_expenses = float(e.sum())

        if total_premium <= 0:
            return OutcomeResult(
                outcome="price_value",
                test_name="fair_value_ratio",
                passed=False,
                metric_value=None,
                threshold=threshold,
                period=period,
                segment=segment,
                details="Cannot compute fair value ratio: total premium is zero or negative.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check data — zero or negative premium sum is not valid."],
            )

        claims_ratio = total_claims / total_premium
        expense_ratio = total_expenses / total_premium
        passed = claims_ratio >= threshold

        details = (
            f"Claims ratio: {claims_ratio:.3f} (threshold: {threshold:.2f}). "
            f"Expense ratio: {expense_ratio:.3f}. "
            f"Total premium: £{total_premium:,.0f}. "
            f"Total claims: £{total_claims:,.0f}."
        )

        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                "Commission a fair value review under PRIN 2A.3.",
                "Assess whether product charges are proportionate to the benefits delivered.",
                "Consider pricing adjustments or enhanced product benefits.",
                "Report outcome to Consumer Duty board champion.",
            ]

        return OutcomeResult(
            outcome="price_value",
            test_name="fair_value_ratio",
            passed=passed,
            metric_value=round(claims_ratio, 6),
            threshold=threshold,
            period=period,
            segment=segment,
            details=details,
            severity=Severity.CRITICAL if not passed else Severity.INFO,
            corrective_actions=corrective_actions,
            extra={
                "total_premium": total_premium,
                "total_claims": total_claims,
                "total_expenses": total_expenses,
                "expense_ratio": round(expense_ratio, 6),
            },
        )

    # ------------------------------------------------------------------
    # Price dispersion by segment
    # ------------------------------------------------------------------

    @staticmethod
    def price_dispersion_by_segment(
        premiums: Sequence[float],
        segment_labels: Sequence[str],
        period: str,
        max_ratio: float = 1.50,
    ) -> list[OutcomeResult]:
        """
        Test whether premium dispersion across segments is within tolerance.

        Computes the median premium for each segment, then checks that the
        max/min median ratio does not exceed ``max_ratio``. A ratio above
        1.50 suggests segments are priced materially differently in a way
        that warrants investigation.

        This is a portfolio-wide dispersion check — it is not a fairness
        test for protected characteristics. Use the discrimination module
        for Equality Act protected groups.

        Parameters
        ----------
        premiums:
            Premium for each policy.
        segment_labels:
            Segment label for each policy.
        period:
            Reporting period.
        max_ratio:
            Maximum acceptable max/min median ratio. Default 1.50.

        Returns
        -------
        list[OutcomeResult]
            One result per segment (individual segment median), plus one
            summary result for the overall dispersion check.
        """
        p = np.asarray(premiums, dtype=float)
        labels = np.asarray(segment_labels)
        unique_segments = np.unique(labels)

        if len(unique_segments) < 2:
            return [
                OutcomeResult(
                    outcome="price_value",
                    test_name="price_dispersion_by_segment",
                    passed=True,
                    metric_value=None,
                    threshold=max_ratio,
                    period=period,
                    details="Only one segment present — dispersion check not applicable.",
                    severity=Severity.INFO,
                )
            ]

        medians: dict[str, float] = {}
        for seg in unique_segments:
            mask = labels == seg
            medians[str(seg)] = float(np.median(p[mask]))

        max_median = max(medians.values())
        min_median = min(medians.values())

        if min_median <= 0:
            return [
                OutcomeResult(
                    outcome="price_value",
                    test_name="price_dispersion_by_segment",
                    passed=False,
                    metric_value=None,
                    threshold=max_ratio,
                    period=period,
                    details="Cannot compute dispersion ratio: minimum segment median premium is zero or negative.",
                    severity=Severity.CRITICAL,
                    corrective_actions=["Check data — zero or negative premiums present."],
                )
            ]

        ratio = max_median / min_median
        passed = ratio <= max_ratio

        results = []
        for seg, median_val in sorted(medians.items()):
            results.append(
                OutcomeResult(
                    outcome="price_value",
                    test_name="price_dispersion_by_segment",
                    passed=True,  # per-segment results are informational
                    metric_value=round(median_val, 2),
                    threshold=None,
                    period=period,
                    segment=seg,
                    details=f"Median premium for segment '{seg}': £{median_val:,.2f}.",
                    severity=Severity.INFO,
                    extra={"segment_count": int((labels == seg).sum())},
                )
            )

        max_seg = max(medians, key=lambda k: medians[k])
        min_seg = min(medians, key=lambda k: medians[k])
        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                f"Investigate pricing basis for segments '{max_seg}' and '{min_seg}'.",
                "Document justification for premium dispersion in fair value assessment.",
                "Escalate to Consumer Duty board champion if no commercial rationale exists.",
            ]

        results.append(
            OutcomeResult(
                outcome="price_value",
                test_name="price_dispersion_summary",
                passed=passed,
                metric_value=round(ratio, 4),
                threshold=max_ratio,
                period=period,
                details=(
                    f"Max/min median premium ratio across {len(unique_segments)} segments: "
                    f"{ratio:.3f} (threshold: {max_ratio:.2f}). "
                    f"Highest median: '{max_seg}' at £{max_median:,.2f}. "
                    f"Lowest median: '{min_seg}' at £{min_median:,.2f}."
                ),
                severity=Severity.WARNING if not passed else Severity.INFO,
                corrective_actions=corrective_actions,
                extra={"segment_medians": {k: round(v, 2) for k, v in medians.items()}},
            )
        )

        return results

    # ------------------------------------------------------------------
    # Renewal vs new business gap (GIPP price-walking check)
    # ------------------------------------------------------------------

    @staticmethod
    def renewal_vs_new_business_gap(
        renewal_premiums: Sequence[float],
        new_business_premiums: Sequence[float],
        exposure: Sequence[float],
        period: str,
        threshold_pct: float = 5.0,
    ) -> OutcomeResult:
        """
        Test for GIPP price-walking: renewal premiums should not exceed
        equivalent new business premiums by more than a de minimis amount.

        The FCA's General Insurance Pricing Practices rules (PS21/5, effective
        January 2022) require that renewal premiums are no higher than the
        equivalent new business price. This test checks whether the
        exposure-weighted mean renewal premium exceeds the exposure-weighted
        mean new business premium by more than ``threshold_pct`` percent.

        Parameters
        ----------
        renewal_premiums:
            Premiums for renewing policyholders.
        new_business_premiums:
            Premiums for new business policyholders (matched risk basis).
        exposure:
            Exposure weight for each renewal policy. Typically earned
            exposure or policy count (use ones for unweighted).
        period:
            Reporting period.
        threshold_pct:
            Maximum acceptable percentage by which renewal exceeds new
            business. Default 5.0% (de minimis tolerance).

        Returns
        -------
        OutcomeResult
        """
        r = np.asarray(renewal_premiums, dtype=float)
        n = np.asarray(new_business_premiums, dtype=float)
        w = np.asarray(exposure, dtype=float)

        if len(r) == 0 or len(n) == 0:
            return OutcomeResult(
                outcome="price_value",
                test_name="renewal_vs_new_business_gap",
                passed=False,
                period=period,
                details="Cannot compute renewal gap: empty premium arrays.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check data pipeline — empty renewal or new business arrays."],
            )

        total_exposure = float(w.sum())
        if total_exposure <= 0:
            return OutcomeResult(
                outcome="price_value",
                test_name="renewal_vs_new_business_gap",
                passed=False,
                period=period,
                details="Cannot compute renewal gap: total exposure is zero or negative.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check exposure values — zero total exposure."],
            )

        weighted_renewal = float(np.average(r, weights=w))
        mean_nb = float(np.mean(n))

        if mean_nb <= 0:
            return OutcomeResult(
                outcome="price_value",
                test_name="renewal_vs_new_business_gap",
                passed=False,
                period=period,
                details="Cannot compute renewal gap: new business mean premium is zero or negative.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check new business premium data."],
            )

        gap_pct = ((weighted_renewal - mean_nb) / mean_nb) * 100.0
        passed = gap_pct <= threshold_pct

        details = (
            f"Exposure-weighted renewal premium: £{weighted_renewal:,.2f}. "
            f"Mean new business premium: £{mean_nb:,.2f}. "
            f"Gap: {gap_pct:+.2f}% (threshold: {threshold_pct:.1f}%)."
        )

        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                "Audit renewal pricing model for price-walking relative to new business equivalent.",
                "Review compliance with PS21/5 (GIPP) renewal pricing obligations.",
                "Identify and remediate affected renewal cohorts.",
                "Notify affected customers if overcharging is confirmed.",
            ]

        return OutcomeResult(
            outcome="price_value",
            test_name="renewal_vs_new_business_gap",
            passed=passed,
            metric_value=round(gap_pct, 4),
            threshold=threshold_pct,
            period=period,
            details=details,
            severity=Severity.CRITICAL if not passed else Severity.INFO,
            corrective_actions=corrective_actions,
            extra={
                "weighted_renewal_premium": round(weighted_renewal, 2),
                "mean_new_business_premium": round(mean_nb, 2),
            },
        )


class ClaimsMetrics:
    """
    Claims outcome tests for FCA Consumer Duty (PRIN 2A.4).

    Tests cover settlement adequacy, decline rate disparity, and timeliness.
    All methods return OutcomeResult objects.
    """

    # ------------------------------------------------------------------
    # Settlement value adequacy
    # ------------------------------------------------------------------

    @staticmethod
    def settlement_value_adequacy(
        agreed_settlements: Sequence[float],
        reference_valuations: Sequence[float],
        period: str,
        threshold_ratio: float = 0.95,
    ) -> OutcomeResult:
        """
        Test whether agreed settlements are adequate relative to reference valuations.

        The ratio is mean(agreed) / mean(reference). A ratio below
        ``threshold_ratio`` suggests customers are systematically accepting
        less than the reference value — a Consumer Duty concern.

        Reference valuations might be independent engineer assessments,
        third-party market values, or actuarial reserving estimates.

        Parameters
        ----------
        agreed_settlements:
            Final agreed settlement amounts.
        reference_valuations:
            Independent or reference valuations for the same claims.
        period:
            Reporting period.
        threshold_ratio:
            Minimum acceptable settlement/reference ratio. Default 0.95.

        Returns
        -------
        OutcomeResult
        """
        s = np.asarray(agreed_settlements, dtype=float)
        r = np.asarray(reference_valuations, dtype=float)

        if len(s) == 0 or len(r) == 0:
            return OutcomeResult(
                outcome="claims",
                test_name="settlement_value_adequacy",
                passed=False,
                period=period,
                details="Cannot compute settlement adequacy: empty arrays.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check claims data pipeline."],
            )

        if len(s) != len(r):
            return OutcomeResult(
                outcome="claims",
                test_name="settlement_value_adequacy",
                passed=False,
                period=period,
                details=(
                    f"Array length mismatch: agreed_settlements has {len(s)} rows, "
                    f"reference_valuations has {len(r)} rows."
                ),
                severity=Severity.CRITICAL,
                corrective_actions=["Ensure agreed_settlements and reference_valuations are matched 1:1."],
            )

        mean_ref = float(np.mean(r))
        if mean_ref <= 0:
            return OutcomeResult(
                outcome="claims",
                test_name="settlement_value_adequacy",
                passed=False,
                period=period,
                details="Cannot compute ratio: mean reference valuation is zero or negative.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check reference valuation data."],
            )

        mean_settlement = float(np.mean(s))
        ratio = mean_settlement / mean_ref
        passed = ratio >= threshold_ratio

        details = (
            f"Mean agreed settlement: £{mean_settlement:,.2f}. "
            f"Mean reference valuation: £{mean_ref:,.2f}. "
            f"Ratio: {ratio:.4f} (threshold: {threshold_ratio:.2f}). "
            f"Based on {len(s)} claims."
        )

        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                "Review claims settlement practices for systematic undervaluation.",
                "Audit claims handler incentives and settlement negotiation protocols.",
                "Consider proactive re-offer to recently settled claims below reference value.",
                "Escalate to Consumer Duty board champion and claims director.",
            ]

        return OutcomeResult(
            outcome="claims",
            test_name="settlement_value_adequacy",
            passed=passed,
            metric_value=round(ratio, 6),
            threshold=threshold_ratio,
            period=period,
            details=details,
            severity=Severity.CRITICAL if not passed else Severity.INFO,
            corrective_actions=corrective_actions,
            extra={
                "mean_agreed_settlement": round(mean_settlement, 2),
                "mean_reference_valuation": round(mean_ref, 2),
                "claim_count": len(s),
            },
        )

    # ------------------------------------------------------------------
    # Decline rate by segment
    # ------------------------------------------------------------------

    @staticmethod
    def decline_rate_by_segment(
        outcomes: Sequence[int],
        segment_labels: Sequence[str],
        period: str,
        max_disparity: float = 1.50,
    ) -> list[OutcomeResult]:
        """
        Test whether claims decline rates are disproportionate across segments.

        A binary outcome of 1 = declined, 0 = paid/accepted. This test checks
        that no segment has a decline rate more than ``max_disparity`` times
        the lowest segment's decline rate.

        This is a Consumer Duty outcome test — a large disparity in decline
        rates across customer groups (e.g. direct vs aggregator, standard vs
        vulnerable) warrants investigation regardless of whether a protected
        characteristic is involved.

        Parameters
        ----------
        outcomes:
            Binary array: 1 = claim declined, 0 = claim accepted/paid.
        segment_labels:
            Segment label for each claim.
        period:
            Reporting period.
        max_disparity:
            Maximum acceptable ratio of highest to lowest decline rate.
            Default 1.50.

        Returns
        -------
        list[OutcomeResult]
            One result per segment (informational), plus one summary result.
        """
        o = np.asarray(outcomes, dtype=int)
        labels = np.asarray(segment_labels)
        unique_segments = np.unique(labels)

        if len(unique_segments) < 2:
            return [
                OutcomeResult(
                    outcome="claims",
                    test_name="decline_rate_by_segment",
                    passed=True,
                    metric_value=None,
                    threshold=max_disparity,
                    period=period,
                    details="Only one segment present — disparity check not applicable.",
                    severity=Severity.INFO,
                )
            ]

        decline_rates: dict[str, float] = {}
        segment_counts: dict[str, int] = {}
        for seg in unique_segments:
            mask = labels == seg
            count = int(mask.sum())
            rate = float(o[mask].mean()) if count > 0 else 0.0
            decline_rates[str(seg)] = rate
            segment_counts[str(seg)] = count

        max_rate = max(decline_rates.values())
        min_rate = min(decline_rates.values())

        results = []
        for seg in sorted(decline_rates):
            rate = decline_rates[seg]
            results.append(
                OutcomeResult(
                    outcome="claims",
                    test_name="decline_rate_by_segment",
                    passed=True,
                    metric_value=round(rate, 6),
                    threshold=None,
                    period=period,
                    segment=seg,
                    details=(
                        f"Decline rate for segment '{seg}': {rate:.2%} "
                        f"({segment_counts[seg]} claims)."
                    ),
                    severity=Severity.INFO,
                    extra={"claim_count": segment_counts[seg]},
                )
            )

        if min_rate <= 0:
            results.append(
                OutcomeResult(
                    outcome="claims",
                    test_name="decline_rate_disparity_summary",
                    passed=True,
                    metric_value=None,
                    threshold=max_disparity,
                    period=period,
                    details=(
                        "Cannot compute disparity ratio: one or more segments have zero "
                        "decline rate. All segments appear to have no declines."
                    ),
                    severity=Severity.INFO,
                )
            )
            return results

        ratio = max_rate / min_rate
        passed = ratio <= max_disparity

        max_seg = max(decline_rates, key=lambda k: decline_rates[k])
        min_seg = min(decline_rates, key=lambda k: decline_rates[k])

        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                f"Investigate claims decline criteria for segments '{max_seg}' and '{min_seg}'.",
                "Review claims handlers' decision-making for consistency across segments.",
                "Assess whether decline criteria are applied proportionately.",
                "Consider root cause analysis of declined claims in highest-rate segment.",
            ]

        results.append(
            OutcomeResult(
                outcome="claims",
                test_name="decline_rate_disparity_summary",
                passed=passed,
                metric_value=round(ratio, 4),
                threshold=max_disparity,
                period=period,
                details=(
                    f"Decline rate ratio (highest/lowest) across {len(unique_segments)} "
                    f"segments: {ratio:.3f} (threshold: {max_disparity:.2f}). "
                    f"Highest rate: '{max_seg}' at {max_rate:.2%}. "
                    f"Lowest rate: '{min_seg}' at {min_rate:.2%}."
                ),
                severity=Severity.WARNING if not passed else Severity.INFO,
                corrective_actions=corrective_actions,
                extra={"segment_decline_rates": {k: round(v, 6) for k, v in decline_rates.items()}},
            )
        )

        return results

    # ------------------------------------------------------------------
    # Timeliness SLA
    # ------------------------------------------------------------------

    @staticmethod
    def timeliness_sla(
        days_to_settlement: Sequence[float],
        period: str,
        sla_days: int = 5,
    ) -> OutcomeResult:
        """
        Test whether claims are settled within the agreed SLA.

        Reports both the percentage meeting the SLA and the mean/median
        settlement time. The Consumer Duty rules require firms to deliver
        good outcomes for customers including prompt claims handling.

        Parameters
        ----------
        days_to_settlement:
            Number of days from claim notification to settlement for each claim.
        period:
            Reporting period.
        sla_days:
            Target SLA in days. Claims settled within this are SLA-compliant.
            Default 5 (a common fast-track motor SLA). For complex claims
            use a higher value.

        Returns
        -------
        OutcomeResult
        """
        d = np.asarray(days_to_settlement, dtype=float)

        if len(d) == 0:
            return OutcomeResult(
                outcome="claims",
                test_name="timeliness_sla",
                passed=False,
                period=period,
                details="Cannot compute timeliness: no claims data.",
                severity=Severity.CRITICAL,
                corrective_actions=["Check claims data pipeline."],
            )

        within_sla = float((d <= sla_days).mean())
        mean_days = float(np.mean(d))
        median_days = float(np.median(d))
        p90_days = float(np.percentile(d, 90))

        # Pass if 80%+ of claims are within SLA — this threshold is configurable
        # via the sla_days parameter but the 80% compliance floor is fixed.
        # Teams with tighter SLA commitments should set sla_days accordingly.
        compliance_floor = 0.80
        passed = within_sla >= compliance_floor

        details = (
            f"Claims within {sla_days}-day SLA: {within_sla:.1%} "
            f"(floor: {compliance_floor:.0%}). "
            f"Mean settlement: {mean_days:.1f} days. "
            f"Median: {median_days:.1f} days. "
            f"90th percentile: {p90_days:.1f} days. "
            f"{len(d)} claims."
        )

        corrective_actions: list[str] = []
        if not passed:
            corrective_actions = [
                f"Investigate claims taking more than {sla_days} days to settle.",
                "Review claims handler capacity and triage processes.",
                "Identify bottlenecks in the settlement workflow.",
                "Consider proactive customer contact for claims exceeding SLA.",
            ]

        return OutcomeResult(
            outcome="claims",
            test_name="timeliness_sla",
            passed=passed,
            metric_value=round(within_sla, 6),
            threshold=compliance_floor,
            period=period,
            details=details,
            severity=Severity.WARNING if not passed else Severity.INFO,
            corrective_actions=corrective_actions,
            extra={
                "mean_days": round(mean_days, 2),
                "median_days": round(median_days, 2),
                "p90_days": round(p90_days, 2),
                "sla_days": sla_days,
                "claim_count": len(d),
                "within_sla_count": int((d <= sla_days).sum()),
            },
        )
