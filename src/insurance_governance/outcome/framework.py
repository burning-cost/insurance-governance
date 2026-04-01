"""
OutcomeTestingFramework — high-level facade for FCA Consumer Duty outcome testing.

This is the entry point for pricing teams. You provide a model card, a policy
DataFrame, and the column names. The framework handles segmentation, runs the
applicable metric tests, and produces a report.

Design decisions:
- The framework uses polars for data handling, consistent with the rest of the library.
- Segmentation is applied before metric computation. Each segment produces its own
  set of results; portfolio-wide results are always computed regardless.
- Column arguments are optional: passing None skips the corresponding test suite.
  This lets teams run only the tests relevant to their data.
- extra_results lets teams inject custom tests (e.g. complaints rate, NPS) using
  the same OutcomeResult type, so they appear in the unified report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from insurance_governance.mrm.model_card import ModelCard as MRMModelCard
from insurance_governance.validation.results import RAGStatus

from .metrics import ClaimsMetrics, PriceValueMetrics
from .report import OutcomeTestingReport
from .results import OutcomeResult, OutcomeSuite, _compute_outcome_rag
from .segments import CustomerSegment


class OutcomeTestingFramework:
    """
    High-level facade for FCA Consumer Duty outcome testing.

    Run price-value and claims outcome tests against live policy data,
    optionally broken down by customer segments, and produce board-ready
    HTML and JSON reports.

    Parameters
    ----------
    model_card:
        MRM model card for the model under review. Identifies the model
        in the report header.
    policy_data:
        Polars DataFrame of policy records for the reporting period.
        Each row is one policy.
    period:
        Reporting period label, e.g. ``'2025-Q4'`` or ``'2024-H2'``.
    price_col:
        Column name in ``policy_data`` containing the gross written premium.
    claim_amount_col:
        Column containing claims paid. If None, fair value tests are skipped.
    claim_outcome_col:
        Column containing binary decline indicator (1=declined, 0=paid).
        Used for decline rate disparity tests.
    days_to_settlement_col:
        Column containing days from notification to settlement.
        Used for timeliness SLA tests.
    expenses_col:
        Column containing expenses attributed to each policy. If None,
        expenses are assumed to be zero for fair value ratio tests.
    reference_valuation_col:
        Column containing independent reference valuations per claim.
        Used for settlement adequacy tests.
    renewal_indicator_col:
        Column containing a boolean/integer flag: 1 = renewal, 0 = new business.
        Used for the GIPP price-walking check.
    exposure_col:
        Column containing exposure weight. Used for weighted renewal gap test.
        If None, unweighted (exposure = 1 for all policies) is used.
    customer_segments:
        List of CustomerSegment objects. Each segment produces a separate set
        of results alongside portfolio-wide results.
    extra_results:
        Additional OutcomeResult objects to include in the report (e.g.
        complaints rate, NPS score, manual qualitative assessments).

    Examples
    --------
    ::

        from insurance_governance import MRMModelCard
        from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
        import polars as pl

        card = MRMModelCard(model_id="motor-v3", model_name="Motor Frequency v3", version="3.0.0")
        df = pl.read_parquet("policies_2025q4.parquet")

        renewal_seg = CustomerSegment(
            name="Renewal",
            filter_fn=lambda df: df["policy_type"] == "renewal",
        )

        framework = OutcomeTestingFramework(
            model_card=card,
            policy_data=df,
            period="2025-Q4",
            price_col="gross_premium",
            claim_amount_col="claims_paid",
            renewal_indicator_col="is_renewal",
            customer_segments=[renewal_seg],
        )

        results = framework.run()
        framework.generate("outcome_report.html")
    """

    def __init__(
        self,
        model_card: MRMModelCard,
        policy_data: pl.DataFrame,
        period: str,
        price_col: str,
        claim_amount_col: str | None = None,
        claim_outcome_col: str | None = None,
        days_to_settlement_col: str | None = None,
        expenses_col: str | None = None,
        reference_valuation_col: str | None = None,
        renewal_indicator_col: str | None = None,
        exposure_col: str | None = None,
        customer_segments: list[CustomerSegment] | None = None,
        extra_results: list[OutcomeResult] | None = None,
    ) -> None:
        self._model_card = model_card
        self._policy_data = policy_data
        self._period = period
        self._price_col = price_col
        self._claim_amount_col = claim_amount_col
        self._claim_outcome_col = claim_outcome_col
        self._days_to_settlement_col = days_to_settlement_col
        self._expenses_col = expenses_col
        self._reference_valuation_col = reference_valuation_col
        self._renewal_indicator_col = renewal_indicator_col
        self._exposure_col = exposure_col
        self._customer_segments = customer_segments or []
        self._extra_results = extra_results or []
        self._results: list[OutcomeResult] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> list[OutcomeResult]:
        """
        Run all applicable outcome tests and return the results.

        Results are cached after the first call — subsequent calls to
        ``run()`` return the cached list without recomputing.

        Returns
        -------
        list[OutcomeResult]
        """
        if self._results is not None:
            return self._results

        results: list[OutcomeResult] = []

        # Portfolio-wide tests
        results.extend(self._run_price_value_tests(self._policy_data, segment=None))
        results.extend(self._run_claims_tests(self._policy_data, segment=None))

        # Segment-level tests
        for seg in self._customer_segments:
            seg_data = seg.apply(self._policy_data)
            if seg_data.height == 0:
                continue
            results.extend(
                self._run_price_value_tests(seg_data, segment=seg.name)
            )
            results.extend(
                self._run_claims_tests(seg_data, segment=seg.name)
            )

        results.extend(self._extra_results)

        self._results = results
        return results

    def get_rag_status(self) -> RAGStatus:
        """
        Return the overall RAG status for this outcome review.

        Runs the tests if they have not already been run.
        """
        return _compute_outcome_rag(self.run())

    def generate(self, path: str | Path) -> Path:
        """
        Generate the HTML outcome testing report and write it to disk.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        Path
            Resolved path to the written HTML file.
        """
        report = self._build_report()
        return report.write_html(path)

    def to_json(self, path: str | Path) -> Path:
        """
        Write the JSON sidecar for audit trail ingestion.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        Path
            Resolved path to the written JSON file.
        """
        report = self._build_report()
        return report.write_json(path)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full outcome suite to a plain dict."""
        return self._build_report().to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self) -> OutcomeTestingReport:
        return OutcomeTestingReport(
            model_card=self._model_card,
            results=self.run(),
            period=self._period,
        )

    def _get_col(self, df: pl.DataFrame, col: str | None) -> list[float] | None:
        """Extract a column as a Python list, or None if col is None."""
        if col is None or col not in df.columns:
            return None
        return df[col].cast(pl.Float64).to_list()

    def _get_col_int(self, df: pl.DataFrame, col: str | None) -> list[int] | None:
        """Extract an integer column as a Python list, or None if col is None."""
        if col is None or col not in df.columns:
            return None
        return df[col].cast(pl.Int32).to_list()

    def _run_price_value_tests(
        self, df: pl.DataFrame, segment: str | None
    ) -> list[OutcomeResult]:
        results: list[OutcomeResult] = []

        premiums = self._get_col(df, self._price_col)
        if premiums is None:
            return results

        # Fair value ratio
        if self._claim_amount_col is not None:
            claims = self._get_col(df, self._claim_amount_col)
            if claims is not None:
                if self._expenses_col is not None:
                    expenses = self._get_col(df, self._expenses_col) or [0.0] * len(premiums)
                else:
                    expenses = [0.0] * len(premiums)
                results.append(
                    PriceValueMetrics.fair_value_ratio(
                        premiums=premiums,
                        claims_paid=claims,
                        expenses=expenses,
                        period=self._period,
                        segment=segment,
                    )
                )

        # Price dispersion by segment — only portfolio-wide (no sub-segment breakdown)
        if segment is None and self._customer_segments:
            labels = self._build_segment_labels(df)
            if labels is not None:
                results.extend(
                    PriceValueMetrics.price_dispersion_by_segment(
                        premiums=premiums,
                        segment_labels=labels,
                        period=self._period,
                    )
                )

        # Renewal vs new business gap
        if segment is None and self._renewal_indicator_col is not None:
            indicator = self._get_col_int(df, self._renewal_indicator_col)
            if indicator is not None:
                import numpy as np
                ind_arr = np.array(indicator)
                renewal_mask = ind_arr == 1
                nb_mask = ind_arr == 0
                p_arr = np.array(premiums)
                if renewal_mask.sum() > 0 and nb_mask.sum() > 0:
                    if self._exposure_col is not None:
                        exposure_all = self._get_col(df, self._exposure_col)
                        if exposure_all is not None:
                            import numpy as np
                            exp_arr = np.array(exposure_all)
                            renewal_exp = exp_arr[renewal_mask].tolist()
                        else:
                            renewal_exp = [1.0] * int(renewal_mask.sum())
                    else:
                        renewal_exp = [1.0] * int(renewal_mask.sum())
                    results.append(
                        PriceValueMetrics.renewal_vs_new_business_gap(
                            renewal_premiums=p_arr[renewal_mask].tolist(),
                            new_business_premiums=p_arr[nb_mask].tolist(),
                            exposure=renewal_exp,
                            period=self._period,
                        )
                    )

        return results

    def _run_claims_tests(
        self, df: pl.DataFrame, segment: str | None
    ) -> list[OutcomeResult]:
        results: list[OutcomeResult] = []

        # Settlement adequacy
        if self._claim_amount_col and self._reference_valuation_col:
            settlements = self._get_col(df, self._claim_amount_col)
            references = self._get_col(df, self._reference_valuation_col)
            if settlements is not None and references is not None:
                r = ClaimsMetrics.settlement_value_adequacy(
                    agreed_settlements=settlements,
                    reference_valuations=references,
                    period=self._period,
                )
                if segment is not None:
                    r.segment = segment
                results.append(r)

        # Decline rate by segment — only portfolio-wide with segment labels
        if segment is None and self._claim_outcome_col and self._customer_segments:
            outcomes = self._get_col_int(df, self._claim_outcome_col)
            if outcomes is not None:
                labels = self._build_segment_labels(df)
                if labels is not None:
                    results.extend(
                        ClaimsMetrics.decline_rate_by_segment(
                            outcomes=outcomes,
                            segment_labels=labels,
                            period=self._period,
                        )
                    )

        # Timeliness SLA
        if self._days_to_settlement_col:
            days = self._get_col(df, self._days_to_settlement_col)
            if days is not None:
                r = ClaimsMetrics.timeliness_sla(
                    days_to_settlement=days,
                    period=self._period,
                )
                if segment is not None:
                    r.segment = segment
                results.append(r)

        return results

    def _build_segment_labels(self, df: pl.DataFrame) -> list[str] | None:
        """
        Build a flat list of segment labels for each row by applying each
        CustomerSegment's filter function. Rows matching no segment are
        labelled 'Other'.
        """
        if not self._customer_segments:
            return None

        labels = ["Other"] * df.height
        for seg in self._customer_segments:
            try:
                mask = seg.filter_fn(df)
                indices = mask.arg_true().to_list()
                for i in indices:
                    labels[i] = seg.name
            except Exception:
                pass

        return labels
