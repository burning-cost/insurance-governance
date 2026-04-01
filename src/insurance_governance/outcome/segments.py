"""
CustomerSegment and SegmentComparison for Consumer Duty outcome testing.

Segments are not demographic slices for fairness purposes — they are
business-relevant groupings for FCA outcome monitoring: new versus renewal,
direct versus aggregator, standard versus vulnerable customers.

Design: filter_fn takes a polars DataFrame and returns a boolean Series.
This keeps the segment definition close to the data schema rather than
hardcoding column names into the metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl


@dataclass
class CustomerSegment:
    """
    A named customer segment with a boolean filter predicate.

    Parameters
    ----------
    name:
        Human-readable segment label, e.g. ``'Renewal'`` or ``'Vulnerable'``.
    filter_fn:
        Callable that accepts a polars DataFrame and returns a boolean Series.
        Rows where the Series is True belong to this segment.
    is_vulnerable:
        Whether this segment is a vulnerable customer group. Vulnerable
        segment results are separated in the report and given higher scrutiny.

    Examples
    --------
    ::

        renewal_seg = CustomerSegment(
            name="Renewal",
            filter_fn=lambda df: df["policy_type"] == "renewal",
        )
        vulnerable_seg = CustomerSegment(
            name="Older Customers (65+)",
            filter_fn=lambda df: df["age"] >= 65,
            is_vulnerable=True,
        )
    """

    name: str
    filter_fn: Callable[[pl.DataFrame], pl.Series]
    is_vulnerable: bool = False

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the subset of df belonging to this segment."""
        mask = self.filter_fn(df)
        return df.filter(mask)

    def count(self, df: pl.DataFrame) -> int:
        """Return the number of rows in df belonging to this segment."""
        return self.apply(df).height


@dataclass
class SegmentComparison:
    """
    A pairwise comparison between two segments on a scalar metric.

    Used to record the outcome of disparity tests — whether the ratio
    of a metric between two groups exceeds an acceptable threshold.

    Parameters
    ----------
    segment_a:
        Label for the reference segment.
    segment_b:
        Label for the comparison segment.
    metric_name:
        What is being compared, e.g. ``'median_premium'`` or ``'decline_rate'``.
    value_a:
        Metric value for segment A.
    value_b:
        Metric value for segment B.
    ratio:
        value_a / value_b (or max/min if comparing across more than two).
    threshold:
        Maximum acceptable ratio before a disparity finding is raised.
    passed:
        Whether the ratio is within threshold.
    """

    segment_a: str
    segment_b: str
    metric_name: str
    value_a: float
    value_b: float
    ratio: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "metric_name": self.metric_name,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "ratio": self.ratio,
            "threshold": self.threshold,
            "passed": self.passed,
        }
