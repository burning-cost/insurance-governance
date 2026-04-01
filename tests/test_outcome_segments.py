"""Tests for CustomerSegment and SegmentComparison."""
import polars as pl
import pytest
from insurance_governance.outcome.segments import CustomerSegment, SegmentComparison


def make_df():
    return pl.DataFrame({
        "age": [25, 35, 45, 55, 65, 75],
        "policy_type": ["new", "renewal", "new", "renewal", "new", "renewal"],
        "premium": [300.0, 280.0, 320.0, 295.0, 340.0, 310.0],
    })


# --- CustomerSegment ---

def test_segment_apply_filter():
    df = make_df()
    seg = CustomerSegment(
        name="Renewal",
        filter_fn=lambda df: df["policy_type"] == "renewal",
    )
    result = seg.apply(df)
    assert result.height == 3
    assert all(result["policy_type"] == "renewal")


def test_segment_count():
    df = make_df()
    seg = CustomerSegment(
        name="Older",
        filter_fn=lambda df: df["age"] >= 60,
    )
    assert seg.count(df) == 2


def test_segment_is_vulnerable_default_false():
    seg = CustomerSegment(name="Standard", filter_fn=lambda df: df["age"] < 65)
    assert seg.is_vulnerable is False


def test_segment_is_vulnerable_set():
    seg = CustomerSegment(
        name="Vulnerable",
        filter_fn=lambda df: df["age"] >= 65,
        is_vulnerable=True,
    )
    assert seg.is_vulnerable is True


def test_segment_apply_empty_result():
    df = make_df()
    seg = CustomerSegment(
        name="Nobody",
        filter_fn=lambda df: df["age"] > 200,
    )
    result = seg.apply(df)
    assert result.height == 0


def test_segment_apply_all_rows():
    df = make_df()
    seg = CustomerSegment(
        name="All",
        filter_fn=lambda df: pl.Series([True] * df.height),
    )
    result = seg.apply(df)
    assert result.height == df.height


# --- SegmentComparison ---

def test_segment_comparison_to_dict():
    comp = SegmentComparison(
        segment_a="Renewal",
        segment_b="New Business",
        metric_name="median_premium",
        value_a=310.0,
        value_b=295.0,
        ratio=1.051,
        threshold=1.50,
        passed=True,
    )
    d = comp.to_dict()
    assert d["segment_a"] == "Renewal"
    assert d["segment_b"] == "New Business"
    assert d["metric_name"] == "median_premium"
    assert d["value_a"] == 310.0
    assert d["passed"] is True


def test_segment_comparison_failed():
    comp = SegmentComparison(
        segment_a="Older",
        segment_b="Younger",
        metric_name="decline_rate",
        value_a=0.25,
        value_b=0.10,
        ratio=2.5,
        threshold=1.50,
        passed=False,
    )
    assert comp.passed is False
    assert comp.ratio == 2.5


def test_segment_comparison_all_dict_keys():
    comp = SegmentComparison(
        segment_a="A", segment_b="B", metric_name="m",
        value_a=1.0, value_b=2.0, ratio=0.5, threshold=1.5, passed=True,
    )
    d = comp.to_dict()
    for key in ("segment_a", "segment_b", "metric_name", "value_a", "value_b", "ratio", "threshold", "passed"):
        assert key in d
