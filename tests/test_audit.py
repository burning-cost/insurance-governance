"""Tests for the insurance_governance.audit subpackage.

Covers:
- ExplainabilityAuditEntry creation, serialisation, and hash verification.
- ExplainabilityAuditLog append, read, verify_chain, and export_period.
- PlainLanguageExplainer output format and edge cases.
- AuditSummaryReport generation (JSON and HTML).
- SHAPExplainer with a mock model (shap package not required).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest.mock as mock
from datetime import datetime, timezone
from pathlib import Path

import pytest

from insurance_governance.audit.entry import (
    DECISION_BASIS_VALUES,
    ExplainabilityAuditEntry,
)
from insurance_governance.audit.log import ExplainabilityAuditLog
from insurance_governance.audit.customer_explanation import (
    PlainLanguageExplainer,
    FactorContribution,
)
from insurance_governance.audit.report import AuditSummaryReport
from insurance_governance.audit.shap_explainer import (
    SHAPExplainer,
    _SHAP_IMPORT_ERROR,
    _SHAP_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(**kwargs) -> ExplainabilityAuditEntry:
    defaults = dict(
        model_id="motor-freq-v3",
        model_version="3.1.0",
        input_features={"driver_age": 32, "ncb_years": 5, "region": "SE"},
        feature_importances={"driver_age": -0.12, "ncb_years": -0.31, "region": 0.08},
        prediction=412.50,
        final_premium=412.50,
        decision_basis="model_output",
    )
    defaults.update(kwargs)
    return ExplainabilityAuditEntry(**defaults)


FEATURE_LABELS = {
    "driver_age": "your age",
    "ncb_years": "your no-claims discount",
    "region": "your postcode area",
}


# ---------------------------------------------------------------------------
# ExplainabilityAuditEntry
# ---------------------------------------------------------------------------

class TestExplainabilityAuditEntryCreation:
    def test_minimal_creation(self):
        entry = _make_entry()
        assert entry.model_id == "motor-freq-v3"
        assert entry.model_version == "3.1.0"
        assert entry.decision_basis == "model_output"
        assert entry.human_reviewed is False
        assert entry.override_applied is False
        assert entry.entry_hash  # auto-computed

    def test_auto_uuid_and_timestamp(self):
        entry = _make_entry()
        # entry_id should be a valid UUID4-ish string
        assert len(entry.entry_id) == 36
        assert "T" in entry.timestamp_utc

    def test_explicit_entry_id(self):
        entry = _make_entry(entry_id="my-custom-id")
        assert entry.entry_id == "my-custom-id"

    def test_all_decision_basis_values(self):
        for basis in DECISION_BASIS_VALUES:
            entry = _make_entry(decision_basis=basis)
            assert entry.decision_basis == basis

    def test_invalid_decision_basis_raises(self):
        with pytest.raises(ValueError, match="decision_basis must be one of"):
            _make_entry(decision_basis="ai_only")

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            _make_entry(model_id="")

    def test_empty_model_version_raises(self):
        with pytest.raises(ValueError, match="model_version cannot be empty"):
            _make_entry(model_version="")

    def test_override_without_reason_raises(self):
        with pytest.raises(ValueError, match="override_reason must be provided"):
            _make_entry(
                override_applied=True,
                override_reason=None,
                decision_basis="human_override",
            )

    def test_override_with_reason_ok(self):
        entry = _make_entry(
            override_applied=True,
            override_reason="Rate too high for renewal customer",
            decision_basis="human_override",
            human_reviewed=True,
            reviewer_id="CF28-sarah.ahmed",
        )
        assert entry.override_applied is True
        assert entry.override_reason == "Rate too high for renewal customer"

    def test_session_id_optional(self):
        entry = _make_entry(session_id="batch-2025-q4-001")
        assert entry.session_id == "batch-2025-q4-001"

    def test_final_premium_none(self):
        entry = _make_entry(final_premium=None)
        assert entry.final_premium is None


class TestExplainabilityAuditEntryHash:
    def test_hash_computed_on_creation(self):
        entry = _make_entry()
        assert len(entry.entry_hash) == 64  # SHA-256 hex digest

    def test_verify_integrity_passes_fresh_entry(self):
        entry = _make_entry()
        assert entry.verify_integrity() is True

    def test_verify_integrity_fails_after_tampering(self):
        entry = _make_entry()
        # Tamper directly
        object.__setattr__(entry, "prediction", 999.99)
        assert entry.verify_integrity() is False

    def test_same_content_same_hash(self):
        e1 = _make_entry(
            entry_id="fixed-id",
            timestamp_utc="2025-01-01T12:00:00+00:00",
        )
        e2 = _make_entry(
            entry_id="fixed-id",
            timestamp_utc="2025-01-01T12:00:00+00:00",
        )
        assert e1.entry_hash == e2.entry_hash

    def test_different_content_different_hash(self):
        e1 = _make_entry(prediction=400.0)
        e2 = _make_entry(prediction=500.0)
        assert e1.entry_hash != e2.entry_hash


class TestExplainabilityAuditEntrySerialisation:
    def test_to_dict_has_all_keys(self):
        entry = _make_entry()
        d = entry.to_dict()
        expected_keys = {
            "entry_id", "model_id", "model_version", "timestamp_utc",
            "session_id", "input_features", "feature_importances", "prediction",
            "final_premium", "human_reviewed", "reviewer_id", "override_applied",
            "override_reason", "decision_basis", "entry_hash",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_roundtrip(self):
        entry = _make_entry()
        d = entry.to_dict()
        entry2 = ExplainabilityAuditEntry.from_dict(d)
        assert entry2.model_id == entry.model_id
        assert entry2.prediction == entry.prediction
        assert entry2.entry_hash == entry.entry_hash

    def test_from_dict_preserves_hash(self):
        entry = _make_entry()
        original_hash = entry.entry_hash
        d = entry.to_dict()
        entry2 = ExplainabilityAuditEntry.from_dict(d)
        assert entry2.entry_hash == original_hash
        assert entry2.verify_integrity() is True

    def test_from_dict_json_roundtrip(self):
        entry = _make_entry(session_id="s123", final_premium=412.50)
        raw = json.dumps(entry.to_dict())
        entry2 = ExplainabilityAuditEntry.from_dict(json.loads(raw))
        assert entry2.session_id == "s123"
        assert entry2.final_premium == 412.50
        assert entry2.verify_integrity() is True


# ---------------------------------------------------------------------------
# ExplainabilityAuditLog
# ---------------------------------------------------------------------------

class TestExplainabilityAuditLog:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._log_path = os.path.join(self._tmpdir, "audit.jsonl")

    def _make_log(self) -> ExplainabilityAuditLog:
        return ExplainabilityAuditLog(
            path=self._log_path,
            model_id="motor-freq-v3",
            model_version="3.1.0",
        )

    def test_creates_empty_file(self):
        log = self._make_log()
        assert Path(self._log_path).exists()
        assert Path(self._log_path).stat().st_size == 0

    def test_append_and_read_single_entry(self):
        log = self._make_log()
        entry = _make_entry()
        log.append(entry)
        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0].model_id == "motor-freq-v3"

    def test_append_multiple_entries(self):
        log = self._make_log()
        for i in range(5):
            log.append(_make_entry(prediction=float(400 + i)))
        entries = log.read_all()
        assert len(entries) == 5
        predictions = [e.prediction for e in entries]
        assert predictions == [400.0, 401.0, 402.0, 403.0, 404.0]

    def test_append_non_entry_raises(self):
        log = self._make_log()
        with pytest.raises(TypeError, match="ExplainabilityAuditEntry"):
            log.append({"not": "an entry"})  # type: ignore

    def test_read_empty_log(self):
        log = self._make_log()
        assert log.read_all() == []

    def test_read_all_preserves_hashes(self):
        log = self._make_log()
        entry = _make_entry()
        log.append(entry)
        loaded = log.read_all()[0]
        assert loaded.verify_integrity() is True

    def test_read_since_filters_correctly(self):
        log = self._make_log()
        early_ts = "2024-06-01T10:00:00+00:00"
        late_ts = "2025-01-15T10:00:00+00:00"
        log.append(_make_entry(timestamp_utc=early_ts))
        log.append(_make_entry(timestamp_utc=late_ts))

        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = log.read_since(cutoff)
        assert len(result) == 1
        assert result[0].timestamp_utc == late_ts

    def test_read_since_returns_all_when_cutoff_before_all(self):
        log = self._make_log()
        for _ in range(3):
            log.append(_make_entry(timestamp_utc="2025-06-01T12:00:00+00:00"))
        cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = log.read_since(cutoff)
        assert len(result) == 3

    def test_read_since_naive_cutoff_treated_as_utc(self):
        log = self._make_log()
        log.append(_make_entry(timestamp_utc="2025-06-01T12:00:00+00:00"))
        cutoff = datetime(2025, 5, 1)  # naive
        result = log.read_since(cutoff)
        assert len(result) == 1

    def test_verify_chain_all_intact(self):
        log = self._make_log()
        for _ in range(3):
            log.append(_make_entry())
        failures = log.verify_chain()
        assert failures == []

    def test_verify_chain_detects_tampered_line(self):
        log = self._make_log()
        entry = _make_entry()
        log.append(entry)

        # Tamper with the file directly
        with open(self._log_path, "r") as fh:
            line = fh.read()
        d = json.loads(line.strip())
        d["prediction"] = 999999.0  # change value, keep original hash
        with open(self._log_path, "w") as fh:
            fh.write(json.dumps(d) + "\n")

        failures = log.verify_chain()
        assert len(failures) == 1
        assert "tampered" in failures[0]["reason"].lower() or "mismatch" in failures[0]["reason"].lower()

    def test_verify_chain_detects_corrupt_json(self):
        log = self._make_log()
        log.append(_make_entry())
        with open(self._log_path, "a") as fh:
            fh.write("not valid json\n")
        failures = log.verify_chain()
        assert any("JSON" in f["reason"] for f in failures)

    def test_export_period_creates_file(self):
        log = self._make_log()
        log.append(_make_entry(timestamp_utc="2025-03-15T10:00:00+00:00"))
        log.append(_make_entry(timestamp_utc="2025-04-01T10:00:00+00:00"))
        log.append(_make_entry(timestamp_utc="2025-05-01T10:00:00+00:00"))

        out_path = os.path.join(self._tmpdir, "export.jsonl")
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 4, 30, tzinfo=timezone.utc)
        result = log.export_period(start, end, out_path)

        assert result == Path(out_path)
        assert result.exists()

        lines = [l for l in result.read_text().splitlines() if not l.startswith("#")]
        assert len(lines) == 2  # March and April entries only

    def test_export_period_has_metadata_header(self):
        log = self._make_log()
        log.append(_make_entry(timestamp_utc="2025-03-01T00:00:00+00:00"))
        out_path = os.path.join(self._tmpdir, "export_meta.jsonl")
        log.export_period(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
            out_path,
        )
        first_line = Path(out_path).read_text().splitlines()[0]
        assert first_line.startswith("#")
        meta = json.loads(first_line[2:])
        assert meta["model_id"] == "motor-freq-v3"
        assert meta["entry_count"] == 1

    def test_export_period_start_after_end_raises(self):
        log = self._make_log()
        out_path = os.path.join(self._tmpdir, "bad.jsonl")
        with pytest.raises(ValueError, match="start must be before"):
            log.export_period(
                datetime(2025, 12, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                out_path,
            )

    def test_properties(self):
        log = self._make_log()
        assert log.model_id == "motor-freq-v3"
        assert log.model_version == "3.1.0"
        assert log.path == Path(self._log_path)


# ---------------------------------------------------------------------------
# PlainLanguageExplainer
# ---------------------------------------------------------------------------

class TestPlainLanguageExplainer:
    def setup_method(self):
        self._explainer = PlainLanguageExplainer(feature_labels=FEATURE_LABELS)

    def test_generate_returns_string(self):
        entry = _make_entry()
        text = self._explainer.generate(entry, base_premium=350.0)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_contains_premium_amount(self):
        entry = _make_entry(final_premium=412.50)
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "412.50" in text

    def test_generate_mentions_reducing_factor(self):
        # ncb_years has negative SHAP — should mention "reduced"
        entry = _make_entry()
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "reduced" in text.lower() or "no-claims" in text.lower()

    def test_generate_mentions_increasing_factor(self):
        # region has positive SHAP — should mention "added"
        entry = _make_entry()
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "added" in text.lower() or "postcode" in text.lower()

    def test_generate_uses_feature_labels_not_names(self):
        entry = _make_entry()
        text = self._explainer.generate(entry, base_premium=350.0)
        # Should use customer label, not internal name
        assert "ncb_years" not in text
        assert "driver_age" not in text

    def test_generate_excludes_unlabelled_features(self):
        entry = _make_entry(
            feature_importances={
                "driver_age": -0.12,
                "unknown_internal_feature": 0.99,
            }
        )
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "unknown_internal_feature" not in text

    def test_generate_handles_empty_importances(self):
        entry = _make_entry(feature_importances={})
        text = self._explainer.generate(entry, base_premium=350.0)
        assert isinstance(text, str)
        assert "contact us" in text.lower() or "detailed breakdown" in text.lower()

    def test_generate_includes_override_note(self):
        entry = _make_entry(
            override_applied=True,
            override_reason="Loyalty discount applied",
            decision_basis="human_override",
            human_reviewed=True,
        )
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "Loyalty discount applied" in text

    def test_generate_includes_rule_fallback_note(self):
        entry = _make_entry(decision_basis="rule_fallback")
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "minimum premium" in text.lower()

    def test_generate_invalid_base_premium_raises(self):
        entry = _make_entry()
        with pytest.raises(ValueError, match="base_premium must be positive"):
            self._explainer.generate(entry, base_premium=-10.0)

    def test_generate_custom_intro(self):
        entry = _make_entry()
        text = self._explainer.generate(
            entry, base_premium=350.0, intro="Dear Customer, here is your breakdown:"
        )
        assert text.startswith("Dear Customer")

    def test_generate_bullet_list(self):
        entry = _make_entry()
        bullets = self._explainer.generate_bullet_list(entry, base_premium=350.0)
        assert isinstance(bullets, list)
        assert len(bullets) >= 1
        # First bullet should be the summary
        assert "412.50" in bullets[0]

    def test_max_factors_limit(self):
        explainer = PlainLanguageExplainer(
            feature_labels={f"f{i}": f"factor {i}" for i in range(20)},
            max_factors=3,
        )
        importances = {f"f{i}": float(i + 1) * 0.05 for i in range(20)}
        entry = _make_entry(
            feature_importances=importances,
            input_features={f"f{i}": i for i in range(20)},
        )
        bullets = explainer.generate_bullet_list(entry, base_premium=100.0)
        # Summary + at most 3 factor bullets
        assert len(bullets) <= 4

    def test_currency_gbp_symbol(self):
        entry = _make_entry(final_premium=412.50)
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "£" in text

    def test_currency_eur_symbol(self):
        explainer = PlainLanguageExplainer(
            feature_labels=FEATURE_LABELS, currency="EUR"
        )
        entry = _make_entry(final_premium=412.50)
        text = explainer.generate(entry, base_premium=350.0)
        assert "€" in text

    def test_uses_prediction_when_no_final_premium(self):
        entry = _make_entry(prediction=388.0, final_premium=None)
        text = self._explainer.generate(entry, base_premium=350.0)
        assert "388.00" in text

    def test_feature_labels_property(self):
        assert self._explainer.feature_labels == FEATURE_LABELS


# ---------------------------------------------------------------------------
# AuditSummaryReport
# ---------------------------------------------------------------------------

class TestAuditSummaryReport:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._log_path = os.path.join(self._tmpdir, "audit.jsonl")
        self._log = ExplainabilityAuditLog(self._log_path, "motor-v3", "3.0.0")
        self._populate_log()

    def _populate_log(self):
        entries = [
            _make_entry(
                model_id="motor-v3",
                model_version="3.0.0",
                timestamp_utc="2025-03-10T09:00:00+00:00",
                input_features={"driver_age": 25, "ncb_years": 2, "region": "NW"},
                feature_importances={"driver_age": 0.20, "ncb_years": -0.05, "region": 0.10},
                prediction=480.0,
                final_premium=480.0,
                decision_basis="model_output",
            ),
            _make_entry(
                model_id="motor-v3",
                model_version="3.0.0",
                timestamp_utc="2025-03-15T14:00:00+00:00",
                input_features={"driver_age": 45, "ncb_years": 8, "region": "SE"},
                feature_importances={"driver_age": -0.10, "ncb_years": -0.40, "region": 0.05},
                prediction=310.0,
                final_premium=310.0,
                decision_basis="model_output",
            ),
            _make_entry(
                model_id="motor-v3",
                model_version="3.0.0",
                timestamp_utc="2025-03-20T11:00:00+00:00",
                input_features={"driver_age": 19, "ncb_years": 0, "region": "NW"},
                feature_importances={"driver_age": 0.55, "ncb_years": 0.30, "region": 0.10},
                prediction=720.0,
                final_premium=695.0,
                decision_basis="human_override",
                human_reviewed=True,
                reviewer_id="CF28-jane",
                override_applied=True,
                override_reason="Young driver cap applied",
            ),
        ]
        for e in entries:
            self._log.append(e)

    def test_build_returns_dict(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert isinstance(data, dict)

    def test_build_has_required_sections(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert "metadata" in data
        assert "decision_volume" in data
        assert "feature_importance" in data
        assert "integrity" in data

    def test_decision_volume_totals(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        vol = data["decision_volume"]
        assert vol["total"] == 3
        assert vol["overridden"] == 1
        assert vol["human_reviewed"] == 1
        assert vol["override_rate_pct"] == pytest.approx(33.33, abs=0.1)

    def test_decision_volume_by_basis(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        by_basis = data["decision_volume"]["by_basis"]
        assert by_basis["model_output"] == 2
        assert by_basis["human_override"] == 1

    def test_feature_importance_sorted_descending(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        fi = data["feature_importance"]
        means = [row["mean_abs_shap"] for row in fi]
        assert means == sorted(means, reverse=True)

    def test_feature_importance_includes_known_features(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        feature_names = {row["feature"] for row in data["feature_importance"]}
        assert "driver_age" in feature_names
        assert "ncb_years" in feature_names

    def test_metadata_period_and_model(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        meta = data["metadata"]
        assert meta["period"] == "2025-Q1"
        assert meta["model_id"] == "motor-v3"
        assert meta["entry_count"] == 3

    def test_integrity_passes_on_clean_log(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert data["integrity"]["pass"] is True
        assert data["integrity"]["failures"] == 0

    def test_segment_analysis_included_when_requested(self):
        report = AuditSummaryReport(
            self._log, period="2025-Q1", segment_feature="region"
        )
        data = report.build()
        assert "segment_analysis" in data
        seg = data["segment_analysis"]
        assert seg["feature"] == "region"
        seg_vals = {row["segment"] for row in seg["rows"]}
        assert "NW" in seg_vals
        assert "SE" in seg_vals

    def test_segment_analysis_absent_when_not_requested(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        data = report.build()
        assert "segment_analysis" not in data

    def test_date_range_filter(self):
        report = AuditSummaryReport(
            self._log,
            period="2025-Q1",
            start=datetime(2025, 3, 12, tzinfo=timezone.utc),
            end=datetime(2025, 3, 18, tzinfo=timezone.utc),
        )
        data = report.build()
        assert data["metadata"]["entry_count"] == 1

    def test_save_json(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        out = os.path.join(self._tmpdir, "report.json")
        result = report.save_json(out)
        assert result.exists()
        loaded = json.loads(result.read_text())
        assert loaded["metadata"]["period"] == "2025-Q1"

    def test_save_html(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        out = os.path.join(self._tmpdir, "report.html")
        result = report.save_html(out)
        assert result.exists()
        html = result.read_text()
        assert "<!DOCTYPE html>" in html
        assert "2025-Q1" in html
        assert "motor-v3" in html

    def test_html_contains_integrity_status(self):
        report = AuditSummaryReport(self._log, period="2025-Q1")
        out = os.path.join(self._tmpdir, "report_integrity.html")
        result = report.save_html(out)
        html = result.read_text()
        assert "PASS" in html

    def test_html_with_segment_feature(self):
        report = AuditSummaryReport(
            self._log, period="2025-Q1", segment_feature="region"
        )
        out = os.path.join(self._tmpdir, "report_seg.html")
        result = report.save_html(out)
        html = result.read_text()
        assert "Per-Segment Analysis" in html
        assert "NW" in html


# ---------------------------------------------------------------------------
# SHAPExplainer (using mocks — shap package not required)
# ---------------------------------------------------------------------------

class TestSHAPExplainerImportError:
    def test_raises_import_error_when_shap_not_available(self):
        """If shap is not installed, SHAPExplainer must raise ImportError."""
        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="pip install shap"):
                SHAPExplainer(
                    model=object(),
                    model_type="tree",
                    feature_names=["a", "b"],
                )

    def test_import_error_message_includes_extra_install(self):
        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="insurance-governance\\[shap\\]"):
                SHAPExplainer(
                    model=object(),
                    model_type="tree",
                    feature_names=["a", "b"],
                )


class TestSHAPExplainerWithMock:
    """Test SHAPExplainer logic with a fully mocked shap library.

    We mock the _shap module and the TreeExplainer so these tests run without
    the shap package installed.
    """

    def _make_explainer(self, n_features: int = 3) -> SHAPExplainer:
        """Create a SHAPExplainer backed by a mock TreeExplainer."""
        import numpy as np

        feature_names = [f"f{i}" for i in range(n_features)]

        # Create mock shap values that TreeExplainer.shap_values returns
        mock_shap_values = np.array([[0.1 * i for i in range(n_features)]])

        mock_tree_explainer = mock.MagicMock()
        mock_tree_explainer.shap_values.return_value = mock_shap_values
        mock_tree_explainer.expected_value = 0.5

        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", True
        ), mock.patch(
            "insurance_governance.audit.shap_explainer._shap"
        ) as mock_shap_module:
            mock_shap_module.TreeExplainer.return_value = mock_tree_explainer
            explainer = SHAPExplainer(
                model=mock.MagicMock(),
                model_type="tree",
                feature_names=feature_names,
            )
            # Attach mock for subsequent calls
            explainer._explainer = mock_tree_explainer

        return explainer, mock_shap_values, feature_names

    def test_invalid_model_type_raises(self):
        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", True
        ), mock.patch("insurance_governance.audit.shap_explainer._shap"):
            with pytest.raises(ValueError, match="model_type must be one of"):
                SHAPExplainer(
                    model=object(),
                    model_type="invalid",
                    feature_names=["a", "b"],
                )

    def test_kernel_without_background_raises(self):
        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", True
        ), mock.patch("insurance_governance.audit.shap_explainer._shap"):
            with pytest.raises(ValueError, match="background dataset"):
                SHAPExplainer(
                    model=mock.MagicMock(),
                    model_type="kernel",
                    feature_names=["a", "b"],
                    background=None,
                )

    def test_explain_returns_list_of_dicts(self):
        import numpy as np
        explainer, mock_vals, feature_names = self._make_explainer(n_features=3)
        X = np.zeros((1, 3))
        result = explainer.explain(X)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert set(result[0].keys()) == set(feature_names)

    def test_explain_column_count_mismatch_raises(self):
        import numpy as np
        explainer, _, _ = self._make_explainer(n_features=3)
        X = np.zeros((1, 5))  # wrong number of columns
        with pytest.raises(ValueError, match="Expected 3 features"):
            explainer.explain(X)

    def test_explain_single_returns_dict(self):
        import numpy as np
        explainer, mock_vals, feature_names = self._make_explainer(n_features=3)
        x = np.zeros(3)
        result = explainer.explain_single(x)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(feature_names)

    def test_explain_values_are_floats(self):
        import numpy as np
        explainer, _, feature_names = self._make_explainer(n_features=3)
        X = np.zeros((1, 3))
        result = explainer.explain(X)
        for val in result[0].values():
            assert isinstance(val, float)

    def test_explain_handles_list_shap_output(self):
        """TreeExplainer returns a list for binary classifiers; we take the last element."""
        import numpy as np
        feature_names = ["a", "b", "c"]
        mock_shap_class0 = np.array([[-0.1, -0.2, -0.3]])
        mock_shap_class1 = np.array([[0.1, 0.2, 0.3]])

        mock_tree_explainer = mock.MagicMock()
        mock_tree_explainer.shap_values.return_value = [mock_shap_class0, mock_shap_class1]
        mock_tree_explainer.expected_value = [0.3, 0.7]

        with mock.patch(
            "insurance_governance.audit.shap_explainer._SHAP_AVAILABLE", True
        ), mock.patch(
            "insurance_governance.audit.shap_explainer._shap"
        ) as mock_shap_module:
            mock_shap_module.TreeExplainer.return_value = mock_tree_explainer
            explainer = SHAPExplainer(
                model=mock.MagicMock(),
                model_type="tree",
                feature_names=feature_names,
            )
            explainer._explainer = mock_tree_explainer

        X = np.zeros((1, 3))
        result = explainer.explain(X)
        assert result[0]["a"] == pytest.approx(0.1)
        assert result[0]["b"] == pytest.approx(0.2)

    def test_feature_names_property(self):
        explainer, _, feature_names = self._make_explainer(n_features=4)
        assert explainer.feature_names == feature_names

    def test_expected_value_scalar(self):
        explainer, _, _ = self._make_explainer()
        explainer._explainer.expected_value = 0.42
        assert explainer.expected_value() == pytest.approx(0.42)

    def test_expected_value_list(self):
        import numpy as np
        explainer, _, _ = self._make_explainer()
        explainer._explainer.expected_value = [0.3, 0.7]
        assert explainer.expected_value() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------

class TestTopLevelReExport:
    def test_audit_classes_importable_from_top_level(self):
        from insurance_governance import (
            ExplainabilityAuditEntry,
            ExplainabilityAuditLog,
            PlainLanguageExplainer,
            AuditSummaryReport,
            SHAPExplainer,
        )
        assert ExplainabilityAuditEntry is not None
        assert ExplainabilityAuditLog is not None
        assert PlainLanguageExplainer is not None
        assert AuditSummaryReport is not None
        assert SHAPExplainer is not None
