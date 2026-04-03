"""
Batch 24 test coverage expansion for insurance-governance.

Targets modules and branches with low coverage:
1. audit.entry — hash tamper detection, from_dict roundtrip, decision_basis validation
2. audit.log — append/read/verify chain, export_period, read_since, corrupt lines
3. audit.report — AuditSummaryReport build/save_json/save_html, segment analysis
4. audit.customer_explanation — PlainLanguageExplainer.generate/generate_bullet_list
5. euaia.classifier — AIActClassifier all branches (unknown LOB, no personal data, etc.)
6. euaia.article13 — Article13Document.flag_gaps, compute_accuracy, subgroup perf
7. euaia.conformity — ConformityAssessment steps, run_all with/without article13
8. outcome.results — OutcomeResult validation, OutcomeSuite properties
9. outcome.metrics — PriceValueMetrics/ClaimsMetrics all edge cases
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# Fixtures / helpers
# ============================================================================

def _make_entry(**kwargs):
    from insurance_governance.audit.entry import ExplainabilityAuditEntry
    defaults = dict(
        model_id="motor-freq-v3",
        model_version="3.1.0",
        input_features={"driver_age": 35, "ncb_years": 5},
        feature_importances={"driver_age": 0.12, "ncb_years": -0.08},
        prediction=425.0,
        final_premium=430.0,
        decision_basis="model_output",
    )
    defaults.update(kwargs)
    return ExplainabilityAuditEntry(**defaults)


# ============================================================================
# 1. audit.entry
# ============================================================================

class TestExplainabilityAuditEntryExtra:

    def test_empty_model_id_raises(self):
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        with pytest.raises(ValueError, match="model_id"):
            ExplainabilityAuditEntry(
                model_id="",
                model_version="1.0",
                input_features={},
                feature_importances={},
                prediction=100.0,
            )

    def test_empty_model_version_raises(self):
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        with pytest.raises(ValueError, match="model_version"):
            ExplainabilityAuditEntry(
                model_id="m1",
                model_version="",
                input_features={},
                feature_importances={},
                prediction=100.0,
            )

    def test_invalid_decision_basis_raises(self):
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        with pytest.raises(ValueError, match="decision_basis"):
            ExplainabilityAuditEntry(
                model_id="m1",
                model_version="1.0",
                input_features={},
                feature_importances={},
                prediction=100.0,
                decision_basis="magic_override",
            )

    def test_override_applied_without_reason_raises(self):
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        with pytest.raises(ValueError, match="override_reason"):
            ExplainabilityAuditEntry(
                model_id="m1",
                model_version="1.0",
                input_features={},
                feature_importances={},
                prediction=100.0,
                override_applied=True,
                override_reason=None,
            )

    def test_override_applied_with_reason_ok(self):
        entry = _make_entry(
            override_applied=True,
            override_reason="Underwriter adjusted for high flood risk area",
            decision_basis="human_override",
        )
        assert entry.override_applied is True
        assert entry.override_reason is not None

    def test_human_override_decision_basis_accepted(self):
        entry = _make_entry(
            decision_basis="human_override",
            override_applied=True,
            override_reason="Manual review",
        )
        assert entry.decision_basis == "human_override"

    def test_rule_fallback_decision_basis_accepted(self):
        entry = _make_entry(decision_basis="rule_fallback")
        assert entry.decision_basis == "rule_fallback"

    def test_verify_integrity_passes_for_new_entry(self):
        entry = _make_entry()
        assert entry.verify_integrity() is True

    def test_tampered_entry_fails_integrity(self):
        entry = _make_entry()
        # Manually corrupt the prediction field after construction
        object.__setattr__(entry, "prediction", 9999.0)
        assert entry.verify_integrity() is False

    def test_to_dict_and_from_dict_roundtrip(self):
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        entry = _make_entry(session_id="batch-2026-01")
        d = entry.to_dict()
        restored = ExplainabilityAuditEntry.from_dict(d)
        assert restored.model_id == entry.model_id
        assert restored.prediction == entry.prediction
        assert restored.entry_hash == entry.entry_hash
        assert restored.verify_integrity() is True

    def test_from_dict_preserves_stored_hash(self):
        """If stored hash differs from recomputed, verify_integrity returns False."""
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        entry = _make_entry()
        d = entry.to_dict()
        d["entry_hash"] = "000000deadbeef"
        restored = ExplainabilityAuditEntry.from_dict(d)
        assert restored.verify_integrity() is False

    def test_from_dict_missing_optional_fields_uses_defaults(self):
        """from_dict with only required fields should not raise."""
        from insurance_governance.audit.entry import ExplainabilityAuditEntry
        entry = ExplainabilityAuditEntry.from_dict({
            "model_id": "m1",
            "model_version": "1.0.0",
            "prediction": 200.0,
        })
        assert entry.model_id == "m1"
        assert entry.final_premium is None

    def test_entry_hash_64_hex_chars(self):
        """SHA-256 hash should be 64 hex chars."""
        entry = _make_entry()
        assert len(entry.entry_hash) == 64
        assert all(c in "0123456789abcdef" for c in entry.entry_hash)

    def test_two_entries_same_content_same_hash(self):
        """Two entries with identical content should produce the same hash."""
        e1 = _make_entry(entry_id="fixed-id", timestamp_utc="2026-01-01T00:00:00+00:00")
        e2 = _make_entry(entry_id="fixed-id", timestamp_utc="2026-01-01T00:00:00+00:00")
        assert e1.entry_hash == e2.entry_hash

    def test_different_predictions_different_hash(self):
        e1 = _make_entry(entry_id="fixed-id", timestamp_utc="2026-01-01T00:00:00+00:00",
                         prediction=100.0)
        e2 = _make_entry(entry_id="fixed-id", timestamp_utc="2026-01-01T00:00:00+00:00",
                         prediction=200.0)
        assert e1.entry_hash != e2.entry_hash


# ============================================================================
# 2. audit.log
# ============================================================================

class TestExplainabilityAuditLog:

    def test_append_and_read_all(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = tmp_path / "audit.jsonl"
        log = ExplainabilityAuditLog(path, "motor-freq-v3", "3.1.0")
        for i in range(3):
            entry = _make_entry(prediction=float(400 + i * 10))
            log.append(entry)
        entries = log.read_all()
        assert len(entries) == 3
        assert entries[0].prediction == 400.0
        assert entries[2].prediction == 420.0

    def test_log_created_if_not_exists(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = tmp_path / "new_log.jsonl"
        assert not path.exists()
        log = ExplainabilityAuditLog(path, "m1", "1.0")
        assert path.exists()

    def test_append_non_entry_raises_type_error(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        with pytest.raises(TypeError, match="ExplainabilityAuditEntry"):
            log.append({"not": "an entry"})  # type: ignore[arg-type]

    def test_read_all_empty_log_returns_empty_list(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "empty.jsonl", "m1", "1.0")
        assert log.read_all() == []

    def test_verify_chain_clean_log(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        log.append(_make_entry())
        log.append(_make_entry())
        failures = log.verify_chain()
        assert failures == []

    def test_verify_chain_corrupt_json_returns_failure(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = tmp_path / "corrupt.jsonl"
        path.write_text("{valid json line}\n{invalid json :::}\n")
        log = ExplainabilityAuditLog(path, "m1", "1.0")
        failures = log.verify_chain()
        # The invalid line should produce a failure
        assert len(failures) >= 1
        assert any(f["reason"] == "JSON parse failure" for f in failures)

    def test_verify_chain_tampered_entry_detected(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = tmp_path / "tampered.jsonl"
        log = ExplainabilityAuditLog(path, "m1", "1.0")
        entry = _make_entry()
        d = entry.to_dict()
        # Tamper with the stored prediction while keeping the old hash
        d["prediction"] = 9999.0
        path.write_text(json.dumps(d) + "\n")
        failures = log.verify_chain()
        assert len(failures) == 1
        assert "tampered" in failures[0]["reason"].lower()

    def test_read_since_naive_cutoff_treated_as_utc(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        entry = _make_entry(timestamp_utc="2026-06-01T12:00:00+00:00")
        log.append(entry)
        # naive cutoff — should be treated as UTC
        cutoff = datetime(2026, 1, 1, 0, 0, 0)
        results = log.read_since(cutoff)
        assert len(results) == 1

    def test_read_since_filters_old_entries(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        old_entry = _make_entry(timestamp_utc="2020-01-01T00:00:00+00:00")
        new_entry = _make_entry(timestamp_utc="2026-06-01T00:00:00+00:00")
        log.append(old_entry)
        log.append(new_entry)
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        results = log.read_since(cutoff)
        assert len(results) == 1
        assert results[0].timestamp_utc == "2026-06-01T00:00:00+00:00"

    def test_export_period_start_after_end_raises(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="start must be before"):
            log.export_period(start, end, tmp_path / "out.jsonl")

    def test_export_period_writes_metadata_header(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        entry = _make_entry(timestamp_utc="2026-03-15T10:00:00+00:00")
        log.append(entry)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        out = tmp_path / "export.jsonl"
        result_path = log.export_period(start, end, out)
        lines = result_path.read_text().splitlines()
        assert lines[0].startswith("# ")
        meta = json.loads(lines[0][2:])
        assert meta["export_type"] == "explainability_audit"
        assert meta["entry_count"] == 1

    def test_export_period_filters_outside_window(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        log.append(_make_entry(timestamp_utc="2025-01-01T00:00:00+00:00"))  # before window
        log.append(_make_entry(timestamp_utc="2026-03-01T00:00:00+00:00"))  # in window
        log.append(_make_entry(timestamp_utc="2027-01-01T00:00:00+00:00"))  # after window
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        out = tmp_path / "export.jsonl"
        log.export_period(start, end, out)
        lines = [l for l in out.read_text().splitlines() if not l.startswith("#") and l.strip()]
        assert len(lines) == 1

    def test_log_path_property(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        path = tmp_path / "log.jsonl"
        log = ExplainabilityAuditLog(path, "m1", "1.0")
        assert log.path == path
        assert log.model_id == "m1"
        assert log.model_version == "1.0"


# ============================================================================
# 3. audit.report — AuditSummaryReport
# ============================================================================

class TestAuditSummaryReport:

    def _populated_log(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "motor-v3", "3.0.0")
        for i in range(5):
            entry = _make_entry(
                timestamp_utc="2026-03-01T10:00:00+00:00",
                prediction=float(400 + i * 10),
                final_premium=float(410 + i * 10),
                input_features={"driver_age": 20 + i * 5, "region": "North" if i < 3 else "South"},
            )
            log.append(entry)
        # One human override
        override_entry = _make_entry(
            timestamp_utc="2026-03-02T10:00:00+00:00",
            human_reviewed=True,
            override_applied=True,
            override_reason="Flood risk area",
            decision_basis="human_override",
        )
        log.append(override_entry)
        return log

    def test_build_returns_required_keys(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        data = report.build()
        assert "metadata" in data
        assert "decision_volume" in data
        assert "feature_importance" in data
        assert "integrity" in data

    def test_decision_volume_counts(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        data = report.build()
        vol = data["decision_volume"]
        assert vol["total"] == 6
        assert vol["overridden"] == 1
        assert vol["human_reviewed"] == 1

    def test_human_reviewed_pct_correct(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        vol = report.build()["decision_volume"]
        assert vol["human_reviewed_pct"] == pytest.approx(100.0 / 6, abs=0.1)

    def test_feature_importance_sorted_descending(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        features = report.build()["feature_importance"]
        means = [r["mean_abs_shap"] for r in features]
        assert means == sorted(means, reverse=True)

    def test_segment_analysis_produced_when_feature_set(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1", segment_feature="region")
        data = report.build()
        assert "segment_analysis" in data
        seg = data["segment_analysis"]
        assert seg["feature"] == "region"
        assert len(seg["rows"]) >= 2  # North and South

    def test_no_segment_analysis_when_feature_not_set(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        data = report.build()
        assert "segment_analysis" not in data

    def test_save_json_creates_file(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        out = tmp_path / "report.json"
        result_path = report.save_json(out)
        assert result_path.exists()
        content = json.loads(result_path.read_text())
        assert content["metadata"]["period"] == "2026-Q1"

    def test_save_html_creates_file(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        out = tmp_path / "report.html"
        result_path = report.save_html(out)
        assert result_path.exists()
        html = result_path.read_text()
        assert "Explainability Audit Report" in html
        assert "2026-Q1" in html

    def test_empty_log_zero_totals(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        from insurance_governance.audit.report import AuditSummaryReport
        log = ExplainabilityAuditLog(tmp_path / "empty.jsonl", "m1", "1.0")
        report = AuditSummaryReport(log, period="2026-Q1")
        data = report.build()
        assert data["decision_volume"]["total"] == 0
        assert data["decision_volume"]["human_reviewed_pct"] == 0.0

    def test_start_end_filter_applied(self, tmp_path):
        from insurance_governance.audit.log import ExplainabilityAuditLog
        from insurance_governance.audit.report import AuditSummaryReport
        log = ExplainabilityAuditLog(tmp_path / "log.jsonl", "m1", "1.0")
        log.append(_make_entry(timestamp_utc="2025-06-01T00:00:00+00:00"))
        log.append(_make_entry(timestamp_utc="2026-03-01T00:00:00+00:00"))
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        report = AuditSummaryReport(log, period="2026", start=start, end=end)
        data = report.build()
        assert data["decision_volume"]["total"] == 1

    def test_html_contains_integrity_badge(self, tmp_path):
        from insurance_governance.audit.report import AuditSummaryReport
        log = self._populated_log(tmp_path)
        report = AuditSummaryReport(log, period="2026-Q1")
        html = report._render_html(report.build())
        assert "PASS" in html or "FAIL" in html


# ============================================================================
# 4. audit.customer_explanation — PlainLanguageExplainer
# ============================================================================

class TestPlainLanguageExplainer:

    def _make_explainer(self, **kwargs):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        labels = {
            "driver_age": "your age",
            "ncb_years": "your no-claims discount",
            "region": "your postcode area",
        }
        return PlainLanguageExplainer(feature_labels=labels, **kwargs)

    def test_generate_basic(self):
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5, "ncb_years": -0.3},
            final_premium=450.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "£450.00" in text
        assert "age" in text.lower() or "no-claims" in text.lower()

    def test_generate_no_contributions_fallback(self):
        """If all features are below min_impact_pct, fallback sentence is returned."""
        explainer = self._make_explainer(min_impact_pct=100.0)  # unreachable threshold
        entry = _make_entry(
            feature_importances={"driver_age": 0.001},
            final_premium=350.0,
        )
        text = explainer.generate(entry, base_premium=349.0)
        assert "risk profile" in text.lower()

    def test_generate_empty_importances_fallback(self):
        explainer = self._make_explainer()
        entry = _make_entry(feature_importances={}, final_premium=350.0)
        text = explainer.generate(entry, base_premium=350.0)
        assert "risk profile" in text.lower()

    def test_generate_override_note_appended(self):
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5},
            final_premium=450.0,
            override_applied=True,
            override_reason="Flood zone adjustment",
            decision_basis="human_override",
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "Flood zone adjustment" in text or "underwriter" in text.lower()

    def test_generate_rule_fallback_note_appended(self):
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5},
            final_premium=450.0,
            decision_basis="rule_fallback",
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "minimum premium" in text.lower()

    def test_generate_custom_intro(self):
        explainer = self._make_explainer()
        entry = _make_entry(feature_importances={"driver_age": 0.5}, final_premium=400.0)
        text = explainer.generate(entry, base_premium=350.0, intro="Custom intro line.")
        assert text.startswith("Custom intro line.")

    def test_generate_zero_base_premium_raises(self):
        explainer = self._make_explainer()
        entry = _make_entry(feature_importances={"driver_age": 0.1}, final_premium=350.0)
        with pytest.raises(ValueError, match="base_premium"):
            explainer.generate(entry, base_premium=0.0)

    def test_generate_negative_impact_sentence(self):
        """Negative SHAP should produce 'reduced your premium' sentence."""
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"ncb_years": -0.5, "driver_age": 0.1},
            final_premium=300.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "reduced" in text.lower()

    def test_generate_positive_impact_sentence(self):
        """Positive SHAP should produce 'added' sentence."""
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.8, "ncb_years": -0.1},
            final_premium=450.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "added" in text.lower()

    def test_generate_uses_prediction_when_no_final_premium(self):
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5},
            final_premium=None,
            prediction=380.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "£380.00" in text

    def test_generate_bullet_list_returns_list(self):
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5, "ncb_years": -0.2},
            final_premium=400.0,
        )
        bullets = explainer.generate_bullet_list(entry, base_premium=350.0)
        assert isinstance(bullets, list)
        assert len(bullets) >= 1
        assert "£400.00" in bullets[0]

    def test_max_factors_limits_output(self):
        explainer = self._make_explainer(max_factors=1)
        entry = _make_entry(
            feature_importances={"driver_age": 0.5, "ncb_years": -0.3, "region": 0.4},
            final_premium=450.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        # Only the top 1 factor should appear; can't have more than max_factors sentences
        # after the intro line
        parts = text.split(".")
        factor_sentences = [p for p in parts if ("added" in p or "reduced" in p)]
        assert len(factor_sentences) <= 1

    def test_currency_eur(self):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels={"driver_age": "your age"}, currency="EUR"
        )
        entry = _make_entry(
            feature_importances={"driver_age": 0.5},
            final_premium=400.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "€" in text

    def test_unknown_currency_uses_code_as_symbol(self):
        from insurance_governance.audit.customer_explanation import PlainLanguageExplainer
        explainer = PlainLanguageExplainer(
            feature_labels={"driver_age": "your age"}, currency="JPY"
        )
        entry = _make_entry(
            feature_importances={"driver_age": 0.5},
            final_premium=400.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "JPY" in text

    def test_feature_not_in_labels_excluded(self):
        """Features not in feature_labels should be silently excluded."""
        explainer = self._make_explainer()
        entry = _make_entry(
            feature_importances={"driver_age": 0.5, "unknown_feature": 0.9},
            final_premium=450.0,
        )
        text = explainer.generate(entry, base_premium=350.0)
        assert "unknown_feature" not in text

    def test_feature_labels_property(self):
        explainer = self._make_explainer()
        labels = explainer.feature_labels
        assert "driver_age" in labels


# ============================================================================
# 5. euaia.classifier — AIActClassifier
# ============================================================================

class TestAIActClassifier:

    def _clf(self):
        from insurance_governance.euaia.classifier import AIActClassifier
        return AIActClassifier()

    def test_glm_motor_out_of_scope(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("glm", "motor", True, True)
        assert result.risk_classification == RiskClassification.OUT_OF_SCOPE
        assert result.is_ai_system is False

    def test_rule_based_out_of_scope(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("rule_based", "home", True, True)
        assert result.risk_classification == RiskClassification.OUT_OF_SCOPE

    def test_gradient_boosting_motor_not_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "motor", True, True)
        assert result.risk_classification == RiskClassification.NOT_HIGH_RISK
        assert result.is_ai_system is True
        assert result.requires_conformity_assessment is False

    def test_gradient_boosting_life_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "life", True, True)
        assert result.risk_classification == RiskClassification.HIGH_RISK
        assert result.requires_conformity_assessment is True
        assert result.assessment_route == "internal_control"

    def test_neural_network_health_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("neural_network", "health", True, True)
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_gradient_boosting_life_no_personal_data_potentially_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "life", False, True)
        assert result.risk_classification == RiskClassification.POTENTIALLY_HIGH_RISK

    def test_gradient_boosting_life_no_automated_decision_potentially_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "life", True, False)
        assert result.risk_classification == RiskClassification.POTENTIALLY_HIGH_RISK

    def test_unknown_model_type_treated_as_ai_system(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("some_fancy_custom_thing", "motor", True, True)
        # Unknown -> ModelType.UNKNOWN -> AI system but motor -> NOT_HIGH_RISK
        assert result.is_ai_system is True
        assert len(result.warnings) > 0

    def test_decision_tree_borderline_warning(self):
        result = self._clf().classify("decision_tree", "life", True, True)
        assert any("decision tree" in w.lower() or "decision_tree" in w.lower()
                   for w in result.warnings)

    def test_unknown_lob_generates_warning(self):
        result = self._clf().classify("gradient_boosting", "exotic_product", True, True)
        assert len(result.warnings) > 0

    def test_rationale_is_populated(self):
        result = self._clf().classify("gradient_boosting", "life", True, True)
        assert len(result.rationale) > 0

    def test_annuity_is_high_risk_lob(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "annuity", True, True)
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_pmi_is_high_risk_lob(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "pmi", True, True)
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_gam_treated_conservatively_as_ai_system(self):
        """GAM is borderline but treated conservatively as an AI system."""
        result = self._clf().classify("gam", "motor", True, True)
        assert result.is_ai_system is True

    def test_regularised_regression_life_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("regularised_regression", "life", True, True)
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_property_lob_not_high_risk(self):
        from insurance_governance.euaia.classifier import RiskClassification
        result = self._clf().classify("gradient_boosting", "property", True, True)
        assert result.risk_classification == RiskClassification.NOT_HIGH_RISK


# ============================================================================
# 6. euaia.article13 — Article13Document
# ============================================================================

class TestArticle13Document:

    def _minimal_doc(self):
        from insurance_governance.euaia.article13 import Article13Document
        return Article13Document(
            provider_name="Acme Insurance Ltd",
            provider_contact="pricing@acme.example.com",
            model_name="Motor Frequency Model v3",
            model_version="3.0.0",
            document_date="2026-01-01",
            intended_purpose="Predict claim frequency for motor insurance pricing.",
            out_of_scope_uses=["Life insurance pricing"],
            known_risks=["Distribution shift on new vehicle types"],
            explanation_tools=["SHAP TreeExplainer"],
            input_features=[{"name": "driver_age", "type": "numeric"}],
            human_oversight_measures=["Senior actuary sign-off"],
            override_procedure="Underwriter manual review form MRC-14",
            anomaly_thresholds={"ae_ratio": 1.25},
            expected_lifetime_months=24,
            output_interpretation_guide="Output is frequency per year. Multiply by severity to get premium.",
            monitoring_metrics=["PSI", "AE ratio"],
            retraining_triggers=["AE ratio > 1.25 for 3 consecutive months"],
        )

    def test_flag_gaps_empty_document(self):
        from insurance_governance.euaia.article13 import Article13Document
        doc = Article13Document()
        gaps = doc.flag_gaps()
        assert len(gaps) > 0
        # Should flag all required fields
        gap_text = " ".join(gaps)
        assert "provider_name" in gap_text

    def test_flag_gaps_minimal_complete_document(self):
        doc = self._minimal_doc()
        doc.accuracy_metrics = {"gini": 0.42}
        gaps = doc.flag_gaps()
        assert len(gaps) == 0

    def test_flag_gaps_missing_accuracy_metrics(self):
        doc = self._minimal_doc()
        doc.accuracy_metrics = {}
        gaps = doc.flag_gaps()
        assert any("accuracy" in g.lower() for g in gaps)

    def test_flag_gaps_zero_expected_lifetime(self):
        doc = self._minimal_doc()
        doc.accuracy_metrics = {"gini": 0.4}
        doc.expected_lifetime_months = 0
        gaps = doc.flag_gaps()
        assert any("expected_lifetime" in g for g in gaps)

    def test_flag_gaps_empty_anomaly_thresholds(self):
        doc = self._minimal_doc()
        doc.accuracy_metrics = {"gini": 0.4}
        doc.anomaly_thresholds = {}
        gaps = doc.flag_gaps()
        assert any("anomaly_thresholds" in g for g in gaps)

    def test_compute_accuracy_returns_gini(self):
        doc = self._minimal_doc()
        rng = np.random.default_rng(42)
        y_true = rng.poisson(0.05, 1000).astype(float)
        y_pred = y_true + rng.normal(0, 0.02, 1000)
        y_pred = np.clip(y_pred, 0, None)
        metrics = doc.compute_accuracy(y_true, y_pred, n_boot=50)
        assert "gini" in metrics
        assert "ae_ratio" in metrics
        assert "n_obs" in metrics
        assert np.isfinite(metrics["gini"])

    def test_compute_accuracy_shape_mismatch_raises(self):
        doc = self._minimal_doc()
        with pytest.raises(ValueError, match="shape"):
            doc.compute_accuracy(np.ones(10), np.ones(9))

    def test_compute_accuracy_stored_in_accuracy_metrics(self):
        doc = self._minimal_doc()
        rng = np.random.default_rng(1)
        y_true = rng.exponential(0.1, 500)
        y_pred = y_true + rng.normal(0, 0.01, 500)
        doc.compute_accuracy(y_true, y_pred, n_boot=20)
        assert "gini" in doc.accuracy_metrics

    def test_compute_subgroup_performance_excludes_small_groups(self):
        doc = self._minimal_doc()
        n = 500
        rng = np.random.default_rng(2)
        y_true = rng.exponential(0.1, n)
        y_pred = y_true + rng.normal(0, 0.01, n)
        groups = {
            "large_group": np.ones(n, dtype=bool),
            "tiny_group": np.arange(n) < 5,  # only 5 obs
        }
        results = doc.compute_subgroup_performance(
            y_true, y_pred, groups, min_group_size=100
        )
        assert "large_group" in results
        assert "tiny_group" not in results
        assert any("tiny_group" in s for s in doc.known_accuracy_limitations)

    def test_to_dict_structure(self):
        doc = self._minimal_doc()
        d = doc.to_dict()
        assert "article13_transparency_document" in d
        inner = d["article13_transparency_document"]
        assert "a_provider" in inner
        assert "b_performance" in inner
        assert "d_human_oversight" in inner
        assert "e_maintenance" in inner

    def test_to_markdown_returns_string(self):
        doc = self._minimal_doc()
        doc.accuracy_metrics = {"gini": 0.42}
        md = doc.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100

    def test_gini_coefficient_perfect_model(self):
        from insurance_governance.euaia.article13 import _gini_coefficient
        # Perfect ranking: sorted equally
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = _gini_coefficient(y_true, y_pred)
        assert np.isfinite(g)

    def test_gini_coefficient_empty_returns_nan(self):
        from insurance_governance.euaia.article13 import _gini_coefficient
        g = _gini_coefficient(np.array([]), np.array([]))
        assert np.isnan(g)

    def test_gini_coefficient_zero_true_returns_nan(self):
        from insurance_governance.euaia.article13 import _gini_coefficient
        y_true = np.zeros(10)
        y_pred = np.ones(10)
        g = _gini_coefficient(y_true, y_pred)
        assert np.isnan(g)

    def test_gini_length_mismatch_raises(self):
        from insurance_governance.euaia.article13 import _gini_coefficient
        with pytest.raises(ValueError):
            _gini_coefficient(np.ones(5), np.ones(6))

    def test_bootstrap_ci_returns_tuple(self):
        from insurance_governance.euaia.article13 import _bootstrap_ci
        rng = np.random.default_rng(42)
        y_true = rng.exponential(1, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)
        lo, hi = _bootstrap_ci(y_true, y_pred, n_boot=100)
        assert lo <= hi
        assert np.isfinite(lo) and np.isfinite(hi)


# ============================================================================
# 7. euaia.conformity — ConformityAssessment
# ============================================================================

class TestConformityAssessment:

    def _base_ca(self):
        from insurance_governance.euaia.conformity import ConformityAssessment
        return ConformityAssessment(
            model_name="Life Propensity Model v2",
            assessor_name="Model Governance Team",
            assessment_date="2026-01-15",
        )

    def test_default_steps_are_seven(self):
        ca = self._base_ca()
        assert len(ca.steps) == 7

    def test_get_step_valid(self):
        ca = self._base_ca()
        step1 = ca.get_step(1)
        assert step1.step_number == 1
        assert "Risk Classification" in step1.title

    def test_get_step_out_of_range_raises(self):
        ca = self._base_ca()
        with pytest.raises(ValueError, match="step_number"):
            ca.get_step(8)
        with pytest.raises(ValueError, match="step_number"):
            ca.get_step(0)

    def test_flag_incomplete_returns_all_steps_initially(self):
        ca = self._base_ca()
        incomplete = ca.flag_incomplete()
        assert len(incomplete) == 7

    def test_complete_all_steps_makes_overall_complete(self):
        from insurance_governance.euaia.conformity import StepStatus
        ca = self._base_ca()
        for s in ca.steps:
            s.status = StepStatus.COMPLETE
        assert ca.overall_status() == StepStatus.COMPLETE

    def test_one_incomplete_step_makes_overall_incomplete(self):
        from insurance_governance.euaia.conformity import StepStatus
        ca = self._base_ca()
        for s in ca.steps:
            s.status = StepStatus.COMPLETE
        ca.get_step(4).status = StepStatus.INCOMPLETE
        assert ca.overall_status() == StepStatus.INCOMPLETE

    def test_not_applicable_counts_as_complete(self):
        from insurance_governance.euaia.conformity import StepStatus
        ca = self._base_ca()
        for s in ca.steps:
            s.status = StepStatus.COMPLETE
        ca.get_step(7).status = StepStatus.NOT_APPLICABLE
        assert ca.overall_status() == StepStatus.COMPLETE

    def test_run_all_without_article13_flags_manual_steps(self):
        ca = self._base_ca()
        findings = ca.run_all()
        # Steps 1, 2, 4, 7 need manual evidence
        manual_steps = [ca.get_step(n) for n in (1, 2, 4, 7)]
        for step in manual_steps:
            assert any("manual input required" in f for f in step.findings)

    def test_run_all_with_article13_auto_checks_step3(self):
        """With a partially-complete Article13Document, step 3 should get findings."""
        from insurance_governance.euaia.article13 import Article13Document
        ca = self._base_ca()
        doc = Article13Document()  # empty — has many gaps
        ca.attach_article13(doc)
        findings = ca.run_all()
        s3 = ca.get_step(3)
        assert len(s3.findings) > 0

    def test_run_all_with_complete_article13_step3_gets_evidence(self):
        from insurance_governance.euaia.article13 import Article13Document
        from insurance_governance.euaia.conformity import StepStatus
        ca = self._base_ca()
        doc = Article13Document(
            provider_name="Acme", provider_contact="x@a.com",
            model_name="Model", model_version="1.0", document_date="2026-01-01",
            intended_purpose="Test", out_of_scope_uses=["X"],
            known_risks=["Y"], explanation_tools=["SHAP"],
            input_features=[{"name": "age"}],
            human_oversight_measures=["Review"], override_procedure="Procedure",
            anomaly_thresholds={"ae_ratio": 1.2},
            expected_lifetime_months=12,
            output_interpretation_guide="Guide",
            monitoring_metrics=["PSI"],
            retraining_triggers=["AE > 1.3"],
            accuracy_metrics={"gini": 0.4},
        )
        ca.attach_article13(doc)
        ca.run_all()
        s3 = ca.get_step(3)
        # No gaps → step 3 should auto-complete
        assert s3.status == StepStatus.COMPLETE or len(s3.findings) == 0

    def test_run_all_step5_human_oversight_flags(self):
        from insurance_governance.euaia.article13 import Article13Document
        ca = self._base_ca()
        doc = Article13Document(
            provider_name="A", provider_contact="a@a.com",
            model_name="M", model_version="1.0", document_date="2026-01-01",
            intended_purpose="P",
        )
        # Intentionally leave human_oversight_measures empty
        ca.attach_article13(doc)
        ca.run_all()
        s5 = ca.get_step(5)
        assert any("oversight" in f.lower() for f in s5.findings)

    def test_to_dict_structure(self):
        ca = self._base_ca()
        d = ca.to_dict()
        assert "conformity_assessment" in d
        inner = d["conformity_assessment"]
        assert inner["model_name"] == "Life Propensity Model v2"
        assert len(inner["steps"]) == 7

    def test_assessment_step_is_complete_method(self):
        from insurance_governance.euaia.conformity import AssessmentStep, StepStatus
        step = AssessmentStep(1, "Test", "Art 1", status=StepStatus.COMPLETE)
        assert step.is_complete() is True
        step.status = StepStatus.INCOMPLETE
        assert step.is_complete() is False
        step.status = StepStatus.NOT_APPLICABLE
        assert step.is_complete() is True

    def test_assessment_step_to_dict(self):
        from insurance_governance.euaia.conformity import AssessmentStep, StepStatus
        step = AssessmentStep(1, "Risk Classification", "Art 6", status=StepStatus.COMPLETE,
                              evidence="Done", findings=["Note 1"])
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["status"] == "complete"
        assert d["findings"] == ["Note 1"]


# ============================================================================
# 8. outcome.results — OutcomeResult and OutcomeSuite
# ============================================================================

class TestOutcomeResult:

    def _make_result(self, passed=True, severity=None, **kwargs):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import Severity
        defaults = dict(
            outcome="price_value",
            test_name="fair_value_ratio",
            passed=passed,
            metric_value=0.75,
            threshold=0.70,
            period="2026-Q1",
            severity=severity or Severity.INFO,
        )
        defaults.update(kwargs)
        return OutcomeResult(**defaults)

    def test_valid_outcomes_accepted(self):
        for outcome in ["price_value", "claims", "support"]:
            r = self._make_result(outcome=outcome)
            assert r.outcome == outcome

    def test_invalid_outcome_raises(self):
        from insurance_governance.outcome.results import OutcomeResult
        from insurance_governance.validation.results import Severity
        with pytest.raises(ValueError, match="outcome"):
            OutcomeResult(
                outcome="invalid_outcome",
                test_name="test",
                passed=True,
                severity=Severity.INFO,
            )

    def test_to_dict_serialisable(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["outcome"] == "price_value"
        assert d["passed"] is True
        # Must be JSON-serialisable
        json.dumps(d)

    def test_to_dict_corrective_actions(self):
        from insurance_governance.validation.results import Severity
        r = self._make_result(
            passed=False,
            severity=Severity.CRITICAL,
            corrective_actions=["Action 1", "Action 2"],
        )
        d = r.to_dict()
        assert d["corrective_actions"] == ["Action 1", "Action 2"]


class TestOutcomeSuite:

    def _make_suite(self):
        from insurance_governance.outcome.results import OutcomeResult, OutcomeSuite
        from insurance_governance.validation.results import Severity, RAGStatus

        results = [
            OutcomeResult("price_value", "fvr", True, 0.8, 0.7, "2026-Q1", severity=Severity.INFO),
            OutcomeResult("claims", "settlement", False, 0.91, 0.95, "2026-Q1",
                          severity=Severity.WARNING),
            OutcomeResult("support", "nps", True, 0.72, 0.6, "2026-Q1", severity=Severity.INFO),
        ]
        return OutcomeSuite(results=results, period="2026-Q1")

    def test_passed_property(self):
        suite = self._make_suite()
        assert len(suite.passed) == 2

    def test_failed_property(self):
        suite = self._make_suite()
        assert len(suite.failed) == 1

    def test_warning_failures(self):
        from insurance_governance.validation.results import Severity
        suite = self._make_suite()
        wf = suite.warning_failures
        assert len(wf) == 1
        assert wf[0].severity == Severity.WARNING

    def test_critical_failures_empty(self):
        suite = self._make_suite()
        assert suite.critical_failures == []

    def test_rag_status_amber_when_warning(self):
        from insurance_governance.validation.results import RAGStatus
        suite = self._make_suite()
        assert suite.rag_status == RAGStatus.AMBER

    def test_rag_status_red_when_critical(self):
        from insurance_governance.outcome.results import OutcomeResult, OutcomeSuite
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("price_value", "fvr", False, 0.5, 0.7, "Q1",
                          severity=Severity.CRITICAL)
        ]
        suite = OutcomeSuite(results=results)
        assert suite.rag_status == RAGStatus.RED

    def test_rag_status_green_when_all_pass(self):
        from insurance_governance.outcome.results import OutcomeResult, OutcomeSuite
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("price_value", "fvr", True, 0.8, 0.7, "Q1", severity=Severity.INFO),
        ]
        suite = OutcomeSuite(results=results)
        assert suite.rag_status == RAGStatus.GREEN

    def test_by_outcome_filter(self):
        suite = self._make_suite()
        claims = suite.by_outcome("claims")
        assert all(r.outcome == "claims" for r in claims)

    def test_summary_counts(self):
        suite = self._make_suite()
        s = suite.summary()
        assert s["total"] == 3
        assert s["passed"] == 2
        assert s["failed"] == 1

    def test_to_dict_has_results_key(self):
        suite = self._make_suite()
        d = suite.to_dict()
        assert "results" in d
        assert len(d["results"]) == 3

    def test_vulnerable_segment_results(self):
        from insurance_governance.outcome.results import OutcomeResult, OutcomeSuite
        from insurance_governance.validation.results import Severity
        results = [
            OutcomeResult("claims", "t1", True, severity=Severity.INFO, segment="elderly_segment"),
            OutcomeResult("claims", "t2", True, severity=Severity.INFO, segment=None),
        ]
        suite = OutcomeSuite(results=results)
        seg_results = suite.vulnerable_segment_results()
        assert len(seg_results) == 1
        assert seg_results[0].segment == "elderly_segment"


# ============================================================================
# 9. outcome.metrics — PriceValueMetrics and ClaimsMetrics
# ============================================================================

class TestPriceValueMetrics:

    def test_fair_value_ratio_passes(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [1000.0] * 100
        claims = [750.0] * 100
        expenses = [100.0] * 100
        result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, "2026-Q1")
        assert result.passed is True
        assert result.metric_value == pytest.approx(0.75)

    def test_fair_value_ratio_fails_below_threshold(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [1000.0] * 100
        claims = [500.0] * 100
        expenses = [100.0] * 100
        result = PriceValueMetrics.fair_value_ratio(premiums, claims, expenses, "2026-Q1")
        assert result.passed is False
        assert result.metric_value == pytest.approx(0.50)

    def test_fair_value_ratio_zero_premium_fails_critical(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        result = PriceValueMetrics.fair_value_ratio([0.0], [1.0], [0.5], "2026-Q1")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.metric_value is None

    def test_fair_value_ratio_custom_threshold(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [1000.0] * 10
        claims = [620.0] * 10  # 62% ratio
        result = PriceValueMetrics.fair_value_ratio(premiums, claims, [0.0] * 10,
                                                     "Q1", threshold=0.60)
        assert result.passed is True
        result2 = PriceValueMetrics.fair_value_ratio(premiums, claims, [0.0] * 10,
                                                      "Q1", threshold=0.70)
        assert result2.passed is False

    def test_price_dispersion_single_segment_returns_single_info(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        premiums = [100.0, 110.0, 105.0]
        segments = ["A", "A", "A"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, segments, "Q1")
        assert len(results) == 1
        assert results[0].passed is True

    def test_price_dispersion_multiple_segments(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        rng = np.random.default_rng(1)
        premiums = list(rng.uniform(100, 120, 50)) + list(rng.uniform(200, 220, 50))
        segments = ["cheap"] * 50 + ["expensive"] * 50
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, segments, "Q1")
        # Should have 3 results: 2 per-segment + 1 summary
        assert len(results) == 3
        # Summary result should be last
        summary = results[-1]
        assert "dispersion" in summary.test_name
        assert summary.metric_value > 1.0  # expensive vs cheap ratio

    def test_price_dispersion_zero_minimum_median_fails_critical(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        from insurance_governance.validation.results import Severity
        premiums = [0.0, 0.0, 200.0, 200.0]
        segments = ["A", "A", "B", "B"]
        results = PriceValueMetrics.price_dispersion_by_segment(premiums, segments, "Q1")
        assert results[0].severity == Severity.CRITICAL

    def test_renewal_vs_new_business_gap_passes(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        renewal = [300.0] * 50
        nb = [295.0] * 50
        exposure = [1.0] * 50
        result = PriceValueMetrics.renewal_vs_new_business_gap(renewal, nb, exposure, "Q1")
        assert result.passed is True  # gap ~1.7% < 5%

    def test_renewal_vs_new_business_gap_fails_price_walking(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        renewal = [400.0] * 50
        nb = [300.0] * 50  # 33% higher
        exposure = [1.0] * 50
        result = PriceValueMetrics.renewal_vs_new_business_gap(renewal, nb, exposure, "Q1")
        assert result.passed is False
        assert result.metric_value > 5.0

    def test_renewal_vs_new_business_empty_arrays_fails(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.renewal_vs_new_business_gap([], [], [], "Q1")
        assert result.passed is False

    def test_renewal_vs_new_business_zero_exposure_fails(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [300.0], [295.0], [0.0], "Q1"
        )
        assert result.passed is False

    def test_renewal_vs_new_business_zero_nb_fails(self):
        from insurance_governance.outcome.metrics import PriceValueMetrics
        result = PriceValueMetrics.renewal_vs_new_business_gap(
            [300.0], [0.0], [1.0], "Q1"
        )
        assert result.passed is False


class TestClaimsMetrics:

    def test_settlement_value_adequacy_passes(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        agreed = [950.0, 980.0, 970.0, 960.0]
        reference = [1000.0, 1000.0, 1000.0, 1000.0]
        result = ClaimsMetrics.settlement_value_adequacy(agreed, reference, "Q1")
        assert result.passed is True
        assert result.metric_value == pytest.approx(0.965)

    def test_settlement_value_adequacy_fails_below_threshold(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        agreed = [800.0] * 10
        reference = [1000.0] * 10
        result = ClaimsMetrics.settlement_value_adequacy(agreed, reference, "Q1")
        assert result.passed is False
        assert result.metric_value == pytest.approx(0.80)

    def test_settlement_value_adequacy_empty_arrays_fails(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        result = ClaimsMetrics.settlement_value_adequacy([], [], "Q1")
        assert result.passed is False

    def test_settlement_value_adequacy_length_mismatch_fails(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        result = ClaimsMetrics.settlement_value_adequacy([100.0, 200.0], [100.0], "Q1")
        assert result.passed is False

    def test_settlement_value_adequacy_zero_reference_fails(self):
        from insurance_governance.outcome.metrics import ClaimsMetrics
        result = ClaimsMetrics.settlement_value_adequacy([100.0], [0.0], "Q1")
        assert result.passed is False


# ============================================================================
# 10. outcome.results — _compute_outcome_rag edge cases
# ============================================================================

class TestComputeOutcomeRag:

    def test_empty_results_returns_green(self):
        from insurance_governance.outcome.results import _compute_outcome_rag
        from insurance_governance.validation.results import RAGStatus
        assert _compute_outcome_rag([]) == RAGStatus.GREEN

    def test_info_failure_returns_green(self):
        from insurance_governance.outcome.results import OutcomeResult, _compute_outcome_rag
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("support", "nps", False, severity=Severity.INFO)
        ]
        assert _compute_outcome_rag(results) == RAGStatus.GREEN

    def test_warning_failure_returns_amber(self):
        from insurance_governance.outcome.results import OutcomeResult, _compute_outcome_rag
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("price_value", "fvr", False, severity=Severity.WARNING)
        ]
        assert _compute_outcome_rag(results) == RAGStatus.AMBER

    def test_critical_failure_returns_red(self):
        from insurance_governance.outcome.results import OutcomeResult, _compute_outcome_rag
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("claims", "settlement", False, severity=Severity.CRITICAL)
        ]
        assert _compute_outcome_rag(results) == RAGStatus.RED

    def test_critical_overrides_warning_returns_red(self):
        from insurance_governance.outcome.results import OutcomeResult, _compute_outcome_rag
        from insurance_governance.validation.results import Severity, RAGStatus
        results = [
            OutcomeResult("price_value", "fvr", False, severity=Severity.WARNING),
            OutcomeResult("claims", "settlement", False, severity=Severity.CRITICAL),
        ]
        assert _compute_outcome_rag(results) == RAGStatus.RED
