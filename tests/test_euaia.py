"""Tests for the insurance_governance.euaia subpackage.

Covers:
- AIActClassifier scope and risk determination
- Article13Document creation, field validation, gap detection
- Article13Document.compute_accuracy() and compute_subgroup_performance()
- Markdown rendering completeness
- ConformityAssessment step tracking and automated checks
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from insurance_governance.euaia import (
    AIActClassifier,
    Article13Document,
    AssessmentStep,
    ClassificationResult,
    ConformityAssessment,
    ModelType,
    RiskClassification,
    StepStatus,
    article13_to_html,
    render_article13_markdown,
    render_conformity_markdown,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def minimal_doc() -> Article13Document:
    """Minimally populated Article13Document for gap-detection tests."""
    return Article13Document(
        provider_name="Burning Cost Ltd",
        provider_contact="mlops@burningcost.com",
        model_name="Life Propensity XGB",
        model_version="2.1.0",
        document_date="2025-11-01",
        intended_purpose="Predict propensity to lapse for UK life policies.",
        out_of_scope_uses=["Credit decisioning", "Employment screening"],
        known_risks=["Model may underperform on new customer segments."],
        explanation_tools=["SHAP TreeExplainer"],
        input_features=[
            {
                "name": "age",
                "type": "integer",
                "range": "18-99",
                "missing_handling": "median imputation",
            }
        ],
        output_interpretation_guide=(
            "Output is a probability in [0,1]. Values above 0.6 indicate "
            "elevated lapse risk."
        ),
        human_oversight_measures=["Weekly model performance review by ML team."],
        override_procedure="Underwriter can override via decision portal.",
        anomaly_thresholds={"ae_ratio": 0.15},
        expected_lifetime_months=24,
        retraining_triggers=["AE ratio outside [0.9, 1.1] for 2 consecutive weeks."],
        monitoring_metrics=["gini", "ae_ratio", "psi"],
    )


@pytest.fixture()
def rng_arrays():
    """Reproducible synthetic y_true / y_pred arrays."""
    rng = np.random.default_rng(99)
    n = 500
    y_true = rng.poisson(0.08, size=n).astype(float)
    # Correlated predictions — model has some discrimination
    y_pred = 0.08 + 0.3 * (y_true - 0.08) + rng.normal(0, 0.02, size=n)
    y_pred = np.clip(y_pred, 0.001, 1.0)
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# AIActClassifier                                                               #
# --------------------------------------------------------------------------- #


class TestAIActClassifier:
    def setup_method(self):
        self.clf = AIActClassifier()

    def test_motor_gradient_boosting_not_high_risk(self):
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="motor",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.NOT_HIGH_RISK
        assert result.is_ai_system is True
        assert result.requires_conformity_assessment is False
        assert result.assessment_route == "n/a"

    def test_property_neural_network_not_high_risk(self):
        result = self.clf.classify(
            model_type="neural_network",
            line_of_business="property",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.NOT_HIGH_RISK

    def test_life_gradient_boosting_high_risk(self):
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK
        assert result.is_ai_system is True
        assert result.requires_conformity_assessment is True
        assert result.assessment_route == "internal_control"

    def test_health_random_forest_high_risk(self):
        result = self.clf.classify(
            model_type="random_forest",
            line_of_business="health",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_pmi_other_ml_high_risk(self):
        result = self.clf.classify(
            model_type="other_ml",
            line_of_business="private_medical_insurance",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_life_glm_out_of_scope(self):
        """GLMs are out of scope per EC Guidelines C/2025/3554 §42."""
        result = self.clf.classify(
            model_type="glm",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.OUT_OF_SCOPE
        assert result.is_ai_system is False
        assert result.requires_conformity_assessment is False

    def test_motor_glm_out_of_scope(self):
        result = self.clf.classify(
            model_type="glm",
            line_of_business="motor",
            uses_personal_data=True,
            automated_decision=False,
        )
        assert result.risk_classification == RiskClassification.OUT_OF_SCOPE

    def test_rule_based_out_of_scope(self):
        result = self.clf.classify(
            model_type="rule_based",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.OUT_OF_SCOPE

    def test_life_no_personal_data_potentially_high_risk(self):
        """Life + ML but no personal data -> potentially high-risk, not confirmed."""
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="life",
            uses_personal_data=False,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.POTENTIALLY_HIGH_RISK

    def test_life_no_automated_decision_potentially_high_risk(self):
        """Life + ML + personal data but human review -> potentially high-risk."""
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=False,
        )
        assert result.risk_classification == RiskClassification.POTENTIALLY_HIGH_RISK

    def test_unknown_model_type_treated_as_ai_system(self):
        result = self.clf.classify(
            model_type="some_exotic_model",
            line_of_business="motor",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.is_ai_system is True
        # Motor is not high-risk regardless
        assert result.risk_classification == RiskClassification.NOT_HIGH_RISK
        assert len(result.warnings) > 0

    def test_rationale_populated(self):
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert len(result.rationale) >= 3

    def test_classification_result_is_dataclass(self):
        result = self.clf.classify(
            model_type="glm",
            line_of_business="motor",
            uses_personal_data=False,
            automated_decision=False,
        )
        assert isinstance(result, ClassificationResult)

    def test_case_insensitive_model_type(self):
        result = self.clf.classify(
            model_type="GRADIENT_BOOSTING",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_term_life_recognised_as_high_risk_lob(self):
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="term_life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK

    def test_annuity_recognised_as_high_risk_lob(self):
        result = self.clf.classify(
            model_type="gradient_boosting",
            line_of_business="annuity",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK


# --------------------------------------------------------------------------- #
# Article13Document — creation and fields                                      #
# --------------------------------------------------------------------------- #


class TestArticle13DocumentCreation:
    def test_default_instantiation(self):
        doc = Article13Document()
        assert doc.provider_name == ""
        assert doc.out_of_scope_uses == []
        assert doc.accuracy_metrics == {}

    def test_full_instantiation(self, minimal_doc):
        assert minimal_doc.provider_name == "Burning Cost Ltd"
        assert minimal_doc.model_name == "Life Propensity XGB"
        assert len(minimal_doc.input_features) == 1

    def test_to_dict_structure(self, minimal_doc):
        d = minimal_doc.to_dict()
        assert "article13_transparency_document" in d
        root = d["article13_transparency_document"]
        assert "a_provider" in root
        assert "b_performance" in root
        assert "c_planned_changes" in root
        assert "d_human_oversight" in root
        assert "e_maintenance" in root

    def test_to_dict_provider_fields(self, minimal_doc):
        d = minimal_doc.to_dict()
        provider = d["article13_transparency_document"]["a_provider"]
        assert provider["provider_name"] == "Burning Cost Ltd"
        assert provider["model_version"] == "2.1.0"

    def test_to_dict_is_serialisable(self, minimal_doc):
        import json
        d = minimal_doc.to_dict()
        # Should not raise
        json.dumps(d)


# --------------------------------------------------------------------------- #
# Article13Document — gap detection                                            #
# --------------------------------------------------------------------------- #


class TestArticle13GapDetection:
    def test_no_gaps_on_complete_doc(self, minimal_doc):
        # Manually set accuracy metrics to avoid that gap
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        gaps = minimal_doc.flag_gaps()
        assert gaps == [], f"Unexpected gaps: {gaps}"

    def test_empty_doc_has_many_gaps(self):
        doc = Article13Document()
        gaps = doc.flag_gaps()
        assert len(gaps) >= 8

    def test_missing_provider_name_flagged(self):
        doc = Article13Document(
            provider_contact="x@y.com",
            model_name="Test",
            model_version="1",
            document_date="2025-01-01",
            intended_purpose="Testing",
        )
        gaps = doc.flag_gaps()
        gap_text = " ".join(gaps)
        assert "provider_name" in gap_text

    def test_missing_accuracy_metrics_flagged(self, minimal_doc):
        # minimal_doc has no accuracy_metrics yet
        gaps = minimal_doc.flag_gaps()
        gap_text = " ".join(gaps)
        assert "accuracy metrics" in gap_text.lower()

    def test_zero_lifetime_flagged(self, minimal_doc):
        minimal_doc.expected_lifetime_months = 0
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        gaps = minimal_doc.flag_gaps()
        assert any("expected_lifetime_months" in g for g in gaps)

    def test_empty_known_risks_flagged(self, minimal_doc):
        minimal_doc.known_risks = []
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        gaps = minimal_doc.flag_gaps()
        assert any("known_risks" in g for g in gaps)

    def test_empty_monitoring_metrics_flagged(self, minimal_doc):
        minimal_doc.monitoring_metrics = []
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        gaps = minimal_doc.flag_gaps()
        assert any("monitoring_metrics" in g for g in gaps)


# --------------------------------------------------------------------------- #
# Article13Document — compute_accuracy                                         #
# --------------------------------------------------------------------------- #


class TestComputeAccuracy:
    def test_basic_gini_returned(self, rng_arrays):
        y_true, y_pred = rng_arrays
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred)
        assert "gini" in metrics
        assert -1.0 <= metrics["gini"] <= 1.0

    def test_ae_ratio_computed(self, rng_arrays):
        y_true, y_pred = rng_arrays
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred)
        assert "ae_ratio" in metrics
        assert metrics["ae_ratio"] > 0

    def test_confidence_interval_keys(self, rng_arrays):
        y_true, y_pred = rng_arrays
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred, ci=0.95)
        assert "gini_ci_lo_95" in metrics
        assert "gini_ci_hi_95" in metrics
        assert metrics["gini_ci_lo_95"] <= metrics["gini"] <= metrics["gini_ci_hi_95"]

    def test_metrics_stored_on_instance(self, rng_arrays):
        y_true, y_pred = rng_arrays
        doc = Article13Document()
        doc.compute_accuracy(y_true, y_pred)
        assert len(doc.accuracy_metrics) > 0

    def test_with_exposure_weights(self, rng_arrays):
        y_true, y_pred = rng_arrays
        exposure = np.ones(len(y_true)) * 0.5
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred, exposure=exposure)
        assert "ae_ratio" in metrics
        # AE with half-exposure should be ~2x the unweighted AE
        metrics_no_exp = Article13Document().compute_accuracy(y_true, y_pred)
        ratio = metrics["ae_ratio"] / metrics_no_exp["ae_ratio"]
        assert abs(ratio - 2.0) < 0.01

    def test_n_obs_correct(self, rng_arrays):
        y_true, y_pred = rng_arrays
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred)
        assert metrics["n_obs"] == float(len(y_true))

    def test_mismatched_arrays_raises(self):
        doc = Article13Document()
        with pytest.raises(ValueError, match="shape"):
            doc.compute_accuracy([1, 2, 3], [1, 2])

    def test_perfect_model_gini_near_one(self):
        """A model that perfectly predicts ranks has Gini near 1."""
        y_true = np.array([0.0] * 50 + [1.0] * 50)
        y_pred = np.array([0.01] * 50 + [0.99] * 50)
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred, n_boot=50)
        assert metrics["gini"] > 0.95

    def test_random_model_gini_near_zero(self):
        """A random model has Gini near 0."""
        rng = np.random.default_rng(7)
        y_true = rng.binomial(1, 0.5, 1000).astype(float)
        y_pred = rng.uniform(0, 1, 1000)
        doc = Article13Document()
        metrics = doc.compute_accuracy(y_true, y_pred, n_boot=50)
        assert abs(metrics["gini"]) < 0.15


# --------------------------------------------------------------------------- #
# Article13Document — compute_subgroup_performance                             #
# --------------------------------------------------------------------------- #


class TestComputeSubgroupPerformance:
    def test_basic_subgroup_computed(self, rng_arrays):
        y_true, y_pred = rng_arrays
        mask_lo = y_pred < np.median(y_pred)
        doc = Article13Document()
        results = doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"low_pred": mask_lo, "high_pred": ~mask_lo},
        )
        assert "low_pred" in results
        assert "high_pred" in results
        for g in results.values():
            assert "gini" in g
            assert "ae_ratio" in g
            assert "n_obs" in g

    def test_small_group_excluded(self, rng_arrays):
        y_true, y_pred = rng_arrays
        # Only 10 rows — below the default min_group_size of 100
        small_mask = np.zeros(len(y_true), dtype=bool)
        small_mask[:10] = True
        large_mask = ~small_mask
        doc = Article13Document()
        results = doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"tiny": small_mask, "large": large_mask},
            min_group_size=100,
        )
        assert "tiny" not in results
        assert "large" in results

    def test_excluded_group_logged_in_limitations(self, rng_arrays):
        y_true, y_pred = rng_arrays
        small_mask = np.zeros(len(y_true), dtype=bool)
        small_mask[:5] = True
        doc = Article13Document()
        doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"tiny": small_mask},
            min_group_size=100,
        )
        assert any("tiny" in lim for lim in doc.known_accuracy_limitations)

    def test_results_stored_on_instance(self, rng_arrays):
        y_true, y_pred = rng_arrays
        mask = y_pred < np.median(y_pred)
        doc = Article13Document()
        doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"half": mask},
            min_group_size=10,
        )
        assert "half" in doc.subgroup_performance

    def test_custom_min_group_size(self, rng_arrays):
        y_true, y_pred = rng_arrays
        mask = np.zeros(len(y_true), dtype=bool)
        mask[:50] = True  # 50 rows — above threshold of 30
        doc = Article13Document()
        results = doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"medium": mask},
            min_group_size=30,
        )
        assert "medium" in results


# --------------------------------------------------------------------------- #
# Markdown rendering                                                           #
# --------------------------------------------------------------------------- #


class TestMarkdownRendering:
    def test_render_produces_string(self, minimal_doc):
        md = minimal_doc.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100

    def test_all_sections_present(self, minimal_doc):
        md = minimal_doc.to_markdown()
        assert "Article 13(3)(a)" in md
        assert "Article 13(3)(b)" in md
        assert "(b)(i)" in md
        assert "(b)(ii)" in md
        assert "(b)(iii)" in md
        assert "(b)(iv)" in md
        assert "(b)(v)" in md
        assert "(b)(vi)" in md
        assert "(b)(vii)" in md
        assert "Article 13(3)(c)" in md
        assert "Article 13(3)(d)" in md
        assert "Article 13(3)(e)" in md

    def test_provider_name_in_output(self, minimal_doc):
        md = minimal_doc.to_markdown()
        assert "Burning Cost Ltd" in md

    def test_model_name_in_output(self, minimal_doc):
        md = minimal_doc.to_markdown()
        assert "Life Propensity XGB" in md

    def test_gap_warning_shown_when_incomplete(self):
        doc = Article13Document(model_name="Incomplete Model")
        md = doc.to_markdown()
        assert "gaps detected" in md.lower() or "gap" in md.lower()

    def test_accuracy_metrics_table_rendered(self, minimal_doc, rng_arrays):
        y_true, y_pred = rng_arrays
        minimal_doc.compute_accuracy(y_true, y_pred, n_boot=50)
        md = minimal_doc.to_markdown()
        assert "gini" in md

    def test_subgroup_table_rendered(self, minimal_doc, rng_arrays):
        y_true, y_pred = rng_arrays
        mask = y_pred < np.median(y_pred)
        minimal_doc.compute_subgroup_performance(
            y_true, y_pred,
            groups={"age_u40": mask, "age_40plus": ~mask},
            min_group_size=10,
        )
        md = minimal_doc.to_markdown()
        assert "age_u40" in md
        assert "age_40plus" in md

    def test_render_article13_markdown_function(self, minimal_doc):
        md = render_article13_markdown(minimal_doc)
        assert "Article 13" in md

    def test_html_render_produces_html(self, minimal_doc):
        html = article13_to_html(minimal_doc)
        assert "<html" in html
        assert "<body>" in html
        assert "Life Propensity XGB" in html


# --------------------------------------------------------------------------- #
# ConformityAssessment                                                         #
# --------------------------------------------------------------------------- #


class TestConformityAssessment:
    def setup_method(self):
        self.ca = ConformityAssessment(
            model_name="Life Propensity XGB 2.1.0",
            assessor_name="Model Governance Team",
            assessment_date="2025-11-01",
        )

    def test_seven_steps_initialised(self):
        assert len(self.ca.steps) == 7

    def test_step_numbers_are_one_to_seven(self):
        nums = [s.step_number for s in self.ca.steps]
        assert nums == list(range(1, 8))

    def test_get_step_returns_correct_step(self):
        step = self.ca.get_step(3)
        assert step.step_number == 3
        assert step.title == "Technical Documentation"

    def test_get_step_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self.ca.get_step(0)
        with pytest.raises(ValueError):
            self.ca.get_step(8)

    def test_all_steps_initially_incomplete(self):
        for step in self.ca.steps:
            assert step.status == StepStatus.INCOMPLETE

    def test_flag_incomplete_returns_all_seven_initially(self):
        gaps = self.ca.flag_incomplete()
        assert len(gaps) == 7

    def test_complete_step_removes_from_gaps(self):
        step = self.ca.get_step(1)
        step.status = StepStatus.COMPLETE
        step.evidence = "Classification: HIGH_RISK confirmed."
        gaps = self.ca.flag_incomplete()
        assert not any("Step 1" in g for g in gaps)
        assert len(gaps) == 6

    def test_not_applicable_step_removes_from_gaps(self):
        step = self.ca.get_step(7)
        step.status = StepStatus.NOT_APPLICABLE
        step.evidence = "Not yet deployed."
        gaps = self.ca.flag_incomplete()
        assert not any("Step 7" in g for g in gaps)

    def test_overall_status_incomplete_when_any_step_incomplete(self):
        assert self.ca.overall_status() == StepStatus.INCOMPLETE

    def test_overall_status_complete_when_all_steps_done(self):
        for step in self.ca.steps:
            step.status = StepStatus.COMPLETE
            step.evidence = "Done."
        assert self.ca.overall_status() == StepStatus.COMPLETE

    def test_to_dict_structure(self):
        d = self.ca.to_dict()
        assert "conformity_assessment" in d
        root = d["conformity_assessment"]
        assert root["model_name"] == "Life Propensity XGB 2.1.0"
        assert len(root["steps"]) == 7

    def test_to_dict_step_has_required_keys(self):
        d = self.ca.to_dict()
        step = d["conformity_assessment"]["steps"][0]
        assert "step_number" in step
        assert "title" in step
        assert "status" in step
        assert "evidence" in step
        assert "findings" in step

    def test_run_all_without_article13_returns_dict(self):
        findings = self.ca.run_all()
        assert isinstance(findings, dict)
        # Should have entries for manual steps at minimum
        assert len(findings) > 0

    def test_run_all_with_complete_article13_reduces_findings(self, minimal_doc):
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        self.ca.attach_article13(minimal_doc)
        findings = self.ca.run_all()
        # Steps 3, 5, 6 should have no findings if doc is complete
        step3_findings = findings.get("Technical Documentation", [])
        # Step 3 should be empty (no gaps in minimal_doc with metrics filled)
        assert step3_findings == []

    def test_run_all_flags_missing_accuracy_metrics(self, minimal_doc):
        # No accuracy metrics on the document
        assert not minimal_doc.accuracy_metrics
        self.ca.attach_article13(minimal_doc)
        findings = self.ca.run_all()
        step6_findings = findings.get("Accuracy, Robustness and Cybersecurity", [])
        assert any("accuracy" in f.lower() for f in step6_findings)

    def test_run_all_flags_missing_override_procedure(self, minimal_doc):
        minimal_doc.accuracy_metrics = {"gini": 0.35}
        minimal_doc.override_procedure = ""  # Remove it
        self.ca.attach_article13(minimal_doc)
        findings = self.ca.run_all()
        step5_findings = findings.get("Human Oversight Design", [])
        assert any("override" in f.lower() for f in step5_findings)

    def test_render_conformity_markdown_function(self):
        md = render_conformity_markdown(self.ca)
        assert "Annex VI" in md

    def test_to_markdown_contains_all_step_titles(self):
        md = self.ca.to_markdown()
        assert "Risk Classification" in md
        assert "Quality Management System" in md
        assert "Technical Documentation" in md
        assert "Risk Management System" in md
        assert "Human Oversight Design" in md
        assert "Accuracy, Robustness and Cybersecurity" in md
        assert "Declaration of Conformity" in md

    def test_to_markdown_shows_incomplete_status(self):
        md = self.ca.to_markdown()
        assert "INCOMPLETE" in md

    def test_to_markdown_shows_complete_status_after_completion(self):
        for step in self.ca.steps:
            step.status = StepStatus.COMPLETE
            step.evidence = "Verified."
        md = self.ca.to_markdown()
        assert "COMPLETE" in md


# --------------------------------------------------------------------------- #
# AssessmentStep unit tests                                                    #
# --------------------------------------------------------------------------- #


class TestAssessmentStep:
    def test_is_complete_when_complete(self):
        step = AssessmentStep(1, "Test", "Art. 1", StepStatus.COMPLETE, "Evidence.")
        assert step.is_complete() is True

    def test_is_complete_when_not_applicable(self):
        step = AssessmentStep(1, "Test", "Art. 1", StepStatus.NOT_APPLICABLE)
        assert step.is_complete() is True

    def test_is_complete_when_incomplete(self):
        step = AssessmentStep(1, "Test", "Art. 1", StepStatus.INCOMPLETE)
        assert step.is_complete() is False

    def test_to_dict_keys(self):
        step = AssessmentStep(2, "QMS", "Art. 17", StepStatus.COMPLETE, "QMS in place.")
        d = step.to_dict()
        assert d["step_number"] == 2
        assert d["status"] == "complete"
        assert d["evidence"] == "QMS in place."


# --------------------------------------------------------------------------- #
# Integration: full workflow                                                   #
# --------------------------------------------------------------------------- #


class TestIntegration:
    def test_classifier_to_assessment_workflow(self, minimal_doc, rng_arrays):
        """Full workflow: classify -> compute metrics -> attach to assessment."""
        y_true, y_pred = rng_arrays

        clf = AIActClassifier()
        result = clf.classify(
            model_type="gradient_boosting",
            line_of_business="life",
            uses_personal_data=True,
            automated_decision=True,
        )
        assert result.risk_classification == RiskClassification.HIGH_RISK

        minimal_doc.compute_accuracy(y_true, y_pred, n_boot=100)
        assert "gini" in minimal_doc.accuracy_metrics

        ca = ConformityAssessment(
            model_name=minimal_doc.model_name,
            assessor_name="Governance Team",
            assessment_date="2025-11-01",
        )
        s1 = ca.get_step(1)
        s1.evidence = f"Classification: {result.risk_classification.value}"
        s1.status = StepStatus.COMPLETE

        ca.attach_article13(minimal_doc)
        ca.run_all()

        md = ca.to_markdown()
        assert "Life Propensity XGB" in md

        art_md = minimal_doc.to_markdown()
        assert "Burning Cost Ltd" in art_md

    def test_top_level_imports(self):
        """Ensure all key classes are importable from insurance_governance."""
        from insurance_governance import (
            AIActClassifier,
            Article13Document,
            ClassificationResult,
            ConformityAssessment,
            AssessmentStep,
            RiskClassification,
            ModelType,
            StepStatus,
        )
        assert AIActClassifier is not None
        assert Article13Document is not None
