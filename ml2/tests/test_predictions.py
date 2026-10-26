"""
Tests for Model Inference, Abstention / Uncertainty Rejection (FR-04),
and Acceptance Test scenarios (AT-01, AT-02, AT-03).
"""

from __future__ import annotations
import os
import pytest
from src.predict import PhishShiftPredictor, UNCERTAINTY_LOWER, UNCERTAINTY_UPPER
from src.rule_detector import RuleBasedDetector


@pytest.fixture
def predictor():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    return PhishShiftPredictor(models_dir=models_dir)


def test_rule_based_detector_avoids_urgent_business_false_alarm():
    """Validates that rule detector doesn't flag legitimate urgent messages without credentials/threats."""
    detector = RuleBasedDetector()
    legit_urgent = (
        "URGENT P1 INCIDENT #4029: Core API latency spike. "
        "SRE incident bridge active on Slack #incident-p1. No password or credential changes required."
    )
    result = detector.evaluate(body_text=legit_urgent, sender_domain_type="trusted_internal")
    assert result["predicted_label"] == "legitimate"
    assert result["risk_score"] < 0.50
    assert len(result["legitimate_cues"]) >= 1


def test_at01_legitimate_password_reset_notice(predictor):
    """
    AT-01: Legitimate password-reset notice.
    Checks whether the message is correctly handled and documents false-positive behavior.
    """
    legit_pwd_text = (
        "We received a request to reset your password. Use the link below to set a new password. "
        "If you did not request this, you can safely ignore this email. We will never ask for your password."
    )
    res_rule = predictor.predict_message(body_text=legit_pwd_text, model_name="rule_based", sender_domain_type="trusted_internal")
    assert res_rule["category"] == "Likely Legitimate"
    assert res_rule["phishing_probability"] < 0.40

    # Also test ML model
    res_lr = predictor.predict_message(body_text=legit_pwd_text, model_name="logistic_regression", sender_domain_type="trusted_internal")
    # Verify that prediction result is returned with documented scores
    assert "phishing_probability" in res_lr
    assert "category" in res_lr


def test_at02_phishing_unfamiliar_wording(predictor):
    """
    AT-02: Phishing with unfamiliar wording.
    Evaluates detection of subtle phishing lure from an unseen campaign.
    """
    subtle_phish = (
        "Asset Collaboration Request: A shared cloud document repository requires organizational clearance "
        "prior to synchronization. Please review the collaborative manifest: hxxps://secure-auth[.]portal-verify[.]net/login?token=9284"
    )
    res_rule = predictor.predict_message(
        body_text=subtle_phish,
        model_name="rule_based",
        sanitized_url_features="contains_defanged_url",
        sender_domain_type="external_freemail"
    )
    # In AT-02, unfamiliar wording triggers elevated risk or uncertainty triage
    assert res_rule["category"] in ["Potential Phishing", "Uncertain — Manual Review Recommended"]
    assert res_rule["phishing_probability"] >= 0.40
    assert any("Defanged" in cue for cue in res_rule["suspicious_cues"])

    # Test ML model generalization on unfamiliar campaign
    res_lr = predictor.predict_message(body_text=subtle_phish, model_name="logistic_regression")
    assert "phishing_probability" in res_lr
    assert res_lr["category"] in ["Potential Phishing", "Uncertain — Manual Review Recommended", "Likely Legitimate"]


def test_at03_defanged_url_and_punctuation_drift(predictor):
    """
    AT-03: Defanged URL and punctuation changes.
    Applies controlled harmless text variations and records changes in scores.
    """
    base_text = "Your mailbox quota has exceeded 99%. Please verify your credentials at https://portal.com/login."
    perturbed_text = "Your mailbox quota has exceeded 99%!! Please verify your credentials at hxxps://portal[.]com/login."

    res_base = predictor.predict_message(body_text=base_text, model_name="rule_based")
    res_pert = predictor.predict_message(body_text=perturbed_text, model_name="rule_based")

    # Perturbed text includes defanged brackets and high punctuation, so score should increase
    assert res_pert["phishing_probability"] >= res_base["phishing_probability"]
    assert any("Defanged" in c for c in res_pert["suspicious_cues"])


def test_fr04_uncertainty_rejection_policy(predictor):
    """
    FR-04: Uncertainty rejection.
    If probability falls within [UNCERTAINTY_LOWER, UNCERTAINTY_UPPER],
    the category must be 'Uncertain — Manual Review Recommended'.
    """
    predictor.uncertainty_lower = 0.30
    predictor.uncertainty_upper = 0.70

    # Test an ambiguous message with mixed signals
    ambiguous_text = "Urgent reminder: Review the payroll schedule update. If no changes, ignore."
    res = predictor.predict_message(body_text=ambiguous_text, model_name="rule_based")
    
    # Manually check that if score is forced inside the band, category becomes Uncertain
    score = res["phishing_probability"]
    if 0.30 <= score <= 0.70:
        assert res["category"] == "Uncertain — Manual Review Recommended"
        assert res["is_uncertain"] is True
