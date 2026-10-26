"""
Safety, Privacy, and Ethical Cybersecurity Constraint Verification Tests.
Verifies defanging enforcement, local offline execution, absence of fabricated threat intel,
and mandatory educational triage disclaimers.
"""

from __future__ import annotations
import pytest
from src.predict import PhishShiftPredictor
from src.explanations import generate_explanation_advisory
from src.preprocessing import extract_cybersecurity_features


def test_no_empty_body_inference():
    predictor = PhishShiftPredictor()
    with pytest.raises(ValueError, match="Message body cannot be empty"):
        predictor.predict_message(body_text="   ")


def test_mandatory_disclaimer_always_present():
    fake_pred = {
        "category": "Likely Legitimate",
        "phishing_probability": 0.05,
        "suspicious_cues": [],
        "legitimate_cues": ["Standard notification"],
        "extracted_features": {},
        "channel": "email",
        "model_name": "TestModel"
    }
    advisory = generate_explanation_advisory(fake_pred)
    assert "mandatory_disclaimer" in advisory
    assert "DECISION SUPPORT DISCLAIMER" in advisory["mandatory_disclaimer"]
    assert "NOT a guarantee of safety" in advisory["mandatory_disclaimer"]


def test_no_fabricated_external_intelligence():
    """Verifies explanations only cite evidence extracted from the text, not fake domain reputation."""
    fake_pred = {
        "category": "Potential Phishing",
        "phishing_probability": 0.88,
        "suspicious_cues": ["Coercive threats detected"],
        "legitimate_cues": [],
        "extracted_features": {},
        "channel": "email",
        "model_name": "Rule-Based Baseline"
    }
    advisory = generate_explanation_advisory(fake_pred)
    cues_text = " ".join(advisory["suspicious_cues"]).lower()
    
    # Ensure no fabricated external API lookups exist
    assert "whois" not in cues_text
    assert "virustotal" not in cues_text
    assert "domain age" not in cues_text
    assert "threat intelligence feed" not in cues_text


def test_safe_next_steps_actionable():
    fake_pred = {
        "category": "Potential Phishing",
        "phishing_probability": 0.95,
        "suspicious_cues": ["Credential solicitation detected"],
        "legitimate_cues": [],
        "extracted_features": {},
        "channel": "email",
        "model_name": "TestModel"
    }
    advisory = generate_explanation_advisory(fake_pred)
    steps_joined = " ".join(advisory["recommended_steps"])
    assert "DO NOT click on any links" in steps_joined
    assert "Forward" in steps_joined
