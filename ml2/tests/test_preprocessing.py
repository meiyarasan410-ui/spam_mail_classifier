"""
Unit tests for text preprocessing, token preservation, and cybersecurity signal extraction.
"""

from __future__ import annotations
import pytest
from src.preprocessing import (
    extract_cybersecurity_features,
    normalize_text_for_tfidf,
    combine_message_text,
)


def test_extract_defanged_url_features():
    text = "Please verify your account at hxxps://auth[.]secure-portal[.]net/login"
    features = extract_cybersecurity_features(text)
    assert features["has_defanged_url"] is True
    assert features["has_ip_url"] is False


def test_extract_ip_and_shortener():
    text = "Check your shipment at hxxp://198.51.100.42/track or bit[.]ly/pkg-938"
    features = extract_cybersecurity_features(text)
    assert features["has_ip_url"] is True
    assert features["has_shortener"] is True


def test_extract_urgency_and_credential_cues():
    text = "CRITICAL ALERT: Your account will be suspended within 24 hours. Enter your password immediately."
    features = extract_cybersecurity_features(text)
    assert features["urgency_count"] >= 1
    assert features["credential_count"] >= 1
    assert features["threat_count"] >= 1


def test_extract_legitimate_context_cues():
    text = "Incident INC-4092: Scheduled routine maintenance tonight. If you did not request this, ignore this email."
    features = extract_cybersecurity_features(text)
    assert features["legit_context_count"] >= 1
    assert features["credential_count"] == 0


def test_normalize_text_preserves_defanged_tokens():
    raw = "Visit hxxps://login[.]portal[.]com/verify now!"
    normalized = normalize_text_for_tfidf(raw, preserve_signals=True)
    assert "__defanged_link__" in normalized
    assert "!" not in normalized  # Punctuation cleaned


def test_combine_message_text():
    row = {
        "subject_or_title": "Account Alert",
        "body_text": "Please verify credentials.",
        "sanitized_url_features": "contains_defanged_url"
    }
    combined = combine_message_text(row)
    assert "Subject: Account Alert" in combined
    assert "Please verify credentials." in combined
    assert "URL_Features: contains_defanged_url" in combined
