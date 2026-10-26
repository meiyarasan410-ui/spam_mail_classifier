"""
Feature Extraction and Text Normalization Module for PhishShift.
Preserves critical cybersecurity signals (defanged URLs, urgency, punctuation)
while safely sanitizing and preparing text for ML and heuristic inspection.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List


# Regex patterns for cybersecurity signal detection
DEFANGED_URL_PATTERN = re.compile(r"(hxxps?://|hXXps?://|\[\.\]|\(\.\)|\[dot\]|\(dot\)|www\[\.\])", re.IGNORECASE)
STANDARD_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
IP_URL_PATTERN = re.compile(r"\b(?:\d{1,3}\[\.\]\d{1,3}\[\.\]\d{1,3}\[\.\]\d{1,3}|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")
SHORTENER_PATTERN = re.compile(r"\b(bit\[\.\]ly|tinyurl\[\.\]com|t\[\.\]co|is\[\.\]gd|ow\[\.\]ly|rb\[\.\]gy|bit\.ly|tinyurl\.com|t\.co|is\.gd|ow\.ly|rb\.gy)\b", re.IGNORECASE)

URGENCY_PATTERNS = [
    r"\b(immediately|urgent|within 24 hours|within 12 hours|immediate action|act now|suspended|suspension|terminated|terminating|final notice|account lockout|locked out|restricted)\b",
    r"\b(expires in|time-sensitive|critical alert|unauthorized access|security breach|unusual activity detected)\b",
]

CREDENTIAL_PATTERNS = [
    r"\b(enter your password|verify your account|confirm your identity|update billing|update payment|update direct deposit|re-enter credentials|login credentials|provide ssn|provide password)\b",
    r"\b(click here to verify|validate your mailbox|quota exceeded|increase mailbox quota|reactivate your account)\b",
]

COERCIVE_THREAT_PATTERNS = [
    r"\b(will be suspended|will be deleted|will be terminated|permanent closure|legal action|arrest warrant|law enforcement|subpoena|penalty fee)\b",
    r"\b(irreversible|cease and desist|account will be disabled|account termination|account suspension)\b",
]

SPAM_LOTTERY_PATTERNS = [
    r"\b(winner|won|claim your prize|free entry|gift card|reward|prize reward|selected to receive|claim code|congratulations you won|cash prize|free ringtone|voucher|jackpot|bonus prize)\b",
]

CALL_ACTION_PATTERNS = [
    r"\b(call \d{6,}|text [a-z0-9]+ to \d{4,}|reply stop|to claim call|call now|text win|dial \d{4,})\b",
]

ACCOUNT_ALERT_PATTERNS = [
    r"\b(account has been locked|unusual activity|suspicious charge|anomalous sign-in|temporary lock|debit charge of|unrecognized wire|security team has flagged)\b",
]

BENIGN_CONVERSATION_PATTERNS = [
    r"\b(lunch|dinner|coffee|weekend|how are you|see you|tomorrow|tonight|thanks|thank you|sounds good|let me know|are you free|meeting at|talk to you later|great job)\b",
]

LEGITIMATE_CONTEXT_PATTERNS = [
    r"\b(if you did not request this|no action is required|ignore this email|security tip|never share your password|routine maintenance|scheduled maintenance)\b",
    r"\b(ticket\s*#?\s*[A-Z0-9-]+|incident\s*#?\s*[A-Z0-9-]+|jira|confluence|slack|gitlab|github|workday|service desk)\b",
    r"\b(monthly pay stub|benefits enrollment|scheduled payroll|quarterly review|daily standup|meeting notes|agenda attached)\b",
]


def extract_cybersecurity_features(text: str, subject: str = "", url_features: str = "") -> Dict[str, Any]:
    """
    Extracts transparent cybersecurity signals and heuristics from message content.
    Returns quantitative indicators for both risk cues and legitimate balancing factors.
    """
    combined_raw = f"{subject} {text} {url_features}".strip()
    
    # 1. Defanged and suspicious URL indicators
    has_defanged = bool(DEFANGED_URL_PATTERN.search(combined_raw)) or "defanged" in url_features.lower()
    has_ip = bool(IP_URL_PATTERN.search(combined_raw)) or "ip_based" in url_features.lower()
    has_shortener = bool(SHORTENER_PATTERN.search(combined_raw)) or "shortener" in url_features.lower()
    has_standard_url = bool(STANDARD_URL_PATTERN.search(combined_raw))

    # 2. Urgency and Threat scores
    urgency_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in URGENCY_PATTERNS)
    credential_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in CREDENTIAL_PATTERNS)
    threat_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in COERCIVE_THREAT_PATTERNS)
    lottery_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in SPAM_LOTTERY_PATTERNS)
    call_action_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in CALL_ACTION_PATTERNS)
    account_alert_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in ACCOUNT_ALERT_PATTERNS)
    
    # 3. Legitimate Balancing context
    legit_context_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in LEGITIMATE_CONTEXT_PATTERNS)
    benign_conv_count = sum(len(re.findall(p, combined_raw, re.IGNORECASE)) for p in BENIGN_CONVERSATION_PATTERNS)

    # 4. Stylistic signals
    text_alpha = [c for c in combined_raw if c.isalpha()]
    total_alpha = len(text_alpha)
    upper_alpha = sum(1 for c in text_alpha if c.isupper())
    uppercase_ratio = (upper_alpha / total_alpha) if total_alpha > 0 else 0.0

    exclamation_count = combined_raw.count("!")
    question_count = combined_raw.count("?")
    currency_count = len(re.findall(r"[\$€£₹]", combined_raw))

    return {
        "has_defanged_url": has_defanged,
        "has_ip_url": has_ip,
        "has_shortener": has_shortener,
        "has_standard_url": has_standard_url,
        "urgency_count": urgency_count,
        "credential_count": credential_count,
        "threat_count": threat_count,
        "lottery_count": lottery_count,
        "call_action_count": call_action_count,
        "account_alert_count": account_alert_count,
        "legit_context_count": legit_context_count,
        "benign_conv_count": benign_conv_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "currency_count": currency_count,
        "char_length": len(combined_raw),
    }


def normalize_text_for_tfidf(text: str, preserve_signals: bool = True) -> str:
    """
    Carefully normalizes text for TF-IDF feature extraction while preserving
    critical cybersecurity tokens (defanged links, IP addresses, currencies).
    """
    if not text or not isinstance(text, str):
        return ""

    normalized = text

    if preserve_signals:
        # Convert defanged URLs to semantic tokens
        normalized = DEFANGED_URL_PATTERN.sub(" __defanged_link__ ", normalized)
        normalized = SHORTENER_PATTERN.sub(" __shortener_link__ ", normalized)
        normalized = IP_URL_PATTERN.sub(" __ip_host__ ", normalized)
        normalized = STANDARD_URL_PATTERN.sub(" __standard_url__ ", normalized)
        
        # Preserve currency and numbers
        normalized = re.sub(r"[\$€£₹]\s*\d+(?:,\d{3})*(?:\.\d+)?", " __currency_amount__ ", normalized)
        normalized = re.sub(r"\b\d{4,}\b", " __long_number__ ", normalized)

    # Lowercase
    normalized = normalized.lower()

    # Clean punctuation while keeping hyphens inside words
    normalized = re.sub(r"[^\w\s_]", " ", normalized)
    # Collapse multiple whitespaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def combine_message_text(row_or_dict: Dict[str, Any]) -> str:
    """Combines subject, body, and sanitized URL features into a single unified text string."""
    subject = str(row_or_dict.get("subject_or_title", "")).strip()
    body = str(row_or_dict.get("body_text", "")).strip()
    urls = str(row_or_dict.get("sanitized_url_features", "")).strip()
    
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    if urls and urls.lower() != "none":
        parts.append(f"URL_Features: {urls}")
        
    return " \n ".join(parts)
