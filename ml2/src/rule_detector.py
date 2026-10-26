"""
Rule-Based Baseline Detector Module (Model 1).
Provides transparent, heuristic-driven risk scoring with explicit
balancing logic to prevent false positives on legitimate urgent/payroll/password notices.
"""

from __future__ import annotations
from typing import Dict, List, Any
from src.preprocessing import extract_cybersecurity_features


class RuleBasedDetector:
    """
    Transparent rule-based phishing and smishing detector.
    Evaluates both coercive risk cues and countervailing operational markers.
    """

    def __init__(self, risk_threshold: float = 0.50):
        self.risk_threshold = risk_threshold

    def evaluate(
        self,
        body_text: str,
        subject: str = "",
        sanitized_url_features: str = "",
        sender_domain_type: str = "unknown",
        channel: str = "email"
    ) -> Dict[str, Any]:
        """
        Evaluates a message and returns detailed rule match explanations and risk score.
        """
        features = extract_cybersecurity_features(
            text=body_text,
            subject=subject,
            url_features=sanitized_url_features
        )

        suspicious_cues: List[str] = []
        legitimate_cues: List[str] = []
        raw_score = 0.10  # Neutral baseline prior

        # --- High-Risk Phishing Indicators ---
        if features["threat_count"] > 0:
            raw_score += 0.35 * min(features["threat_count"], 2)
            suspicious_cues.append(
                f"Coercive threats detected ({features['threat_count']} instances, e.g. account termination/legal penalty)"
            )

        if features["credential_count"] > 0:
            raw_score += 0.35 * min(features["credential_count"], 2)
            suspicious_cues.append(
                f"Credential or sensitive data solicitation detected ({features['credential_count']} instances)"
            )

        if features.get("lottery_count", 0) > 0:
            raw_score += 0.45 * min(features["lottery_count"], 2)
            suspicious_cues.append(
                f"Prize, lottery, or unsolicited reward claim detected ({features['lottery_count']} instances)"
            )

        if features.get("call_action_count", 0) > 0:
            raw_score += 0.35 * min(features["call_action_count"], 2)
            suspicious_cues.append(
                f"High-risk action trigger detected ({features['call_action_count']} instances, e.g. dial number, text keyword, reply stop)"
            )

        if features.get("account_alert_count", 0) > 0:
            raw_score += 0.30
            suspicious_cues.append("Suspicious account/fraud lock notification pattern detected")

        if features["has_defanged_url"]:
            raw_score += 0.30
            suspicious_cues.append("Defanged or obfuscated URL pattern identified (e.g. hxxp, [.])")

        if features["has_ip_url"]:
            raw_score += 0.30
            suspicious_cues.append("Direct IP address host identified instead of legitimate domain name")

        if features["has_shortener"]:
            raw_score += 0.20
            suspicious_cues.append("URL shortener detected, masking true landing destination")

        if features["urgency_count"] > 0 and features["credential_count"] > 0:
            raw_score += 0.20
            suspicious_cues.append("Correlated urgency combined with credential request (classic social engineering lure)")
        elif features["urgency_count"] > 0:
            raw_score += 0.15
            suspicious_cues.append(f"Urgent action language detected ({features['urgency_count']} instances)")

        if features["currency_count"] > 0 and (features.get("lottery_count", 0) > 0 or features["urgency_count"] > 0):
            raw_score += 0.15
            suspicious_cues.append("Monetary amount referenced alongside urgency/reward lures")

        if sender_domain_type.lower() in ["lookalike_spoofed", "defanged_shortlink"]:
            raw_score += 0.30
            suspicious_cues.append(f"Suspicious or spoofed sender domain category: '{sender_domain_type}'")
        elif sender_domain_type.lower() == "external_freemail" and (features["credential_count"] > 0 or features["threat_count"] > 0):
            raw_score += 0.25
            suspicious_cues.append("External freemail address requesting organizational action or credentials")

        if features["uppercase_ratio"] > 0.35:
            raw_score += 0.15
            suspicious_cues.append(f"Unusually high uppercase character ratio ({features['uppercase_ratio']*100:.1f}%)")

        if features["exclamation_count"] >= 3:
            raw_score += 0.10
            suspicious_cues.append(f"Excessive exclamation punctuation ({features['exclamation_count']} exclamation marks)")

        # --- Channel-Specific SMS Smishing Indicators ---
        if channel.lower() == "sms":
            if (features["has_shortener"] or features["has_defanged_url"]) and features["urgency_count"] > 0:
                raw_score += 0.25
                suspicious_cues.append("SMS contains shortened link alongside urgency cue (smishing signature)")

        # --- Countervailing Legitimate Balancing Signals ---
        if features["legit_context_count"] > 0:
            discount = 0.25 * min(features["legit_context_count"], 2)
            raw_score -= discount
            legitimate_cues.append(
                f"Standard operational security phrasing present (e.g., 'if you did not request this, ignore', internal ticket ID)"
            )

        if features.get("benign_conv_count", 0) > 0:
            discount = 0.20 * min(features["benign_conv_count"], 2)
            raw_score -= discount
            legitimate_cues.append("Standard conversational/social context detected")

        if sender_domain_type.lower() in ["trusted_internal", "telecom_shortcode"]:
            raw_score -= 0.20
            legitimate_cues.append(f"Sender domain indicates authorized origin: '{sender_domain_type}'")

        has_high_risk_triggers = (
            features.get("lottery_count", 0) > 0 or
            features.get("call_action_count", 0) > 0 or
            features.get("threat_count", 0) > 0 or
            features.get("account_alert_count", 0) > 0 or
            features["has_defanged_url"] or
            features["has_ip_url"] or
            features["has_shortener"]
        )

        if features["credential_count"] == 0 and not has_high_risk_triggers:
            raw_score -= 0.15
            legitimate_cues.append("No credentials requested and no defanged/IP URLs present")

        # Clamp score between 0.0 and 1.0
        risk_score = round(max(0.0, min(1.0, raw_score)), 4)
        predicted_label = "phishing" if risk_score >= self.risk_threshold else "legitimate"

        return {
            "model_name": "Rule-Based Baseline Detector",
            "predicted_label": predicted_label,
            "risk_score": risk_score,
            "decision_confidence": abs(risk_score - self.risk_threshold) * 2,
            "is_phishing": predicted_label == "phishing",
            "suspicious_cues": suspicious_cues,
            "legitimate_cues": legitimate_cues,
            "extracted_features": features,
        }

    def predict(self, df_or_texts) -> List[str]:
        """Batch prediction returning list of string labels."""
        if hasattr(df_or_texts, "iterrows"):
            preds = []
            for _, row in df_or_texts.iterrows():
                res = self.evaluate(
                    body_text=row["body_text"],
                    subject=row.get("subject_or_title", ""),
                    sanitized_url_features=row.get("sanitized_url_features", ""),
                    sender_domain_type=row.get("sender_domain_type", "unknown"),
                    channel=row.get("channel", "email")
                )
                preds.append(res["predicted_label"])
            return preds
        elif isinstance(df_or_texts, list):
            return [self.evaluate(t)["predicted_label"] for t in df_or_texts]
        else:
            raise TypeError("Unsupported input type for batch prediction.")
