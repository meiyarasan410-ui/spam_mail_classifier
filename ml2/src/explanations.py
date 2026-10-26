"""
Explanations and Cybersecurity Advisory Module (FR-04).
Generates strictly evidence-grounded risk cues, countervailing legitimate factors,
actionable next steps, and transparent safety disclaimers without fabricating external intelligence.
"""

from __future__ import annotations
from typing import Dict, List, Any


def generate_explanation_advisory(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms prediction results into clear, safe, non-fabricated user-facing explanations.
    """
    category = prediction_result.get("category", "Uncertain")
    phishing_prob = prediction_result.get("phishing_probability", 0.5)
    suspicious_cues = prediction_result.get("suspicious_cues", [])
    legitimate_cues = prediction_result.get("legitimate_cues", [])
    extracted_features = prediction_result.get("extracted_features", {})
    channel = prediction_result.get("channel", "email")

    # 1. Summary Narrative
    if category == "Potential Phishing":
        summary_narrative = (
            f"The scanner flagged this {channel.upper()} message as Potential Phishing with a "
            f"calculated risk score of {phishing_prob*100:.1f}%. The message contains linguistic "
            "or structural patterns commonly employed in social engineering, urgency lures, or credential harvesting."
        )
    elif category == "Likely Legitimate":
        summary_narrative = (
            f"The scanner classified this {channel.upper()} message as Likely Legitimate with a "
            f"low estimated risk score of {phishing_prob*100:.1f}%. It matches standard business or notification patterns "
            "and lacks classic coercive threat or credential-harvesting indicators."
        )
    else:
        summary_narrative = (
            f"The scanner returned an Uncertain classification (risk score: {phishing_prob*100:.1f}%). "
            "The message exhibits conflicting signals (e.g. urgent operational terms combined with legitimate business context) "
            "that fall into the designated uncertainty margin. Manual review is recommended."
        )

    # 2. Recommended Safe Next Steps
    recommended_steps: List[str] = []
    
    if category == "Potential Phishing" or category == "Uncertain — Manual Review Recommended":
        recommended_steps.append("DO NOT click on any links or download attached files.")
        recommended_steps.append(
            "Do NOT reply to the sender, even to request removal or ask if the message is authentic."
        )
        recommended_steps.append(
            "Navigate independently to the organization's official website or service portal by typing the known URL into a new browser tab."
        )
        recommended_steps.append(
            "Forward the suspicious message to your organization's IT Security / SOC team (e.g., abuse@yourcompany.com or security@company.com)."
        )
        if channel == "sms":
            recommended_steps.append(
                "For suspicious SMS (smishing), forward the text to 7726 (SPAM) on major telecommunication carriers."
            )
    else:
        recommended_steps.append(
            "While flagged as Likely Legitimate, remain vigilant: verify unexpected payment or account alteration requests out-of-band."
        )
        recommended_steps.append(
            "Check that sender headers match standard internal communication channels."
        )
        recommended_steps.append(
            "If in doubt, contact the alleged sender using an established, independent directory or internal chat."
        )

    # 3. Model Transparency & Limitations
    model_name = prediction_result.get("model_name", "Detector")
    if "Rule" in model_name:
        limitations_text = (
            "Heuristic rules evaluate specific explicit patterns (deadlines, credential words, defanged links). "
            "Rules cannot understand broader conversational nuances and may not detect zero-day or subtly phrased lures."
        )
    elif "SVM" in model_name:
        limitations_text = (
            "Linear SVM separates messages using a maximum-margin linear boundary over TF-IDF n-grams with Platt calibration. "
            "It is sensitive to vocabulary overlap and cannot infer real-time external domain reputation."
        )
    else:
        limitations_text = (
            "Logistic Regression calculates probabilities from linear combinations of n-gram TF-IDF weights. "
            "Probability values reflect statistical frequency in training data rather than absolute guarantees of malice."
        )

    # 4. Mandatory Safety Disclaimer
    mandatory_disclaimer = (
        "DECISION SUPPORT DISCLAIMER: PhishShift is an educational triage tool designed for risk evaluation. "
        "Predictions are NOT a guarantee of safety or malice. Advanced spear-phishing attacks can closely mimic legitimate correspondence, "
        "and legitimate notifications may contain urgent operational language. Always verify sensitive actions through trusted, independent channels."
    )

    return {
        "summary_narrative": summary_narrative,
        "suspicious_cues": suspicious_cues if suspicious_cues else ["No explicit high-risk heuristic triggers matched."],
        "legitimate_cues": legitimate_cues if legitimate_cues else ["No specific countervailing legitimate markers matched."],
        "recommended_steps": recommended_steps,
        "model_limitations": limitations_text,
        "mandatory_disclaimer": mandatory_disclaimer,
    }
