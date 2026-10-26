"""
Inference and Abstention Engine for PhishShift (FR-04).
Implements configurable reject/abstain thresholds for high-uncertainty samples,
ensuring users receive actionable decision-support triage without false certainty.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional
import joblib
import numpy as np

from src.preprocessing import combine_message_text, extract_cybersecurity_features
from src.rule_detector import RuleBasedDetector


UNCERTAINTY_LOWER = 0.40
UNCERTAINTY_UPPER = 0.65


class PhishShiftPredictor:
    """
    Unified predictor supporting:
      1. Rule-Based Baseline
      2. TF-IDF + Logistic Regression
      3. TF-IDF + Calibrated Linear SVM
    """

    def __init__(
        self,
        models_dir: str = "models",
        uncertainty_lower: float = UNCERTAINTY_LOWER,
        uncertainty_upper: float = UNCERTAINTY_UPPER
    ):
        self.models_dir = models_dir
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        self.rule_detector = RuleBasedDetector()
        
        self.lr_pipeline = None
        self.svm_pipeline = None
        self._load_available_models()

    def _load_available_models(self):
        lr_path = os.path.join(self.models_dir, "logistic_regression.joblib")
        if os.path.exists(lr_path):
            try:
                self.lr_pipeline = joblib.load(lr_path)
            except Exception:
                self.lr_pipeline = None

        svm_path = os.path.join(self.models_dir, "linear_svm.joblib")
        if os.path.exists(svm_path):
            try:
                self.svm_pipeline = joblib.load(svm_path)
            except Exception:
                self.svm_pipeline = None

    def predict_message(
        self,
        body_text: str,
        subject: str = "",
        sanitized_url_features: str = "",
        sender_domain_type: str = "unknown",
        channel: str = "email",
        model_name: str = "logistic_regression"
    ) -> Dict[str, Any]:
        """
        Executes prediction with abstention and transparent risk scoring.
        """
        if not body_text or not body_text.strip():
            raise ValueError("Message body cannot be empty.")

        message_dict = {
            "body_text": body_text,
            "subject_or_title": subject,
            "sanitized_url_features": sanitized_url_features,
            "sender_domain_type": sender_domain_type,
            "channel": channel,
        }
        combined_text = combine_message_text(message_dict)

        # 1. Rule-based detector
        rule_eval = self.rule_detector.evaluate(
            body_text=body_text,
            subject=subject,
            sanitized_url_features=sanitized_url_features,
            sender_domain_type=sender_domain_type,
            channel=channel
        )

        model_key = model_name.lower().replace(" ", "_").replace("-", "_")

        if "rule" in model_key:
            phishing_prob = rule_eval["risk_score"]
            raw_score = rule_eval["risk_score"]
            model_display_name = "Rule-Based Baseline Detector"
            score_type_explanation = "Heuristic risk score accumulated from suspicious keyword patterns, domain indicators, and URL structure."
        elif "svm" in model_key:
            if self.svm_pipeline is None:
                self._load_available_models()
            if self.svm_pipeline is None:
                raise RuntimeError("Linear SVM model has not been trained yet. Please train models first.")
            
            # Predict calibrated probability using CalibratedClassifierCV
            classes = list(self.svm_pipeline.classes_)
            phish_idx = classes.index("phishing") if "phishing" in classes else 1
            probs = self.svm_pipeline.predict_proba([combined_text])[0]
            ml_prob = float(probs[phish_idx])
            
            # Check vocabulary overlap to mitigate out-of-vocabulary linear intercept distortion
            tfidf_vec = self.svm_pipeline.named_steps["tfidf"].transform([combined_text])
            if tfidf_vec.nnz == 0:
                phishing_prob = float(rule_eval["risk_score"])
            else:
                phishing_prob = ml_prob
                # Align if rule detector identifies strong explicit threat or clean benign context
                if rule_eval["risk_score"] >= 0.70 and phishing_prob < 0.45:
                    phishing_prob = max(phishing_prob, (phishing_prob + rule_eval["risk_score"]) / 2)
                elif rule_eval["risk_score"] <= 0.15 and len(rule_eval["suspicious_cues"]) == 0 and phishing_prob > 0.55:
                    phishing_prob = min(phishing_prob, (phishing_prob + rule_eval["risk_score"]) / 2)

            raw_score = phishing_prob
            model_display_name = "TF-IDF + Calibrated Linear SVM"
            score_type_explanation = "Platt-calibrated probability score computed via internal 3-fold cross-validation over the linear maximum-margin hyperplane."
        else:
            # Default: Logistic Regression
            if self.lr_pipeline is None:
                self._load_available_models()
            if self.lr_pipeline is None:
                raise RuntimeError("Logistic Regression model has not been trained yet. Please train models first.")
            
            classes = list(self.lr_pipeline.classes_)
            phish_idx = classes.index("phishing") if "phishing" in classes else 1
            probs = self.lr_pipeline.predict_proba([combined_text])[0]
            ml_prob = float(probs[phish_idx])

            # Check vocabulary overlap to mitigate out-of-vocabulary linear intercept distortion
            tfidf_vec = self.lr_pipeline.named_steps["tfidf"].transform([combined_text])
            if tfidf_vec.nnz == 0:
                phishing_prob = float(rule_eval["risk_score"])
            else:
                phishing_prob = ml_prob
                if rule_eval["risk_score"] >= 0.70 and phishing_prob < 0.45:
                    phishing_prob = max(phishing_prob, (phishing_prob + rule_eval["risk_score"]) / 2)
                elif rule_eval["risk_score"] <= 0.15 and len(rule_eval["suspicious_cues"]) == 0 and phishing_prob > 0.55:
                    phishing_prob = min(phishing_prob, (phishing_prob + rule_eval["risk_score"]) / 2)

            raw_score = phishing_prob
            model_display_name = "TF-IDF + Logistic Regression"
            score_type_explanation = "Logistic regression sigmoid probability score derived from TF-IDF n-gram feature weights."

        # FR-04: Apply Reject/Abstain Policy for Uncertainty
        if self.uncertainty_lower <= phishing_prob <= self.uncertainty_upper:
            category = "Uncertain — Manual Review Recommended"
            is_uncertain = True
            category_color = "warning"
            confidence_level = "Low (Borderline Decision)"
        elif phishing_prob > self.uncertainty_upper:
            category = "Potential Phishing"
            is_uncertain = False
            category_color = "danger"
            confidence_level = "High" if phishing_prob > 0.85 else "Moderate"
        else:
            category = "Likely Legitimate"
            is_uncertain = False
            category_color = "success"
            confidence_level = "High" if phishing_prob < 0.20 else "Moderate"

        return {
            "model_name": model_display_name,
            "category": category,
            "is_uncertain": is_uncertain,
            "phishing_probability": round(phishing_prob, 4),
            "risk_score_percent": round(phishing_prob * 100, 1),
            "confidence_level": confidence_level,
            "category_color": category_color,
            "score_type_explanation": score_type_explanation,
            "suspicious_cues": rule_eval["suspicious_cues"],
            "legitimate_cues": rule_eval["legitimate_cues"],
            "extracted_features": rule_eval["extracted_features"],
            "channel": channel,
            "uncertainty_band": [self.uncertainty_lower, self.uncertainty_upper],
        }
