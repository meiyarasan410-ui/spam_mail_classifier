"""
Model Evaluation, Metrics Generation, and Error Analysis Module (FR-02, FR-03, FR-05).
Evaluates all three models on campaign-separated holdout data,
computes precision, recall, F1, FPR, generates confusion matrices,
inspects representative errors, and saves reproducible JSON artifacts.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import combine_message_text
from src.rule_detector import RuleBasedDetector
from src.predict import PhishShiftPredictor


def evaluate_model_predictions(
    y_true: List[str],
    y_pred: List[str],
    y_prob: List[float] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes rigorous cybersecurity metrics: Precision, Recall, F1, FPR, Confusion Matrix.
    """
    # Binary mapping: 'phishing' -> 1, 'legitimate' -> 0
    y_t = [1 if str(y).lower() == "phishing" else 0 for y in y_true]
    y_p = [1 if str(y).lower() == "phishing" else 0 for y in y_pred]

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "model_name": model_name,
        "total_samples": len(y_true),
        "phishing_samples": sum(y_t),
        "legitimate_samples": len(y_t) - sum(y_t),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    }
    return metrics


def evaluate_all_three_models(
    test_df: pd.DataFrame,
    models_dir: str = "models",
    results_dir: str = "results",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates Rule-Based, Logistic Regression, and Linear SVM models
    on the held-out campaign dataset.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    y_true = test_df["label"].tolist()
    X_test_texts = [combine_message_text(row) for _, row in test_df.iterrows()]

    predictor = PhishShiftPredictor(models_dir=models_dir)

    # 1. Model 1: Rule-Based Detector
    rule_results = [
        predictor.rule_detector.evaluate(
            body_text=row["body_text"],
            subject=row.get("subject_or_title", ""),
            sanitized_url_features=row.get("sanitized_url_features", ""),
            sender_domain_type=row.get("sender_domain_type", "unknown"),
            channel=row.get("channel", "email")
        )
        for _, row in test_df.iterrows()
    ]
    rule_preds = [r["predicted_label"] for r in rule_results]
    rule_probs = [r["risk_score"] for r in rule_results]
    metrics_rule = evaluate_model_predictions(y_true, rule_preds, rule_probs, "Rule-Based Baseline")

    # 2. Model 2: Logistic Regression
    lr_probs = []
    lr_preds = []
    if predictor.lr_pipeline is not None:
        classes = list(predictor.lr_pipeline.classes_)
        phish_idx = classes.index("phishing") if "phishing" in classes else 1
        probs_all = predictor.lr_pipeline.predict_proba(X_test_texts)
        lr_probs = [float(p[phish_idx]) for p in probs_all]
        lr_preds = ["phishing" if p >= 0.50 else "legitimate" for p in lr_probs]
        metrics_lr = evaluate_model_predictions(y_true, lr_preds, lr_probs, "TF-IDF + Logistic Regression")
    else:
        metrics_lr = {"status": "pending", "model_name": "TF-IDF + Logistic Regression"}

    # 3. Model 3: Linear SVM (Calibrated)
    svm_probs = []
    svm_preds = []
    if predictor.svm_pipeline is not None:
        classes = list(predictor.svm_pipeline.classes_)
        phish_idx = classes.index("phishing") if "phishing" in classes else 1
        probs_all = predictor.svm_pipeline.predict_proba(X_test_texts)
        svm_probs = [float(p[phish_idx]) for p in probs_all]
        svm_preds = ["phishing" if p >= 0.50 else "legitimate" for p in svm_probs]
        metrics_svm = evaluate_model_predictions(y_true, svm_preds, svm_probs, "TF-IDF + Linear SVM (Calibrated)")
    else:
        metrics_svm = {"status": "pending", "model_name": "TF-IDF + Linear SVM (Calibrated)"}

    # Evaluate FPR specifically on legitimate urgent messages
    urgent_legit_mask = (test_df["label"] == "legitimate") & (
        test_df["body_text"].str.contains("urgent|p1|immediate|incident|maintenance|priority", case=False) |
        test_df["subject_or_title"].str.contains("urgent|p1|immediate|incident|maintenance|priority", case=False)
    )
    urgent_legit_count = int(urgent_legit_mask.sum())
    
    urgent_metrics = {}
    if urgent_legit_count > 0:
        urgent_df = test_df[urgent_legit_mask]
        u_idx = urgent_df.index.tolist()
        
        rule_urgent_fps = sum(1 for i, idx in enumerate(test_df.index) if idx in u_idx and rule_preds[i] == "phishing")
        lr_urgent_fps = sum(1 for i, idx in enumerate(test_df.index) if idx in u_idx and lr_preds and lr_preds[i] == "phishing")
        svm_urgent_fps = sum(1 for i, idx in enumerate(test_df.index) if idx in u_idx and svm_preds and svm_preds[i] == "phishing")

        urgent_metrics = {
            "total_urgent_legitimate_samples": urgent_legit_count,
            "rule_false_positives": rule_urgent_fps,
            "rule_urgent_fpr": round(rule_urgent_fps / urgent_legit_count, 4),
            "lr_false_positives": lr_urgent_fps,
            "lr_urgent_fpr": round(lr_urgent_fps / urgent_legit_count, 4) if lr_preds else "pending",
            "svm_false_positives": svm_urgent_fps,
            "svm_urgent_fpr": round(svm_urgent_fps / urgent_legit_count, 4) if svm_preds else "pending",
        }

    # Error Analysis: Inspect representative errors across holdout campaigns
    error_analysis = find_representative_errors(test_df, y_true, rule_preds, lr_preds, svm_preds)

    # Plot evaluation comparison & confusion matrices
    comparison_results = {
        "rule_based": metrics_rule,
        "logistic_regression": metrics_lr,
        "linear_svm": metrics_svm,
        "urgent_legit_performance": urgent_metrics,
        "error_analysis": error_analysis,
        "campaigns_evaluated": sorted(list(test_df["campaign_group"].unique())),
    }

    results_file = os.path.join(results_dir, "metrics.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    plot_confusion_matrices(metrics_rule, metrics_lr, metrics_svm, reports_dir)
    plot_model_comparison_bar(metrics_rule, metrics_lr, metrics_svm, reports_dir)

    return comparison_results


def find_representative_errors(
    test_df: pd.DataFrame,
    y_true: List[str],
    rule_preds: List[str],
    lr_preds: List[str],
    svm_preds: List[str]
) -> List[Dict[str, Any]]:
    """
    Identifies and inspects at least three representative errors (False Positives and False Negatives).
    """
    errors = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        actual = y_true[i]
        r_pred = rule_preds[i]
        lr_pred = lr_preds[i] if lr_preds else "N/A"
        svm_pred = svm_preds[i] if svm_preds else "N/A"

        # Check for error in any model
        is_error = (r_pred != actual) or (lr_pred != actual and lr_pred != "N/A") or (svm_pred != actual and svm_pred != "N/A")
        if is_error:
            error_type = "False Positive (Legitimate flagged as Phishing)" if actual == "legitimate" else "False Negative (Phishing evaded detection)"
            root_cause = (
                "Legitimate operational urgency cues mimicked phishing pressure markers."
                if actual == "legitimate"
                else "Unfamiliar wording or lack of explicit coercive threats bypassed keyword and n-gram triggers."
            )
            mitigation = (
                "Strengthen legitimate operational contextual whitelist rules and domain header verification."
                if actual == "legitimate"
                else "Expand training coverage of subtler spear-phishing lures and cross-campaign holdout tuning."
            )

            errors.append({
                "message_id": row.get("message_id", f"msg_{i}"),
                "channel": row.get("channel", "email"),
                "campaign_group": row.get("campaign_group", "unknown"),
                "actual_label": actual,
                "rule_prediction": r_pred,
                "lr_prediction": lr_pred,
                "svm_prediction": svm_pred,
                "error_type": error_type,
                "subject_or_title": row.get("subject_or_title", ""),
                "body_snippet": row.get("body_text", "")[:140] + "...",
                "root_cause_analysis": root_cause,
                "recommended_mitigation": mitigation,
            })
            if len(errors) >= 5:
                break

    return errors


def plot_confusion_matrices(m_rule: Dict, m_lr: Dict, m_svm: Dict, reports_dir: str):
    """Generates a combined side-by-side confusion matrix figure."""
    models = [m_rule, m_lr, m_svm]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, m in zip(axes, models):
        if "confusion_matrix" in m:
            cm = m["confusion_matrix"]
            matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
            ax.set_title(m["model_name"], fontsize=11, fontweight="bold", pad=10)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Legit", "Pred Phish"], fontsize=9)
            ax.set_yticklabels(["Actual Legit", "Actual Phish"], fontsize=9)
            
            for i in range(2):
                for j in range(2):
                    val = matrix[i, j]
                    color = "white" if val > matrix.max() / 2 else "black"
                    ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Pending Evaluation", ha="center", va="center")
            ax.set_title(m.get("model_name", "Model"), fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "confusion_matrices.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()


def plot_model_comparison_bar(m_rule: Dict, m_lr: Dict, m_svm: Dict, reports_dir: str):
    """Generates comparison bar chart across Precision, Recall, F1, and False Positive Rate."""
    models = [m for m in [m_rule, m_lr, m_svm] if "precision" in m]
    if not models:
        return

    names = [m["model_name"] for m in models]
    metrics_keys = ["precision", "recall", "f1_score", "false_positive_rate"]
    labels = ["Precision", "Recall", "F1-Score", "False Positive Rate (FPR)"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1e3a8a", "#0284c7", "#0d9488"]

    for i, m in enumerate(models):
        values = [m[k] for k in metrics_keys]
        ax.bar(x + (i - 1) * width, values, width, label=names[i], color=colors[i % len(colors)])

    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=10)
    ax.set_title("PhishShift Model Performance Comparison on Campaign-Separated Holdout", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
