"""
Batch CSV Processing and Spam / Phishing File Scanner Module.
Supports arbitrary external CSV datasets (e.g. SMS Spam Collection, Enron, custom logs),
auto-detects text and label columns, executes high-throughput inference across models,
and computes evaluation metrics when ground-truth labels are present.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.predict import PhishShiftPredictor


# Common column name heuristics in public spam & phishing datasets
TEXT_COLUMN_CANDIDATES = [
    "body_text", "message", "text", "v2", "sms", "email_text", "content", "body", "mail", "msg", "sms_text"
]
LABEL_COLUMN_CANDIDATES = [
    "label", "v1", "category", "target", "class", "type", "is_spam", "spam", "classification"
]
SUBJECT_COLUMN_CANDIDATES = [
    "subject_or_title", "subject", "title", "header", "subject_text"
]


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Intelligently identifies text, label, and subject columns from a DataFrame.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    
    # 1. Text column detection
    text_col = None
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in cols_lower:
            text_col = cols_lower[cand]
            break
            
    # Fallback for text: select string column with longest average length
    if not text_col:
        string_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype == "string"]
        if string_cols:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in string_cols}
            text_col = max(avg_lens, key=avg_lens.get)

    # 2. Label column detection
    label_col = None
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    # 3. Subject column detection
    subject_col = None
    for cand in SUBJECT_COLUMN_CANDIDATES:
        if cand in cols_lower:
            subject_col = cols_lower[cand]
            break

    return text_col, label_col, subject_col


def standardize_label_values(series: pd.Series) -> pd.Series:
    """
    Normalizes diverse label representations (e.g. ham/spam, 0/1, True/False)
    into standard 'phishing' and 'legitimate' strings.
    """
    def map_val(val):
        s = str(val).strip().lower()
        if s in ["spam", "phishing", "phish", "1", "true", "malicious", "fraud"]:
            return "phishing"
        elif s in ["ham", "legitimate", "legit", "0", "false", "benign", "safe", "normal"]:
            return "legitimate"
        return "unknown"

    return series.apply(map_val)


def run_csv_batch_scan(
    df: pd.DataFrame,
    text_col: str,
    predictor: PhishShiftPredictor,
    model_name: str = "logistic_regression",
    subject_col: Optional[str] = None,
    label_col: Optional[str] = None,
    channel: str = "email",
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scans an uploaded CSV dataframe row-by-row, returning an enriched DataFrame
    with risk scores, predicted categories, detected cues, and aggregate metrics.
    """
    df_to_scan = df.copy()
    if max_rows and len(df_to_scan) > max_rows:
        df_to_scan = df_to_scan.iloc[:max_rows].copy()

    results_list = []
    
    for idx, row in df_to_scan.iterrows():
        body = str(row.get(text_col, "")).strip()
        subject = str(row.get(subject_col, "")) if subject_col and subject_col in row else ""
        
        # If body is completely empty, use placeholder
        if not body:
            body = "[Empty Message Body]"

        try:
            pred_res = predictor.predict_message(
                body_text=body,
                subject=subject,
                channel=channel,
                model_name=model_name
            )
            cues_str = "; ".join(pred_res["suspicious_cues"][:3]) if pred_res["suspicious_cues"] else "None"
            legit_str = "; ".join(pred_res["legitimate_cues"][:3]) if pred_res["legitimate_cues"] else "None"

            results_list.append({
                "predicted_category": pred_res["category"],
                "risk_score": pred_res["phishing_probability"],
                "risk_score_percent": pred_res["risk_score_percent"],
                "confidence_level": pred_res["confidence_level"],
                "is_uncertain": pred_res["is_uncertain"],
                "suspicious_cues": cues_str,
                "legitimate_signals": legit_str,
            })
        except Exception as e:
            results_list.append({
                "predicted_category": "Error",
                "risk_score": 0.5,
                "risk_score_percent": 50.0,
                "confidence_level": "Error",
                "is_uncertain": True,
                "suspicious_cues": f"Inference error: {str(e)}",
                "legitimate_signals": "None",
            })

    results_df = pd.DataFrame(results_list)
    enriched_df = pd.concat([df_to_scan.reset_index(drop=True), results_df], axis=1)

    # Compute summary statistics
    cat_counts = enriched_df["predicted_category"].value_counts().to_dict()
    mean_risk = float(enriched_df["risk_score"].mean()) if "risk_score" in enriched_df else 0.0

    summary_stats: Dict[str, Any] = {
        "total_scanned": len(enriched_df),
        "potential_phishing_count": cat_counts.get("Potential Phishing", 0),
        "likely_legitimate_count": cat_counts.get("Likely Legitimate", 0),
        "uncertain_review_count": cat_counts.get("Uncertain — Manual Review Recommended", 0),
        "mean_risk_score": round(mean_risk, 4),
        "mean_risk_percent": round(mean_risk * 100, 1),
        "model_used": model_name,
        "text_column_used": text_col,
    }

    # If ground-truth label column exists, calculate evaluation metrics
    if label_col and label_col in df_to_scan.columns:
        norm_labels = standardize_label_values(df_to_scan[label_col])
        enriched_df["ground_truth_standardized"] = norm_labels
        
        valid_mask = norm_labels.isin(["phishing", "legitimate"])
        if valid_mask.sum() > 0:
            y_true_binary = [1 if l == "phishing" else 0 for l in norm_labels[valid_mask]]
            # For evaluation, map Potential Phishing -> 1, Likely Legitimate -> 0, Uncertain treated as 1 or separate
            y_pred_binary = [1 if c == "Potential Phishing" else 0 for c in enriched_df.loc[valid_mask, "predicted_category"]]

            cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            summary_stats["ground_truth_metrics"] = {
                "evaluated_rows": int(valid_mask.sum()),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }

    return enriched_df, summary_stats
