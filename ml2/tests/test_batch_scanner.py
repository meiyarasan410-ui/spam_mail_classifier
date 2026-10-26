"""
Unit tests for the Batch CSV Spam File Scanner module.
"""

from __future__ import annotations
import os
import pandas as pd
import pytest

from src.batch_scanner import detect_columns, standardize_label_values, run_csv_batch_scan
from src.predict import PhishShiftPredictor


def test_detect_columns_heuristics():
    # Test SMS Spam style: v1, v2
    df_sms = pd.DataFrame({"v1": ["ham", "spam"], "v2": ["Hello friend", "WINNER! Call now"]})
    text_col, label_col, subject_col = detect_columns(df_sms)
    assert text_col == "v2"
    assert label_col == "v1"

    # Test standard message, label
    df_std = pd.DataFrame({"label": ["spam"], "message": ["Your account is suspended"]})
    text_col2, label_col2, _ = detect_columns(df_std)
    assert text_col2 == "message"
    assert label_col2 == "label"


def test_standardize_label_values():
    s = pd.Series(["spam", "ham", "Spam", "Ham", "phishing", "legitimate", "1", "0", "random"])
    mapped = standardize_label_values(s)
    assert mapped[0] == "phishing"
    assert mapped[1] == "legitimate"
    assert mapped[2] == "phishing"
    assert mapped[3] == "legitimate"
    assert mapped[4] == "phishing"
    assert mapped[5] == "legitimate"
    assert mapped[6] == "phishing"
    assert mapped[7] == "legitimate"
    assert mapped[8] == "unknown"


def test_run_csv_batch_scan_with_labels():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    predictor = PhishShiftPredictor(models_dir=models_dir)

    sample_df = pd.DataFrame({
        "label": ["ham", "spam", "ham"],
        "message": [
            "Hey Alice, see you at the cafe at 4pm.",
            "FINAL WARNING: Your account is suspended within 24 hours. Verify at hxxps://auth[.]net now!",
            "Routine monthly salary advice is posted on the company portal."
        ]
    })

    enriched_df, stats = run_csv_batch_scan(
        df=sample_df,
        text_col="message",
        label_col="label",
        predictor=predictor,
        model_name="rule_based"
    )

    assert len(enriched_df) == 3
    assert "predicted_category" in enriched_df.columns
    assert "risk_score" in enriched_df.columns
    assert "suspicious_cues" in enriched_df.columns
    assert stats["total_scanned"] == 3
    assert "ground_truth_metrics" in stats
    assert stats["ground_truth_metrics"]["precision"] >= 0.0
