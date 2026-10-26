"""
Data Loader and Schema Validation Module for PhishShift Message Risk Scanner.
Preserves raw data integrity and validates strict schema specifications.
"""

from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple, Any
import pandas as pd

REQUIRED_COLUMNS = [
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
]

ALLOWED_LABELS = {"phishing", "legitimate"}
ALLOWED_CHANNELS = {"email", "sms"}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates that a DataFrame adheres to the ML-07 schema specifications.
    
    Returns:
        (is_valid, error_messages, summary_stats)
    """
    errors: List[str] = []
    
    # 1. Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors, {}

    # 2. Check for null or empty body_text
    empty_bodies = df["body_text"].isna().sum() + (df["body_text"].astype(str).str.strip() == "").sum()
    if empty_bodies > 0:
        errors.append(f"Found {empty_bodies} rows with empty or null body_text.")

    # 3. Check labels
    invalid_labels = set(df["label"].dropna().unique()) - ALLOWED_LABELS
    if invalid_labels:
        errors.append(f"Invalid labels found: {invalid_labels}. Allowed: {ALLOWED_LABELS}")

    # 4. Check channels
    invalid_channels = set(df["channel"].dropna().str.lower().unique()) - ALLOWED_CHANNELS
    if invalid_channels:
        errors.append(f"Invalid channels found: {invalid_channels}. Allowed: {ALLOWED_CHANNELS}")

    # 5. Check campaign_group
    null_campaigns = df["campaign_group"].isna().sum()
    if null_campaigns > 0:
        errors.append(f"Found {null_campaigns} rows with null campaign_group.")

    # 6. Check unique message_id
    duplicate_ids = df["message_id"].duplicated().sum()
    if duplicate_ids > 0:
        errors.append(f"Found {duplicate_ids} duplicate message_id entries.")

    # Compute summary statistics
    summary_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values_per_col": df[REQUIRED_COLUMNS].isna().sum().to_dict(),
        "label_distribution": df["label"].value_counts().to_dict(),
        "channel_distribution": df["channel"].value_counts().to_dict(),
        "campaign_count": df["campaign_group"].nunique(),
        "campaigns": df["campaign_group"].value_counts().to_dict(),
        "date_range": (
            str(df["collected_date"].min()) if "collected_date" in df and not df["collected_date"].empty else "N/A",
            str(df["collected_date"].max()) if "collected_date" in df and not df["collected_date"].empty else "N/A"
        )
    }

    is_valid = len(errors) == 0
    return is_valid, errors, summary_stats


def load_dataset(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads a CSV dataset, validates schema, and returns a sanitized DataFrame copy.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Try standard encodings
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not decode {filepath} with any supported encoding.")

    # Standardize string fields
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["channel"] = df["channel"].astype(str).str.strip().str.lower()
    df["subject_or_title"] = df["subject_or_title"].fillna("").astype(str)
    df["body_text"] = df["body_text"].fillna("").astype(str)
    df["sanitized_url_features"] = df["sanitized_url_features"].fillna("none").astype(str)
    df["sender_domain_type"] = df["sender_domain_type"].fillna("unknown").astype(str)
    df["campaign_group"] = df["campaign_group"].fillna("unknown_campaign").astype(str)
    df["collected_date"] = df["collected_date"].fillna("2026-01-01").astype(str)

    is_valid, errors, stats = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    return df, stats
