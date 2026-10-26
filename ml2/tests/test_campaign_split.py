"""
Unit tests for FR-03: Campaign-separated holdout splitting and leakage prevention audit.
"""

from __future__ import annotations
import pandas as pd
import pytest
from src.data_splitting import split_by_campaign_group, perform_leakage_audit


def test_split_by_campaign_group_ensures_zero_overlap():
    # Construct synthetic dataset with 5 distinct campaigns
    data = []
    for c_id in range(1, 6):
        camp = f"campaign_{c_id}"
        for m_id in range(1, 10):
            data.append({
                "message_id": f"MSG_{c_id}_{m_id}",
                "campaign_group": camp,
                "label": "phishing" if c_id % 2 == 0 else "legitimate",
                "body_text": f"Body for message {m_id} belonging strictly to {camp} with unique terms {c_id}.",
                "subject_or_title": f"Subject {c_id}"
            })
    df = pd.DataFrame(data)

    train_df, test_df, meta = split_by_campaign_group(df, test_size=0.4, random_state=42)

    # 1. Verify train and test are non-empty
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(df)

    # 2. FR-03 Critical check: Zero campaign overlap
    train_camps = set(train_df["campaign_group"].unique())
    test_camps = set(test_df["campaign_group"].unique())
    assert train_camps.isdisjoint(test_camps)

    # 3. Leakage audit verification
    audit = meta["leakage_audit"]
    assert audit["is_leakage_free"] is True
    assert audit["campaign_overlap_count"] == 0
    assert audit["verbatim_leakage_count"] == 0


def test_split_raises_error_on_insufficient_campaigns():
    data = [
        {"message_id": "1", "campaign_group": "c1", "label": "phishing", "body_text": "Sample 1"},
        {"message_id": "2", "campaign_group": "c2", "label": "legitimate", "body_text": "Sample 2"},
    ]
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Reliable campaign-separated evaluation requires at least 4"):
        split_by_campaign_group(df)


def test_perform_leakage_audit_detects_contamination():
    # Artificial contamination
    train_data = [
        {"message_id": "T1", "campaign_group": "camp_alpha", "label": "phishing", "body_text": "Shared identical phishing text lure."},
        {"message_id": "T2", "campaign_group": "camp_beta", "label": "legitimate", "body_text": "Legitimate notice."},
    ]
    test_data = [
        {"message_id": "E1", "campaign_group": "camp_alpha", "label": "phishing", "body_text": "Shared identical phishing text lure."},  # Campaign and text collision!
    ]
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    audit = perform_leakage_audit(train_df, test_df)
    assert audit["is_leakage_free"] is False
    assert audit["campaign_overlap_count"] == 1
    assert audit["verbatim_leakage_count"] == 1
