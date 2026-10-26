"""
Unit tests for FR-01: Deduplication and near-duplicate template abstraction.
"""

from __future__ import annotations
import pandas as pd
import pytest
from src.deduplication import (
    extract_template_signature,
    compute_shingle_set,
    jaccard_similarity,
    deduplicate_dataset,
)


def test_extract_template_signature_masks_variables():
    msg1 = "Hello Alice, your ticket INC-10294 with amount $450.00 is due on 2026-03-15. Visit https://portal.com/auth"
    msg2 = "Hello Bob, your ticket INC-99382 with amount $920.00 is due on 2026-04-20. Visit https://portal.com/auth"

    sig1 = extract_template_signature(msg1)
    sig2 = extract_template_signature(msg2)

    assert "<greeting>" in sig1
    assert "<id>" in sig1 or "<num>" in sig1
    assert "<amount>" in sig1
    assert "<date>" in sig1
    assert "<url>" in sig1
    # Both signatures should be nearly identical after variable masking
    assert sig1 == sig2


def test_jaccard_similarity_identical_and_disjoint():
    s1 = {"urgent", "account", "suspended"}
    s2 = {"urgent", "account", "suspended"}
    s3 = {"routine", "payroll", "statement"}

    assert jaccard_similarity(s1, s2) == 1.0
    assert jaccard_similarity(s1, s3) == 0.0


def test_deduplicate_dataset_removes_exact_and_near_duplicates():
    data = [
        # Cluster 1: Near duplicate phishing templates with slight variations
        {"message_id": "M1", "label": "phishing", "body_text": "Your account #1001 is suspended. Verify at hxxps://auth1[.]net now.", "subject_or_title": ""},
        {"message_id": "M2", "label": "phishing", "body_text": "Your account #2002 is suspended. Verify at hxxps://auth2[.]net now.", "subject_or_title": ""},
        {"message_id": "M3", "label": "phishing", "body_text": "Your account #3003 is suspended. Verify at hxxps://auth3[.]net now.", "subject_or_title": ""},
        # Cluster 2: Legitimate notices
        {"message_id": "M4", "label": "legitimate", "body_text": "Routine monthly pay stub is available in Workday.", "subject_or_title": ""},
        {"message_id": "M5", "label": "legitimate", "body_text": "Routine monthly pay stub is available in Workday.", "subject_or_title": ""},  # Exact duplicate
    ]
    df = pd.DataFrame(data)
    deduped_df, report = deduplicate_dataset(df, similarity_threshold=0.80)

    assert report["raw_count"] == 5
    assert report["exact_duplicates_count"] == 1
    assert report["retained_count"] < 5
    # Should retain 1 from Cluster 1 and 1 from Cluster 2 -> 2 unique templates
    assert len(deduped_df) == 2
