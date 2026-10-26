"""
Deduplication and Template Leakage Prevention Module (FR-01).
Detects exact duplicates and near-identical message templates using
token masking and Jaccard similarity across normalized shingles.
"""

from __future__ import annotations
import hashlib
import re
from typing import Dict, List, Set, Tuple, Any
import pandas as pd


# Regex for abstracting dynamic variable tokens in templates
DYNAMIC_TOKEN_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "<EMAIL>"),
    (re.compile(r"https?://\S+|hxxps?://\S+|www\.\S+|bit\[\.\]ly/\S+", re.IGNORECASE), "<URL>"),
    (re.compile(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\w+ \d{1,2},? \d{4})\b"), "<DATE>"),
    (re.compile(r"\b(?:[A-Z]{2,}-\d{3,}|\b\d{5,}\b)"), "<ID>"),
    (re.compile(r"[\$€£₹]\s*\d+(?:,\d{3})*(?:\.\d+)?"), "<AMOUNT>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
    (re.compile(r"\b(?:Dear|Hi|Hello)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", re.IGNORECASE), "<GREETING>"),
]


def extract_template_signature(text: str) -> str:
    """
    Transforms a message into an abstracted template signature by replacing
    dynamic variables (names, numbers, dates, ticket IDs, amounts, URLs).
    """
    if not text or not isinstance(text, str):
        return ""
    
    masked = text
    for pattern, replacement in DYNAMIC_TOKEN_PATTERNS:
        masked = pattern.sub(replacement, masked)
        
    # Lowercase and standardize whitespace
    masked = re.sub(r"\s+", " ", masked).strip().lower()
    return masked


def compute_shingle_set(text: str, k: int = 3) -> Set[str]:
    """Generates word k-shingles (k-grams) for similarity comparison."""
    words = text.split()
    if len(words) < k:
        return set(words)
    return set(" ".join(words[i:i+k]) for i in range(len(words) - k + 1))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two shingle sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def deduplicate_dataset(
    df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    keep_strategy: str = "first",
    text_col: str | None = None,
    label_col: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Implements template deduplication: Removes exact duplicates and identifies near-identical templates.
    
    Parameters:
        df: Input DataFrame with message fields.
        similarity_threshold: Jaccard similarity threshold for near-duplicate clustering.
        keep_strategy: "first" keeps the first instance of each near-duplicate template cluster.
        text_col: Optional name of the text/body column. Auto-detected if None.
        label_col: Optional name of the label column. Auto-detected if None.
        
    Returns:
        (deduplicated_df, deduplication_report)
    """
    raw_count = len(df)
    if raw_count == 0:
        return df.copy(), {
            "raw_count": 0,
            "exact_duplicates_count": 0,
            "near_duplicate_instances_removed": 0,
            "retained_count": 0,
            "template_clusters_count": 0,
            "similarity_threshold_used": similarity_threshold,
            "removal_ratio": 0.0,
            "clusters": [],
            "duplicate_clusters": [],
        }

    # Resolve text column
    if text_col is None or text_col not in df.columns:
        for candidate in ["body_text", "message", "text", "sms", "content", "body"]:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            # Fallback to first string/object column or first column
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            text_col = str_cols[0] if str_cols else df.columns[0]

    # Resolve label column
    has_label = False
    if label_col is not None and label_col in df.columns:
        has_label = True
    elif "label" in df.columns:
        label_col = "label"
        has_label = True
    elif "target" in df.columns:
        label_col = "target"
        has_label = True

    # 1. Step 1: Exact Duplication detection (Verbatim exact text)
    df_working = df.copy().reset_index(drop=True)
    df_working["_exact_hash"] = df_working[text_col].apply(
        lambda t: hashlib.sha256(str(t).strip().encode("utf-8")).hexdigest()
    )
    exact_duplicates_count = df_working.duplicated(subset=["_exact_hash"]).sum()

    # 2. Step 2: Near-duplicate template abstraction
    subj_col = "subject_or_title" if "subject_or_title" in df_working.columns else None
    df_working["_template_sig"] = df_working.apply(
        lambda r: extract_template_signature(f"{r.get(subj_col, '') if subj_col else ''} {r[text_col]}"),
        axis=1
    )
    df_working["_shingles"] = df_working["_template_sig"].apply(lambda s: compute_shingle_set(s, k=3))

    # Inverted index for fast candidate retrieval
    shingle_to_ids: Dict[str, List[int]] = {}
    for idx, sh_set in enumerate(df_working["_shingles"]):
        for sh in sh_set:
            if sh not in shingle_to_ids:
                shingle_to_ids[sh] = []
            shingle_to_ids[sh].append(idx)

    # Cluster near-duplicates
    n = len(df_working)
    visited = set()
    clusters: List[List[int]] = []
    
    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        shingles_i = df_working.at[i, "_shingles"]
        label_i = df_working.at[i, label_col] if has_label else None

        # Candidate selection via inverted index
        candidate_indices: Set[int] = set()
        for sh in shingles_i:
            for cand_id in shingle_to_ids.get(sh, []):
                if cand_id > i and cand_id not in visited:
                    candidate_indices.add(cand_id)

        for j in sorted(candidate_indices):
            if j in visited:
                continue
            # Compare within same label if label exists
            if not has_label or df_working.at[j, label_col] == label_i:
                sim = jaccard_similarity(shingles_i, df_working.at[j, "_shingles"])
                if sim >= similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
                    
        clusters.append(cluster)

    # Indices to retain based on keep_strategy
    retained_indices = [cluster[0] for cluster in clusters]
    near_duplicate_instances_count = raw_count - len(retained_indices)

    deduped_df = df.iloc[retained_indices].copy().reset_index(drop=True)

    # Clean up working columns from deduped_df if present
    for col in ["_exact_hash", "_template_sig", "_shingles"]:
        if col in deduped_df.columns:
            deduped_df.drop(columns=[col], inplace=True)

    report = {
        "raw_count": raw_count,
        "exact_duplicates_count": int(exact_duplicates_count),
        "near_duplicate_instances_removed": int(near_duplicate_instances_count),
        "retained_count": len(deduped_df),
        "template_clusters_count": len(clusters),
        "similarity_threshold_used": similarity_threshold,
        "removal_ratio": round((raw_count - len(deduped_df)) / raw_count, 4) if raw_count > 0 else 0.0,
        "clusters": clusters,
        "duplicate_clusters": [c for c in clusters if len(c) > 1],
    }

    return deduped_df, report
