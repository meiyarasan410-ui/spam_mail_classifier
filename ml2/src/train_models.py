"""
Model Training and Serialization Pipeline (Models 2 and 3).
Trains TF-IDF + Logistic Regression and TF-IDF + Calibrated Linear SVM
strictly on campaign-separated training data to guarantee zero leakage.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.preprocessing import normalize_text_for_tfidf, combine_message_text


def tfidf_preprocessor(text: str) -> str:
    """Top-level preprocessor function for TF-IDF pickling compatibility."""
    return normalize_text_for_tfidf(text, preserve_signals=True)


def create_vectorizer() -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer that preserves custom cybersecurity tokens."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b|__\w+__",
        preprocessor=tfidf_preprocessor,
    )


def train_logistic_regression(
    X_train_texts: list[str],
    y_train: list[str],
    c_param: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Model 2: TF-IDF + Logistic Regression.
    Returns trained Scikit-learn Pipeline.
    """
    vectorizer = create_vectorizer()
    classifier = LogisticRegression(
        C=c_param,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced"
    )
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])
    pipeline.fit(X_train_texts, y_train)
    return pipeline


def train_linear_svm(
    X_train_texts: list[str],
    y_train: list[str],
    c_param: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Model 3: TF-IDF + Linear SVM.
    Uses CalibratedClassifierCV with Platt scaling (sigmoid calibration)
    so outputs represent genuine calibrated confidence rather than uncalibrated raw margins.
    """
    vectorizer = create_vectorizer()
    base_svc = LinearSVC(
        C=c_param,
        dual="auto",
        max_iter=3000,
        random_state=random_state,
        class_weight="balanced"
    )
    # CalibratedClassifierCV fits sigmoid calibration on internal cross-validation folds
    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=3
    )
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", calibrated_svc)
    ])
    pipeline.fit(X_train_texts, y_train)
    return pipeline


def train_and_save_all_models(
    train_df: pd.DataFrame,
    models_dir: str = "models",
    metadata_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Trains both ML models on training partition and persists them to disk with provenance metadata.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Prepare training text and target labels
    X_train = [combine_message_text(row) for _, row in train_df.iterrows()]
    y_train = train_df["label"].tolist()

    # 1. Train Model 2: Logistic Regression
    lr_pipeline = train_logistic_regression(X_train, y_train)
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    joblib.dump(lr_pipeline, lr_path)

    # 2. Train Model 3: Calibrated Linear SVM
    svm_pipeline = train_linear_svm(X_train, y_train)
    svm_path = os.path.join(models_dir, "linear_svm.joblib")
    joblib.dump(svm_pipeline, svm_path)

    # Extract vocabulary size and feature metrics
    vocab_size = len(lr_pipeline.named_steps["tfidf"].vocabulary_)

    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "total_train_samples": len(train_df),
        "train_campaigns": sorted(list(train_df["campaign_group"].unique())),
        "vocabulary_size": vocab_size,
        "models": {
            "logistic_regression": {
                "file": "logistic_regression.joblib",
                "type": "TfidfVectorizer + LogisticRegression",
                "hyperparameters": {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"},
                "limitations": "Linear probability estimation based on token frequency; susceptible to out-of-vocabulary evasion and synonym substitution.",
            },
            "linear_svm": {
                "file": "linear_svm.joblib",
                "type": "TfidfVectorizer + CalibratedClassifierCV(LinearSVC)",
                "hyperparameters": {"C": 1.0, "calibration": "sigmoid (Platt scaling)", "cv": 3},
                "limitations": "Maximum-margin hyperplane with Platt sigmoid calibration; robust against sparse noise, but does not capture non-linear sentence semantics.",
            }
        },
        "experiment_settings": metadata_info or {}
    }

    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "lr_pipeline": lr_pipeline,
        "svm_pipeline": svm_pipeline,
        "lr_path": lr_path,
        "svm_path": svm_path,
        "metadata_path": meta_path,
        "vocab_size": vocab_size
    }
