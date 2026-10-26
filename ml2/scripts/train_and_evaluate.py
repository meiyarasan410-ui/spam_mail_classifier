"""
End-to-End Training, Leakage Audit, and Evaluation Pipeline for PhishShift.
Executes deduplication, campaign splitting, model training, holdout evaluation,
and robustness perturbation analysis, producing persistent artifacts.
"""

from __future__ import annotations
import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset
from src.deduplication import deduplicate_dataset
from src.data_splitting import split_by_campaign_group
from src.train_models import train_and_save_all_models
from src.evaluation import evaluate_all_three_models
from src.predict import PhishShiftPredictor
from src.perturbations import run_robustness_perturbation_study


def run_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "synthetic_messages.csv")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    reports_dir = os.path.join(project_root, "reports")

    print("=" * 70)
    print("PHISHSHIFT MESSAGE RISK SCANNER: REPRODUCIBLE PIPELINE (ML-07)")
    print("=" * 70)

    # 1. Load raw dataset & validate schema
    print("\n[Step 1/6] Loading and validating raw dataset...")
    df_raw, raw_stats = load_dataset(data_path)
    print(f" -> Raw dataset loaded: {raw_stats['total_rows']} total rows across {raw_stats['campaign_count']} campaigns.")
    print(f" -> Label distribution: {raw_stats['label_distribution']}")

    # 2. FR-01: Deduplication (Exact + Near-Duplicate Templates)
    print("\n[Step 2/6] Executing FR-01 deduplication and template clustering...")
    df_deduped, dedup_report = deduplicate_dataset(df_raw, similarity_threshold=0.85)
    print(f" -> Exact duplicates found: {dedup_report['exact_duplicates_count']}")
    print(f" -> Near-duplicate instances removed: {dedup_report['near_duplicate_instances_removed']}")
    print(f" -> Retained clean messages: {dedup_report['retained_count']}")

    # 3. FR-03: Campaign-Separated Holdout Splitting & Leakage Audit
    print("\n[Step 3/6] Executing FR-03 campaign-separated holdout splitting...")
    # Explicitly designate diverse unseen campaigns for testing generalization:
    # 2 phishing campaigns (1 subtle unfamiliar, 1 sms smishing) + 2 legitimate campaigns (1 urgent incident, 1 pwd reset)
    holdout_campaigns = [
        "phish_unfamiliar_wording_cloud_v4",
        "phish_sms_bank_fraud_v6",
        "legit_urgent_incident",
        "legit_pwd_reset_routine",
    ]
    train_df, test_df, split_meta = split_by_campaign_group(
        df_deduped,
        test_size=0.25,
        random_state=42,
        holdout_campaigns=holdout_campaigns
    )
    print(f" -> Training partition: {len(train_df)} samples across {len(split_meta['train_campaigns'])} campaigns.")
    print(f" -> Holdout test partition: {len(test_df)} samples across {len(split_meta['test_campaigns'])} campaigns.")
    print(f" -> Held-out campaigns: {split_meta['test_campaigns']}")
    
    # Leakage check verdict
    audit = split_meta["leakage_audit"]
    print(f" -> Data Leakage Audit: {audit['audit_verdict']}")
    print(f"    - Campaign overlap: {audit['campaign_overlap_count']}")
    print(f"    - Verbatim body collisions: {audit['verbatim_leakage_count']}")
    print(f"    - Near-duplicate cross-partition links: {audit['near_duplicate_cross_partition_count']}")
    print(f"    - Label contradictions: {audit['label_contradiction_count']}")

    # 4. Train Models 2 and 3 on Training Partition ONLY
    print("\n[Step 4/6] Training Model 2 (Logistic Regression) & Model 3 (Calibrated Linear SVM)...")
    train_artifacts = train_and_save_all_models(
        train_df,
        models_dir=models_dir,
        metadata_info={
            "deduplication_report": dedup_report,
            "split_metadata": {
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "test_campaigns": split_meta["test_campaigns"],
            }
        }
    )
    print(f" -> Vocabulary extracted: {train_artifacts['vocab_size']} n-gram features.")
    print(f" -> Models serialized to: {train_artifacts['lr_path']} and {train_artifacts['svm_path']}")

    # 5. Evaluate all 3 Models on Holdout Campaigns
    print("\n[Step 5/6] Evaluating all 3 models on held-out campaigns (FR-02, FR-03)...")
    eval_results = evaluate_all_three_models(
        test_df,
        models_dir=models_dir,
        results_dir=results_dir,
        reports_dir=reports_dir
    )
    print("\n--- Model Comparison Summary ---")
    for key, name in [
        ("rule_based", "1. Rule-Based Baseline"),
        ("logistic_regression", "2. TF-IDF + Logistic Regression"),
        ("linear_svm", "3. TF-IDF + Calibrated Linear SVM"),
    ]:
        m = eval_results.get(key, {})
        print(f"[{name}]")
        print(f"  Precision: {m.get('precision', 'N/A')} | Recall: {m.get('recall', 'N/A')} | F1: {m.get('f1_score', 'N/A')} | FPR: {m.get('false_positive_rate', 'N/A')}")
        if "confusion_matrix" in m:
            cm = m["confusion_matrix"]
            print(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")

    urgent_perf = eval_results.get("urgent_legit_performance", {})
    if urgent_perf:
        print("\n--- Legitimate Urgent Messages Performance ---")
        print(f"  Total urgent legitimate messages tested: {urgent_perf.get('total_urgent_legitimate_samples')}")
        print(f"  Rule-Based False Positives: {urgent_perf.get('rule_false_positives')} (FPR: {urgent_perf.get('rule_urgent_fpr')})")
        print(f"  Logistic Regression False Positives: {urgent_perf.get('lr_false_positives')} (FPR: {urgent_perf.get('lr_urgent_fpr')})")
        print(f"  Linear SVM False Positives: {urgent_perf.get('svm_false_positives')} (FPR: {urgent_perf.get('svm_urgent_fpr')})")

    # 6. FR-05: Robustness Perturbation Study
    print("\n[Step 6/6] Executing FR-05 robustness and perturbation study (8 controlled tests)...")
    predictor = PhishShiftPredictor(models_dir=models_dir)
    pert_summary = run_robustness_perturbation_study(predictor, results_dir=results_dir)
    print(f" -> Completed {pert_summary['total_perturbation_tests']} controlled perturbation tests.")
    print(f" -> Saved perturbation study to {os.path.join(results_dir, 'perturbation_results.json')}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE: ALL ARTIFACTS SAVED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
