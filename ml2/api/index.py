"""
PhishShift Message Risk Scanner - Complete Multi-View Cybersecurity Dashboard.
Serves the full educational cybersecurity application for Vercel Serverless and local WSGI execution.
Includes Dashboard, Message Scanner, Batch CSV Scanner, Model Comparison, Dataset & Deduplication, and Robustness Lab.
"""

from __future__ import annotations
import os
import sys
import json
import io
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import PhishShiftPredictor, UNCERTAINTY_LOWER, UNCERTAINTY_UPPER
from src.explanations import generate_explanation_advisory
from src.rule_detector import RuleBasedDetector
from src.deduplication import deduplicate_dataset
from src.batch_scanner import detect_columns, run_csv_batch_scan
from src.data_loader import validate_schema

app = Flask(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

predictor = None
rule_detector = None

def get_engine():
    global predictor, rule_detector
    if predictor is None:
        predictor = PhishShiftPredictor(models_dir=MODELS_DIR)
        rule_detector = RuleBasedDetector()
    return predictor, rule_detector


# -----------------------------------------------------------------------------
# STATIC ARTIFACT SERVING
# -----------------------------------------------------------------------------
@app.route("/reports/<path:filename>", methods=["GET"])
def serve_report_file(filename):
    return send_from_directory(REPORTS_DIR, filename)


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "PhishShift Message Risk Scanner", "version": "2.0.0"})


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "metrics not found"}), 404


@app.route("/api/perturbations", methods=["GET"])
def get_perturbations():
    pert_path = os.path.join(RESULTS_DIR, "perturbation_results.json")
    if os.path.exists(pert_path):
        with open(pert_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "perturbations not found"}), 404


@app.route("/api/dataset-info", methods=["GET"])
def get_dataset_info():
    dataset_type = request.args.get("type", "synthetic")
    if dataset_type == "sms_spam":
        csv_path = os.path.join(DATA_DIR, "sample_spam_collection.csv")
        encoding = "latin1"
    else:
        csv_path = os.path.join(DATA_DIR, "synthetic_messages.csv")
        encoding = "utf-8"

    if not os.path.exists(csv_path):
        return jsonify({"error": f"Dataset file {csv_path} not found"}), 404

    df = pd.read_csv(csv_path, encoding=encoding)
    det_text, det_label, _ = detect_columns(df)
    
    col_summary = []
    for c in df.columns:
        null_count = int(df[c].isna().sum())
        col_summary.append({
            "name": c,
            "type": str(df[c].dtype),
            "missing": null_count,
            "fill_rate": round(((len(df) - null_count) / len(df)) * 100, 1),
        })

    label_counts = {}
    if det_label and det_label in df.columns:
        label_counts = {str(k): int(v) for k, v in df[det_label].value_counts().items()}

    campaign_counts = {}
    if "campaign_group" in df.columns:
        campaign_counts = {str(k): int(v) for k, v in df["campaign_group"].value_counts().head(10).items()}

    channel_counts = {}
    if "channel" in df.columns:
        channel_counts = {str(k): int(v) for k, v in df["channel"].value_counts().items()}

    # Preview first 10 rows
    preview = df.head(10).fillna("").to_dict(orient="records")

    return jsonify({
        "filename": os.path.basename(csv_path),
        "total_rows": len(df),
        "columns": list(df.columns),
        "text_column": det_text,
        "label_column": det_label,
        "label_counts": label_counts,
        "campaign_counts": campaign_counts,
        "channel_counts": channel_counts,
        "columns_summary": col_summary,
        "preview": preview,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_message():
    payload = request.get_json(force=True) or {}
    body_text = payload.get("body") or payload.get("text") or ""
    if not body_text.strip():
        return jsonify({"error": "Message body cannot be empty."}), 400

    subject = payload.get("subject", "")
    channel = payload.get("channel", "email")
    model_choice = payload.get("model", "svm")
    lower_cutoff = float(payload.get("lower_cutoff", UNCERTAINTY_LOWER))
    upper_cutoff = float(payload.get("upper_cutoff", UNCERTAINTY_UPPER))

    pred, rule_det = get_engine()

    # Apply custom threshold bounds if passed
    orig_lower, orig_upper = pred.uncertainty_lower, pred.uncertainty_upper
    pred.uncertainty_lower, pred.uncertainty_upper = lower_cutoff, upper_cutoff

    try:
        pred_res = pred.predict_message(
            body_text=body_text,
            subject=subject,
            channel=channel,
            model_name=model_choice,
        )
        advisory = generate_explanation_advisory(pred_res)
    finally:
        pred.uncertainty_lower, pred.uncertainty_upper = orig_lower, orig_upper

    return jsonify({
        "predicted_category": pred_res["category"],
        "risk_score_percent": pred_res["risk_score_percent"],
        "confidence_level": pred_res["confidence_level"],
        "model_name": pred_res["model_name"],
        "suspicious_cues": advisory["suspicious_cues"],
        "legitimate_cues": advisory["legitimate_cues"],
        "recommended_steps": advisory["recommended_steps"],
        "model_limitations": advisory["model_limitations"],
        "mandatory_disclaimer": advisory["mandatory_disclaimer"],
        "extracted_features": pred_res.get("extracted_features", {}),
    })


@app.route("/api/deduplicate", methods=["POST"])
def run_deduplication():
    payload = request.get_json(force=True) or {}
    dataset_type = payload.get("dataset_type", "synthetic")
    threshold = float(payload.get("threshold", 0.85))

    if dataset_type == "sms_spam":
        csv_path = os.path.join(DATA_DIR, "sample_spam_collection.csv")
        df = pd.read_csv(csv_path, encoding="latin1")
    else:
        csv_path = os.path.join(DATA_DIR, "synthetic_messages.csv")
        df = pd.read_csv(csv_path, encoding="utf-8")

    det_text, det_label, _ = detect_columns(df)
    deduped_df, report = deduplicate_dataset(
        df,
        similarity_threshold=threshold,
        text_col=det_text,
        label_col=det_label,
    )

    # Format sample duplicate clusters for UI inspection
    cluster_details = []
    for c_indices in report.get("duplicate_clusters", [])[:10]:
        rep_row = df.iloc[c_indices[0]]
        variants = [str(df.iloc[idx][det_text]) for idx in c_indices[1:5]]
        cluster_details.append({
            "size": len(c_indices),
            "retained_row_index": c_indices[0],
            "retained_text": str(rep_row[det_text]),
            "label": str(rep_row[det_label]) if det_label in rep_row else "unlabeled",
            "sample_variants": variants,
            "total_variants_count": len(c_indices) - 1,
        })

    return jsonify({
        "report": {
            "raw_count": report["raw_count"],
            "exact_duplicates_count": report["exact_duplicates_count"],
            "near_duplicate_instances_removed": report["near_duplicate_instances_removed"],
            "retained_count": report["retained_count"],
            "template_clusters_count": report["template_clusters_count"],
            "similarity_threshold_used": report["similarity_threshold_used"],
            "removal_ratio": report["removal_ratio"],
        },
        "duplicate_clusters": cluster_details,
    })


@app.route("/api/batch-scan", methods=["POST"])
def batch_scan():
    try:
        df = None
        text_col_override = None
        label_col_override = None
        model_choice = "svm"
        limit = 200

        # Check if multipart/form-data upload
        if request.files and "file" in request.files:
            file = request.files["file"]
            model_choice = request.form.get("model", "svm")
            limit = int(request.form.get("limit", 200))
            text_col_override = request.form.get("text_column") or None
            label_col_override = request.form.get("label_column") or None

            file_bytes = file.read()
            for enc in ["utf-8", "latin1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                    break
                except Exception:
                    continue
        else:
            payload = request.get_json(force=True, silent=True) or {}
            csv_data = payload.get("csv_data")
            dataset_type = payload.get("dataset_type", "custom")
            model_choice = payload.get("model", "svm")
            limit = int(payload.get("limit", 200))
            text_col_override = payload.get("text_column") or None
            label_col_override = payload.get("label_column") or None

            if csv_data and csv_data.strip():
                for enc in ["utf-8", "latin1", "cp1252"]:
                    try:
                        df = pd.read_csv(io.StringIO(csv_data), encoding=enc)
                        break
                    except Exception:
                        continue
            elif dataset_type == "sms_spam":
                csv_path = os.path.join(DATA_DIR, "sample_spam_collection.csv")
                df = pd.read_csv(csv_path, encoding="latin1")
            elif dataset_type == "synthetic":
                csv_path = os.path.join(DATA_DIR, "synthetic_messages.csv")
                df = pd.read_csv(csv_path, encoding="utf-8")

        if df is None or df.empty:
            return jsonify({"error": "CSV file is empty or could not be parsed."}), 400

        if limit and limit > 0 and limit < len(df):
            df = df.head(limit).copy()

        det_text, det_label, det_subject = detect_columns(df)
        if text_col_override and text_col_override in df.columns:
            det_text = text_col_override
        if label_col_override and label_col_override in df.columns:
            det_label = label_col_override

        if not det_text:
            det_text = df.columns[0]

        pred, _ = get_engine()

        results_df, summary = run_csv_batch_scan(
            df=df,
            text_col=det_text,
            predictor=pred,
            model_name=model_choice,
            label_col=det_label,
            subject_col=det_subject,
        )

        # Build preview records
        keep_cols = [det_text, "predicted_category", "risk_score_percent", "confidence_level", "suspicious_cues"]
        if det_label and det_label in results_df.columns:
            keep_cols.insert(1, det_label)

        safe_cols = [c for c in keep_cols if c in results_df.columns]
        records = results_df[safe_cols].fillna("").to_dict(orient="records")

        # Full enriched CSV export
        enriched_csv = results_df.to_csv(index=False)

        return jsonify({
            "summary": summary,
            "text_column": det_text,
            "label_column": det_label,
            "available_columns": list(df.columns),
            "records": records[:100],
            "total_scanned": len(results_df),
            "enriched_csv": enriched_csv,
        })
    except Exception as e:
        return jsonify({"error": f"Batch scan error: {str(e)}"}), 500


# -----------------------------------------------------------------------------
# MAIN FULL-APPLICATION HTML TEMPLATE
# -----------------------------------------------------------------------------
FULL_WEBSITE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhishShift | Message Risk Scanner</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome 6 Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet">
    <!-- Chart.js for Responsive Visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

    <!-- Inline Theme Bootstrap Script (Prevent FOUC) -->
    <script>
        (function() {
            var saved = localStorage.getItem('phishshift_theme');
            if (saved) {
                document.documentElement.setAttribute('data-theme', saved);
            } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
                document.documentElement.setAttribute('data-theme', 'light');
            } else {
                document.documentElement.setAttribute('data-theme', 'dark');
            }
        })();
    </script>

    <style>
        :root, [data-theme="dark"] {
            --bg-main: #0b1120;
            --sidebar-bg: #070d19;
            --card-bg: #111e38;
            --card-hover: #162444;
            --panel-bg: #0f172a;
            --snippet-bg: #020617;
            --snippet-text: #e2e8f0;
            --border-color: #1e3a8a;
            --accent-cyan: #38bdf8;
            --accent-blue: #0284c7;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --input-bg: #0f172a;
            --input-border: #1e3a8a;
            --input-text: #ffffff;
            --header-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            --header-text: #ffffff;
            --header-subtext: #94a3b8;
            --table-th-bg: #0f172a;
            --table-td-bg: #111e38;
            --table-border: #1e3a8a;
            --cue-bg: #0f172a;
            --disclaimer-bg: rgba(30, 58, 138, 0.25);
            --disclaimer-border: #1e40af;
            --disclaimer-text: #cbd5e1;
            --preset-btn-bg: #1e293b;
            --preset-btn-border: #334155;
            --preset-btn-text: #cbd5e1;
            --metric-card-bg: #111e38;
            --metric-value: #38bdf8;
            --accordion-bg: #111e38;
            --accordion-header-bg: #0f172a;
            --theme-toggle-bg: #172554;
            --theme-toggle-border: #1e3a8a;
            --theme-toggle-text: #f1f5f9;
            --badge-danger-bg: rgba(239, 68, 68, 0.15);
            --badge-danger-color: #fca5a5;
            --badge-danger-border: #ef4444;
            --badge-success-bg: rgba(16, 185, 129, 0.15);
            --badge-success-color: #6ee7b7;
            --badge-success-border: #10b981;
            --badge-warning-bg: rgba(245, 158, 11, 0.15);
            --badge-warning-color: #fde68a;
            --badge-warning-border: #f59e0b;
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }

        [data-theme="light"] {
            --bg-main: #f1f5f9;
            --sidebar-bg: #ffffff;
            --card-bg: #ffffff;
            --card-hover: #f8fafc;
            --panel-bg: #f8fafc;
            --snippet-bg: #f8fafc;
            --snippet-text: #0f172a;
            --border-color: #cbd5e1;
            --accent-cyan: #0284c7;
            --accent-blue: #0284c7;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --input-bg: #ffffff;
            --input-border: #cbd5e1;
            --input-text: #0f172a;
            --header-gradient: linear-gradient(135deg, #e0f2fe 0%, #e2e8f0 50%, #bae6fd 100%);
            --header-text: #0369a1;
            --header-subtext: #334155;
            --table-th-bg: #f1f5f9;
            --table-td-bg: #ffffff;
            --table-border: #e2e8f0;
            --cue-bg: #f8fafc;
            --disclaimer-bg: #eff6ff;
            --disclaimer-border: #bfdbfe;
            --disclaimer-text: #1e40af;
            --preset-btn-bg: #f1f5f9;
            --preset-btn-border: #cbd5e1;
            --preset-btn-text: #334155;
            --metric-card-bg: #ffffff;
            --metric-value: #0284c7;
            --accordion-bg: #ffffff;
            --accordion-header-bg: #f8fafc;
            --theme-toggle-bg: #f1f5f9;
            --theme-toggle-border: #cbd5e1;
            --theme-toggle-text: #0f172a;
            --badge-danger-bg: #fee2e2;
            --badge-danger-color: #b91c1c;
            --badge-danger-border: #f87171;
            --badge-success-bg: #d1fae5;
            --badge-success-color: #047857;
            --badge-success-border: #34d399;
            --badge-warning-bg: #fef3c7;
            --badge-warning-color: #b45309;
            --badge-warning-border: #fbbf24;
            --danger: #dc2626;
            --success: #059669;
            --warning: #d97706;
        }

        body {
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            min-height: 100vh;
            margin: 0;
            overflow-x: hidden;
            transition: background-color 0.25s ease, color 0.25s ease;
        }
        /* Dynamic Theme Overrides for Utility Classes */
        .border-navy { border-color: var(--border-color) !important; }
        .bg-dark { background-color: var(--panel-bg) !important; color: var(--text-primary) !important; }
        .bg-black { background-color: var(--snippet-bg) !important; color: var(--snippet-text) !important; }
        .text-light { color: var(--text-primary) !important; }
        .text-muted { color: var(--text-secondary) !important; }

        /* Layout Sidebar */
        .sidebar {
            width: 290px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            min-height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            padding: 24px 16px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            transition: background-color 0.25s ease, border-color 0.25s ease;
        }
        .main-content {
            margin-left: 290px;
            padding: 24px 36px;
            min-height: 100vh;
        }
        @media (max-width: 992px) {
            .sidebar {
                position: relative;
                width: 100%;
                min-height: auto;
            }
            .main-content {
                margin-left: 0;
                padding: 18px;
            }
        }
        .brand-logo {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
        }
        .brand-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }
        .nav-pills .nav-link {
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 6px;
            font-weight: 500;
            font-size: 0.92rem;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s ease;
        }
        .nav-pills .nav-link:hover {
            background-color: var(--card-hover);
            color: var(--accent-cyan);
        }
        .nav-pills .nav-link.active {
            background-color: var(--accent-blue);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
        }
        .sidebar-section-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
            margin-top: 20px;
            margin-bottom: 10px;
            padding-left: 8px;
            font-weight: 700;
        }

        /* Theme Toggle Components */
        .theme-toggle-btn {
            background-color: var(--theme-toggle-bg);
            border: 1px solid var(--theme-toggle-border);
            color: var(--theme-toggle-text);
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }
        .theme-toggle-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
        .theme-badge {
            background-color: var(--card-hover);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            font-size: 0.72rem;
        }
        .theme-top-btn {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 20px;
            padding: 6px 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        }
        .theme-top-btn:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-1px);
        }

        /* Top Header */
        .cyber-header {
            background: var(--header-gradient);
            border: 1px solid var(--border-color);
            padding: 26px 32px;
            border-radius: 12px;
            margin-bottom: 28px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        }
        .cyber-header h1 {
            color: var(--header-text);
        }
        .cyber-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            color: var(--text-primary);
        }
        .metric-card {
            background: var(--metric-card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--metric-value);
            margin-top: 4px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        /* Form Controls */
        .form-control, .form-select {
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            color: var(--input-text);
            border-radius: 8px;
        }
        .form-control:focus, .form-select:focus {
            background-color: var(--input-bg);
            border-color: var(--accent-cyan);
            color: var(--input-text);
            box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25);
        }
        .btn-preset {
            background-color: var(--preset-btn-bg);
            border: 1px solid var(--preset-btn-border);
            color: var(--preset-btn-text);
            font-size: 0.82rem;
            border-radius: 6px;
            padding: 6px 12px;
            transition: all 0.2s ease;
        }
        .btn-preset:hover {
            background-color: var(--accent-blue);
            border-color: var(--accent-cyan);
            color: #ffffff;
        }
        /* Badges */
        .badge-danger-custom {
            background-color: var(--badge-danger-bg);
            border: 1px solid var(--badge-danger-border);
            color: var(--badge-danger-color);
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .badge-success-custom {
            background-color: var(--badge-success-bg);
            border: 1px solid var(--badge-success-border);
            color: var(--badge-success-color);
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .badge-warning-custom {
            background-color: var(--badge-warning-bg);
            border: 1px solid var(--badge-warning-border);
            color: var(--badge-warning-color);
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .cue-box {
            background: var(--cue-bg);
            border-left: 4px solid var(--accent-cyan);
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
            font-size: 0.88rem;
            color: var(--text-primary);
        }
        .cue-danger { border-left-color: var(--danger); }
        .cue-success { border-left-color: var(--success); }
        .disclaimer-box {
            background-color: var(--disclaimer-bg);
            border: 1px solid var(--disclaimer-border);
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 0.8rem;
            color: var(--disclaimer-text);
            margin-top: 20px;
            line-height: 1.5;
        }
        /* Tables */
        .cyber-table {
            color: var(--text-primary);
            background-color: transparent;
            font-size: 0.88rem;
        }
        .cyber-table th {
            background-color: var(--table-th-bg);
            color: var(--accent-cyan);
            border-color: var(--table-border);
            font-weight: 600;
        }
        .cyber-table td {
            background-color: var(--table-td-bg);
            border-color: var(--table-border);
            color: var(--text-primary);
        }
        .accordion-item {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }
        .accordion-button {
            background-color: var(--panel-bg) !important;
            color: var(--text-primary) !important;
        }
        .accordion-button:not(.collapsed) {
            background-color: var(--card-hover) !important;
            color: var(--accent-cyan) !important;
        }
        .accordion-body {
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }
    </style>
</head>
<body>

    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="brand-logo">
            <i class="fa-solid fa-shield-halved text-info"></i> PhishShift
        </div>
        <div class="brand-subtitle">Message Risk Scanner</div>

        <ul class="nav nav-pills flex-column mb-auto" id="sidebarMenu">
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link active" data-view="dashboard" onclick="switchView('dashboard', this); return false;">
                    <i class="fa-solid fa-chart-pie me-2"></i> Dashboard <span class="badge bg-secondary ms-auto">1</span>
                </a>
            </li>
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link" data-view="scanner" onclick="switchView('scanner', this); return false;">
                    <i class="fa-solid fa-magnifying-glass me-2"></i> Message Scanner <span class="badge bg-secondary ms-auto">2</span>
                </a>
            </li>
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link" data-view="batch" onclick="switchView('batch', this); return false;">
                    <i class="fa-solid fa-file-csv me-2"></i> Batch CSV Scanner <span class="badge bg-secondary ms-auto">3</span>
                </a>
            </li>
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link" data-view="comparison" onclick="switchView('comparison', this); return false;">
                    <i class="fa-solid fa-scale-balanced me-2"></i> Model Comparison <span class="badge bg-secondary ms-auto">4</span>
                </a>
            </li>
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link" data-view="dataset" onclick="switchView('dataset', this); return false;">
                    <i class="fa-solid fa-folder-open me-2"></i> Dataset & Deduplication <span class="badge bg-secondary ms-auto">5</span>
                </a>
            </li>
            <li class="nav-item">
                <a href="javascript:void(0)" class="nav-link view-nav-link" data-view="robustness" onclick="switchView('robustness', this); return false;">
                    <i class="fa-solid fa-flask-vial me-2"></i> Robustness Lab <span class="badge bg-secondary ms-auto">6</span>
                </a>
            </li>
        </ul>

        <div class="sidebar-section-title">🎨 Theme Mode</div>
        <div class="px-2 mb-3">
            <button id="themeToggleBtn" class="btn theme-toggle-btn w-100 d-flex align-items-center justify-content-between py-2 px-3" onclick="toggleTheme()" title="Switch Light/Dark Mode">
                <span><i id="themeIcon" class="fa-solid fa-moon me-2 text-warning"></i> <span id="themeText">Dark Mode</span></span>
                <span class="badge rounded-pill theme-badge"><i class="fa-solid fa-circle-half-stroke"></i> Switch</span>
            </button>
        </div>

        <div class="sidebar-section-title">⚙️ Uncertainty Cutoffs</div>
        <div class="px-2">
            <label class="form-label small text-muted mb-0 d-flex justify-content-between">
                <span>Lower Cutoff:</span>
                <span id="lowerCutoffVal" class="text-info fw-bold">0.40</span>
            </label>
            <input type="range" class="form-range" min="0.20" max="0.49" step="0.05" value="0.40" id="lowerCutoffInput" oninput="updateCutoffs()">

            <label class="form-label small text-muted mb-0 mt-2 d-flex justify-content-between">
                <span>Upper Cutoff:</span>
                <span id="upperCutoffVal" class="text-danger fw-bold">0.65</span>
            </label>
            <input type="range" class="form-range" min="0.51" max="0.85" step="0.05" value="0.65" id="upperCutoffInput" oninput="updateCutoffs()">
        </div>
    </div>

    <!-- Main Content Container -->
    <div class="main-content">

        <!-- Top Utility Header with Theme Toggle -->
        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-navy">
            <div class="d-flex align-items-center gap-2">
                <span class="badge bg-primary-subtle text-primary border border-primary px-2 py-1">
                    <i class="fa-solid fa-signal me-1"></i> System Online
                </span>
                <span class="small text-muted d-none d-md-inline">PhishShift Multi-Engine Risk Scanner</span>
            </div>
            <div>
                <button class="btn btn-sm theme-top-btn d-flex align-items-center gap-2" onclick="toggleTheme()" id="topThemeToggleBtn" title="Toggle Light / Dark Mode">
                    <i id="themeTopIcon" class="fa-solid fa-sun text-warning"></i>
                    <span id="themeTopText" class="small fw-bold">Light Mode</span>
                </button>
            </div>
        </div>

        <!-- Top Navigation Pills (Quick Clickable View Switcher) -->
        <div class="cyber-card p-2 mb-4">
            <div class="d-flex flex-wrap gap-2 justify-content-start align-items-center">
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link active" data-view="dashboard" onclick="switchView('dashboard', this); return false;">
                    <i class="fa-solid fa-chart-pie me-1"></i> Dashboard <span class="badge bg-secondary ms-1">1</span>
                </button>
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link" data-view="scanner" onclick="switchView('scanner', this); return false;">
                    <i class="fa-solid fa-magnifying-glass me-1"></i> Message Scanner <span class="badge bg-secondary ms-1">2</span>
                </button>
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link" data-view="batch" onclick="switchView('batch', this); return false;">
                    <i class="fa-solid fa-file-csv me-1"></i> Batch CSV Scanner <span class="badge bg-secondary ms-1">3</span>
                </button>
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link" data-view="comparison" onclick="switchView('comparison', this); return false;">
                    <i class="fa-solid fa-scale-balanced me-1"></i> Model Comparison <span class="badge bg-secondary ms-1">4</span>
                </button>
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link" data-view="dataset" onclick="switchView('dataset', this); return false;">
                    <i class="fa-solid fa-folder-open me-1"></i> Dataset & Deduplication <span class="badge bg-secondary ms-1">5</span>
                </button>
                <button type="button" class="btn btn-sm btn-outline-info view-nav-link" data-view="robustness" onclick="switchView('robustness', this); return false;">
                    <i class="fa-solid fa-flask-vial me-1"></i> Robustness Lab <span class="badge bg-secondary ms-1">6</span>
                </button>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 1: DASHBOARD -->
        <!-- ================================================================= -->
        <div id="view-dashboard">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-shield-halved text-info me-2"></i> PhishShift Message Risk Scanner</h1>
                <p class="mb-0 text-muted small">
                    Educational cybersecurity triage tool for analyzing email and SMS communications.
                    Engineered to detect subtle phishing cues while minimizing false positives on legitimate urgent business notifications.
                </p>
            </div>

            <div class="row g-3 mb-4">
                <div class="col-sm-6 col-lg-3">
                    <div class="metric-card">
                        <div class="metric-label">Total Messages</div>
                        <div class="metric-value" id="dashTotalMsgs">420</div>
                    </div>
                </div>
                <div class="col-sm-6 col-lg-3">
                    <div class="metric-card">
                        <div class="metric-label">Phishing Callset</div>
                        <div class="metric-value text-danger" id="dashPhishCount">210</div>
                    </div>
                </div>
                <div class="col-sm-6 col-lg-3">
                    <div class="metric-card">
                        <div class="metric-label">Legitimate Callset</div>
                        <div class="metric-value text-success" id="dashLegitCount">210</div>
                    </div>
                </div>
                <div class="col-sm-6 col-lg-3">
                    <div class="metric-card">
                        <div class="metric-label">Active Campaigns</div>
                        <div class="metric-value text-info" id="dashCampaignCount">13</div>
                    </div>
                </div>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-microchip text-info me-2"></i> Active Machine Learning Detection Engines</h5>
                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-primary bg-dark">
                            <span class="badge bg-primary mb-2">Primary Linear Model</span>
                            <h6 class="fw-bold mb-1">TF-IDF + Calibrated Linear SVM</h6>
                            <p class="small text-muted mb-0">Max-margin linear separation with Platt probability scaling. Evaluated on campaign holdouts.</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-info bg-dark">
                            <span class="badge bg-info text-dark mb-2">Linear Baseline</span>
                            <h6 class="fw-bold mb-1">TF-IDF + Logistic Regression</h6>
                            <p class="small text-muted mb-0">Standard probabilistic baseline classifier trained over character & word n-gram features.</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-warning bg-dark">
                            <span class="badge bg-warning text-dark mb-2">Transparent Heuristic</span>
                            <h6 class="fw-bold mb-1">Rule-Based Behavioral Detector</h6>
                            <p class="small text-muted mb-0">Pattern analysis with urgency balancing rules to prevent false alarms on internal operations.</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-list-check text-info me-2"></i> Key Operational Safeguards</h5>
                <ul class="text-muted small ps-3 mb-0">
                    <li class="mb-2"><strong>Zero External Network Calls:</strong> Models execute 100% locally with offline weights. No user messages or credentials are transmitted off-device.</li>
                    <li class="mb-2"><strong>Safe Defanged Link Analysis:</strong> Links are sanitized and inspected structurally (e.g. <code>hxxps://</code>, <code>[.]</code>) without ever downloading content or executing attachments.</li>
                    <li class="mb-2"><strong>Leakage-Free Campaign Holdouts:</strong> Near-duplicate templates are deduplicated and partitioned by campaign cluster before splitting training and testing data.</li>
                    <li><strong>Uncertainty Rejection Margin:</strong> Borderline cases between 0.40 and 0.65 are routed to manual review rather than forcing an uncertain classification.</li>
                </ul>
            </div>

            <div class="disclaimer-box">
                <strong>DECISION SUPPORT DISCLAIMER:</strong> PhishShift is an educational triage decision-support tool. Predictions are probabilistic assessments, not guarantees of security. Advanced phishing attacks can mimic legitimate business correspondence, and legitimate operational emails may contain urgent keywords. Always independently verify sensitive requests through established channels.
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 2: MESSAGE SCANNER -->
        <!-- ================================================================= -->
        <div id="view-scanner" style="display: none;">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-magnifying-glass text-info me-2"></i> Message Risk Scanner</h1>
                <p class="mb-0 text-muted small">
                    Inspect individual email or SMS communications for social engineering lures, credential harvesting signals, and urgency triggers.
                </p>
            </div>

            <div class="row">
                <!-- Input Column -->
                <div class="col-lg-7">
                    <div class="cyber-card">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-envelope-open-text text-info me-2"></i> Enter Message to Analyze</h5>

                        <label class="form-label text-muted small mb-1">Quick Test Presets:</label>
                        <div class="d-flex flex-wrap gap-2 mb-3">
                            <button type="button" class="btn btn-preset" onclick="loadPreset('lock')">🚨 Phishing: Account Lock</button>
                            <button type="button" class="btn btn-preset" onclick="loadPreset('pwd')">✅ Legit: Password Reset</button>
                            <button type="button" class="btn btn-preset" onclick="loadPreset('urgent')">⚡ Legit: P1 Incident</button>
                            <button type="button" class="btn btn-preset" onclick="loadPreset('smish')">📱 Phishing: SMS Smishing</button>
                        </div>

                        <form id="scanForm" onsubmit="event.preventDefault(); analyzeSingleMessage();">
                            <div class="row g-2 mb-3">
                                <div class="col-md-6">
                                    <label class="form-label small text-muted">Channel:</label>
                                    <select id="channelSelect" class="form-select form-select-sm">
                                        <option value="email">Email</option>
                                        <option value="sms">SMS</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label small text-muted">Model Engine:</label>
                                    <select id="modelSelect" class="form-select form-select-sm">
                                        <option value="svm" selected>TF-IDF + Calibrated Linear SVM (Recommended)</option>
                                        <option value="lr">TF-IDF + Logistic Regression</option>
                                        <option value="rule">Rule-Based Behavioral Engine</option>
                                    </select>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label small text-muted">Subject / Title (Optional):</label>
                                <input type="text" id="subjectInput" class="form-control form-control-sm" placeholder="e.g. Critical Security Alert: Immediate Action Required">
                            </div>

                            <div class="mb-3">
                                <label class="form-label small text-muted">Message Body Text (Required):</label>
                                <textarea id="bodyInput" class="form-control" rows="5" placeholder="Paste full email body or SMS text here..." required></textarea>
                            </div>

                            <div class="d-flex gap-2">
                                <button type="submit" id="analyzeBtn" class="btn btn-primary px-4 fw-semibold">
                                    <i class="fa-solid fa-magnifying-glass me-1"></i> Scan Message Risk
                                </button>
                                <button type="button" class="btn btn-outline-secondary" onclick="clearScannerForm()">
                                    Clear
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Output Column -->
                <div class="col-lg-5">
                    <div class="cyber-card" id="scanResultsCard" style="display: none;">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-line text-info me-2"></i> Triage Assessment</h5>

                        <div id="badgeBox" class="mb-3"></div>

                        <div class="mb-3">
                            <div class="d-flex justify-content-between small text-muted mb-1">
                                <span>Calculated Risk Score</span>
                                <span id="riskScoreLabel" class="fw-bold text-info">0.0%</span>
                            </div>
                            <div class="progress" style="height: 10px; background-color: #0f172a;">
                                <div id="riskProgressBar" class="progress-bar" role="progressbar" style="width: 0%;"></div>
                            </div>
                            <div class="small text-muted mt-1" id="confidenceDesc"></div>
                        </div>

                        <div class="mb-3">
                            <h6 class="small fw-bold text-uppercase text-danger mb-2"><i class="fa-solid fa-triangle-exclamation me-1"></i> Suspicious Behavioral Cues</h6>
                            <div id="suspiciousCuesList"></div>
                        </div>

                        <div class="mb-3">
                            <h6 class="small fw-bold text-uppercase text-success mb-2"><i class="fa-solid fa-shield-check me-1"></i> Countervailing Signals</h6>
                            <div id="legitCuesList"></div>
                        </div>

                        <div class="mb-2">
                            <h6 class="small fw-bold text-uppercase text-info mb-1"><i class="fa-solid fa-compass me-1"></i> Recommended Steps</h6>
                            <ul id="recommendedStepsList" class="small text-muted ps-3 mb-0"></ul>
                        </div>

                        <div class="disclaimer-box" id="scannerDisclaimerText"></div>
                    </div>

                    <div class="cyber-card text-center py-5" id="scannerIdleCard">
                        <i class="fa-solid fa-fingerprint fa-3x text-muted mb-3 opacity-50"></i>
                        <h6 class="text-muted">Awaiting Input</h6>
                        <p class="small text-muted px-4 mb-0">
                            Select a preset above or paste an email/SMS message to evaluate phishing probabilities and behavioral cues.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 3: BATCH CSV SCANNER -->
        <!-- ================================================================= -->
        <div id="view-batch" style="display: none;">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-file-csv text-info me-2"></i> Batch CSV Spam & Phishing Scanner</h1>
                <p class="mb-0 text-muted small">
                    High-throughput batch triage engine for scanning CSV datasets with automated column detection, risk scoring, and enriched CSV export.
                </p>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-sliders text-info me-2"></i> Batch Configuration</h5>
                <div class="row g-3 mb-3">
                    <div class="col-md-6">
                        <label class="form-label small text-muted">Select Target Dataset Source:</label>
                        <select id="batchSourceSelect" class="form-select form-select-sm" onchange="onBatchSourceChange()">
                            <option value="custom" selected>📤 Upload Custom CSV File (Drag & Drop or Browse)</option>
                            <option value="synthetic">📂 Bundled Synthetic Demonstration (420 messages)</option>
                            <option value="sms_spam">📂 Bundled SMS Spam Collection (5,574 messages)</option>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label small text-muted">Scan Engine:</label>
                        <select id="batchModelSelect" class="form-select form-select-sm">
                            <option value="svm" selected>Calibrated Linear SVM</option>
                            <option value="lr">Logistic Regression</option>
                            <option value="rule">Rule-Based Detector</option>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label small text-muted">Scan Limit (Rows):</label>
                        <select id="batchLimitSelect" class="form-select form-select-sm">
                            <option value="50">First 50 rows</option>
                            <option value="100">First 100 rows</option>
                            <option value="200" selected>First 200 rows</option>
                            <option value="500">First 500 rows</option>
                            <option value="1000">First 1,000 rows</option>
                            <option value="999999">Entire CSV File</option>
                        </select>
                    </div>
                </div>

                <!-- Custom CSV Upload Drag-and-Drop Area -->
                <div id="batchUploadContainer" class="mb-3">
                    <div class="p-4 text-center rounded upload-zone" id="batchDropZone" style="border: 2px dashed var(--accent-cyan); background-color: var(--panel-bg); cursor: pointer;" onclick="document.getElementById('batchCsvFileInput').click()">
                        <input type="file" id="batchCsvFileInput" accept=".csv" style="display: none;" onchange="handleBatchFileSelect(event)">
                        <i class="fa-solid fa-cloud-arrow-up fa-2x text-info mb-2"></i>
                        <h6 class="fw-bold mb-1">Click to Browse or Drag & Drop your CSV File Here</h6>
                        <p class="small text-muted mb-0">Supports arbitrary CSV datasets (UTF-8, Latin-1, CP1252)</p>
                        <div id="batchFileInfo" class="mt-2 text-info small fw-bold" style="display: none;"></div>
                    </div>
                </div>

                <!-- Column Selectors (Auto-detected / Manual override) -->
                <div id="batchColumnSelectors" class="row g-2 mb-3" style="display: none;">
                    <div class="col-md-6">
                        <label class="form-label small text-muted">Message Text Column (Auto-Detected):</label>
                        <select id="batchTextColSelect" class="form-select form-select-sm"></select>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label small text-muted">Ground-Truth Label Column (Optional):</label>
                        <select id="batchLabelColSelect" class="form-select form-select-sm"></select>
                    </div>
                </div>

                <button type="button" id="runBatchScanBtn" class="btn btn-primary" onclick="runBatchScan()">
                    <i class="fa-solid fa-play me-1"></i> Execute Batch Scan
                </button>
            </div>

            <!-- Batch Summary Metrics -->
            <div id="batchResultsSection" style="display: none;">
                <!-- Benchmark accuracy banner if ground-truth labels present -->
                <div id="batchBenchmarkBanner" style="display: none;"></div>

                <div class="row g-3 mb-4">
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Total Evaluated</div>
                            <div class="metric-value" id="batchTotalCount">0</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Potential Phishing</div>
                            <div class="metric-value text-danger" id="batchPhishPct">0%</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Likely Legitimate</div>
                            <div class="metric-value text-success" id="batchLegitPct">0%</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Uncertain Manual Review</div>
                            <div class="metric-value text-warning" id="batchUncertainPct">0%</div>
                        </div>
                    </div>
                </div>

                <div class="cyber-card">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="fw-bold mb-0"><i class="fa-solid fa-table-list text-info me-2"></i> Scanned Messages Preview</h5>
                        <button type="button" class="btn btn-outline-info btn-sm" onclick="downloadBatchCSV()">
                            <i class="fa-solid fa-download me-1"></i> Download Enriched Results (CSV)
                        </button>
                    </div>
                    <div class="table-responsive">
                        <table class="table cyber-table table-bordered table-hover mb-0" id="batchTable">
                            <thead>
                                <tr id="batchTableHeadRow">
                                    <th>Message Text</th>
                                    <th>Category</th>
                                    <th>Risk Score</th>
                                    <th>Confidence</th>
                                    <th>Suspicious Cues</th>
                                </tr>
                            </thead>
                            <tbody id="batchTableBody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 4: MODEL COMPARISON -->
        <!-- ================================================================= -->
        <div id="view-comparison" style="display: none;">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-scale-balanced text-info me-2"></i> Model Comparison & Benchmark Evaluation</h1>
                <p class="mb-0 text-muted small">
                    Multi-model benchmarking across campaign-separated holdout partitions to detect overfitting and evaluate real generalization.
                </p>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-simple text-info me-2"></i> Holdout Evaluation Benchmark (Campaign-Separated)</h5>
                <div class="table-responsive">
                    <table class="table cyber-table table-bordered mb-0">
                        <thead>
                            <tr>
                                <th>Model Name</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>False Positive Rate (FPR)</th>
                                <th>TP</th>
                                <th>FP</th>
                                <th>TN</th>
                                <th>FN</th>
                                <th>Evaluated</th>
                            </tr>
                        </thead>
                        <tbody id="comparisonTableBody">
                            <tr><td colspan="10" class="text-center text-muted">Loading metrics...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-2"><i class="fa-solid fa-bolt text-warning me-2"></i> False Positive Impact on Legitimate Urgent Messages</h5>
                <p class="small text-muted mb-3">
                    Stress-tested against legitimate business notifications containing high-pressure keywords (e.g. "urgent", "P1 incident", "priority review", "payroll"):
                </p>
                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-warning bg-dark">
                            <div class="metric-label">Rule-Based Urgent FPR</div>
                            <div class="metric-value text-success" id="ruleUrgentFpr">0.0%</div>
                            <div class="small text-muted mt-1" id="ruleUrgentDesc">0 False Alarms</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-info bg-dark">
                            <div class="metric-label">Logistic Regression Urgent FPR</div>
                            <div class="metric-value text-warning" id="lrUrgentFpr">28.6%</div>
                            <div class="small text-muted mt-1" id="lrUrgentDesc">2 False Alarms</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 rounded border border-primary bg-dark">
                            <div class="metric-label">Calibrated Linear SVM Urgent FPR</div>
                            <div class="metric-value text-danger" id="svmUrgentFpr">57.1%</div>
                            <div class="small text-muted mt-1" id="svmUrgentDesc">4 False Alarms</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-image text-info me-2"></i> Evaluation Diagnostic Visualizations</h5>
                <div class="row g-3">
                    <div class="col-md-6">
                        <h6 class="small text-muted fw-bold">Holdout Confusion Matrices:</h6>
                        <img src="/reports/confusion_matrices.png" class="img-fluid rounded border border-navy" alt="Confusion Matrices" onerror="this.style.display='none'; document.getElementById('cmFallback').style.display='block';">
                        <div id="cmFallback" style="display:none;" class="p-4 text-center text-muted border border-navy rounded">Confusion Matrix available in reports directory</div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="small text-muted fw-bold">Benchmark Comparison:</h6>
                        <img src="/reports/model_comparison.png" class="img-fluid rounded border border-navy" alt="Model Comparison" onerror="this.style.display='none'; document.getElementById('compFallback').style.display='block';">
                        <div id="compFallback" style="display:none;" class="p-4 text-center text-muted border border-navy rounded">Benchmark chart available in reports directory</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 5: DATASET & DEDUPLICATION -->
        <!-- ================================================================= -->
        <div id="view-dataset" style="display: none;">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-folder-open text-info me-2"></i> Dataset Discovery, Schema Validation & Deduplication</h1>
                <p class="mb-0 text-muted small">
                    Inspect dataset composition, missing values, near-duplicate template clustering, and schema compliance.
                </p>
            </div>

            <div class="cyber-card">
                <div class="row g-3 align-items-center mb-3">
                    <div class="col-md-6">
                        <label class="form-label small text-muted">Select Dataset:</label>
                        <select id="datasetSelect" class="form-select form-select-sm" onchange="loadDatasetDetails()">
                            <option value="synthetic" selected>📂 Bundled Synthetic Demonstration (420 messages, 13 campaigns)</option>
                            <option value="sms_spam">📂 Bundled SMS Spam Collection (5,574 messages)</option>
                        </select>
                    </div>
                    <div class="col-md-6 text-md-end mt-4">
                        <span class="badge bg-success py-2 px-3"><i class="fa-solid fa-check me-1"></i> Strict Schema Validated</span>
                    </div>
                </div>

                <div class="row g-3 mb-4">
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Total Messages</div>
                            <div class="metric-value" id="dsTotalRows">-</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Columns</div>
                            <div class="metric-value" id="dsTotalCols">-</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Phishing / Spam Ratio</div>
                            <div class="metric-value text-danger" id="dsPhishRatio">-</div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-lg-3">
                        <div class="metric-card">
                            <div class="metric-label">Campaign Groups</div>
                            <div class="metric-value text-info" id="dsCampaigns">-</div>
                        </div>
                    </div>
                </div>

                <!-- Distribution Charts Canvas -->
                <div class="row g-3 mb-4">
                    <div class="col-md-6">
                        <div class="p-3 rounded border border-navy bg-dark">
                            <h6 class="small text-muted fw-bold mb-2">Class Balance Distribution</h6>
                            <canvas id="classBalanceChart" height="180"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="p-3 rounded border border-navy bg-dark">
                            <h6 class="small text-muted fw-bold mb-2">Campaign Distribution (Top Groups)</h6>
                            <canvas id="campaignChart" height="180"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Schema Summary -->
                <h6 class="small text-muted fw-bold mb-2">Schema Completeness & Missing Values:</h6>
                <div class="table-responsive mb-4">
                    <table class="table cyber-table table-bordered mb-0">
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Data Type</th>
                                <th>Missing Values</th>
                                <th>Fill Rate (%)</th>
                                <th>Completeness Status</th>
                            </tr>
                        </thead>
                        <tbody id="schemaTableBody"></tbody>
                    </table>
                </div>

                <!-- Deduplication Section -->
                <div class="p-3 rounded border border-primary bg-dark mb-4">
                    <h5 class="fw-bold mb-1"><i class="fa-solid fa-broom text-info me-2"></i> Template Deduplication Engine</h5>
                    <p class="small text-muted mb-3">
                        Detects exact duplicate messages and near-identical template variations by masking dynamic variables (IDs, amounts, dates, usernames, URLs) and computing Jaccard shingle similarity.
                    </p>

                    <div class="row g-3 align-items-center mb-3">
                        <div class="col-md-8">
                            <label class="form-label small text-muted d-flex justify-content-between mb-1">
                                <span>Near-Duplicate Jaccard Similarity Threshold:</span>
                                <span id="dedupThreshVal" class="text-info fw-bold">0.85</span>
                            </label>
                            <input type="range" class="form-range" min="0.60" max="0.95" step="0.05" value="0.85" id="dedupThreshInput" oninput="document.getElementById('dedupThreshVal').innerText = this.value">
                        </div>
                        <div class="col-md-4 text-md-end">
                            <button type="button" class="btn btn-outline-info" onclick="recalculateDeduplication()">
                                <i class="fa-solid fa-arrows-rotate me-1"></i> Recalculate Deduplication
                            </button>
                        </div>
                    </div>

                    <div class="row g-2 mb-3 text-center">
                        <div class="col"><div class="p-2 border rounded"><div class="small text-muted">Raw Samples</div><strong id="dedupRaw">420</strong></div></div>
                        <div class="col"><div class="p-2 border rounded"><div class="small text-muted">Exact Duplicates</div><strong id="dedupExact">0</strong></div></div>
                        <div class="col"><div class="p-2 border rounded"><div class="small text-muted">Near-Duplicates Removed</div><strong class="text-warning" id="dedupNear">210</strong></div></div>
                        <div class="col"><div class="p-2 border rounded"><div class="small text-muted">Retained Unique</div><strong class="text-success" id="dedupRetained">210</strong></div></div>
                        <div class="col"><div class="p-2 border rounded"><div class="small text-muted">Removal Ratio</div><strong class="text-info" id="dedupRatio">50.0%</strong></div></div>
                    </div>

                    <!-- Template Cluster Drilldown -->
                    <div id="clusterDrilldownBox">
                        <h6 class="small text-info fw-bold mb-2">🔍 Detected Template Group Drill-down:</h6>
                        <select id="clusterPicker" class="form-select form-select-sm mb-2" onchange="renderSelectedCluster()"></select>
                        <div class="row g-2" id="clusterInspectionRow"></div>
                    </div>
                </div>

                <!-- Provenance & Licensing Card -->
                <div class="p-3 rounded border border-navy bg-dark">
                    <h6 class="fw-bold mb-2"><i class="fa-solid fa-scroll text-info me-2"></i> Dataset Source, Provenance & Licensing</h6>
                    <ul class="small text-muted ps-3 mb-0">
                        <li><strong>Synthetic Demonstration Callset:</strong> 420 messages across 13 campaign clusters. Locally generated under the MIT License specifically for safe educational cybersecurity evaluation.</li>
                        <li><strong>SMS Spam Collection:</strong> 5,574 mobile messages compiled from the UCI Machine Learning Repository (Tiago A. Almeida and José María Gómez Hidalgo). Open Access for academic research.</li>
                        <li><strong>Privacy Safeguards:</strong> All test messages use defanged URL indicators (e.g. <code>hxxps://</code>, <code>[.]</code>), synthetic numbers, and placeholder domains. Zero enterprise credentials or personal emails are collected or stored.</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- VIEW 6: ROBUSTNESS & PERTURBATION LAB -->
        <!-- ================================================================= -->
        <div id="view-robustness" style="display: none;">
            <div class="cyber-header">
                <h1 class="h3 fw-bold mb-1"><i class="fa-solid fa-flask-vial text-info me-2"></i> Robustness, Perturbation & Error Analysis</h1>
                <p class="mb-0 text-muted small">
                    Inspects model resilience against harmless text variations, defanged URLs, punctuation changes, and representative error root causes.
                </p>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-vial-circle-check text-info me-2"></i> 8 Controlled Harmless Perturbation Tests</h5>
                <p class="small text-muted mb-3">
                    Each test applies a harmless syntax or punctuation alteration to measure classification drift and model brittleness:
                </p>

                <div class="table-responsive mb-4">
                    <table class="table cyber-table table-bordered mb-0">
                        <thead>
                            <tr>
                                <th>Test ID</th>
                                <th>Perturbation Name</th>
                                <th>Expected</th>
                                <th>LR Probability Drift</th>
                                <th>SVM Probability Drift</th>
                                <th>Rule Drift</th>
                                <th>LR Flipped?</th>
                                <th>SVM Flipped?</th>
                            </tr>
                        </thead>
                        <tbody id="pertTableBody">
                            <tr><td colspan="8" class="text-center text-muted">Loading perturbation tests...</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- Perturbation Deep Dive Accordion -->
                <h6 class="small text-muted fw-bold mb-2">Inspect Individual Perturbation Pairs:</h6>
                <div class="accordion" id="pertAccordion"></div>
            </div>

            <div class="cyber-card">
                <h5 class="fw-bold mb-2"><i class="fa-solid fa-lightbulb text-warning me-2"></i> Empirical Insights from Perturbation Study</h5>
                <ul class="small text-muted ps-3 mb-0" id="pertConclusionsList">
                    <li class="mb-2"><strong>Vocabulary Shift Vulnerability:</strong> Replacing standard URL markers with defanged patterns (e.g. <code>hxxps://</code>) can alter TF-IDF feature weights if defanged tokens were not observed in training.</li>
                    <li class="mb-2"><strong>Syntactic Urgency Drift:</strong> Adding subtle punctuation (e.g. exclamation marks, capitalized words) artificially increases linear model confidence on legitimate operational alerts.</li>
                    <li><strong>Heuristic Resilience:</strong> The rule-based engine remains stable across harmless syntactic changes because it employs explicit balancing discounts for operational context.</li>
                </ul>
            </div>
        </div>

    </div>

    <!-- JavaScript Application Logic -->
    <script>
        let currentView = 'dashboard';
        let cachedMetrics = null;
        let cachedPerturbations = null;
        let cachedDatasetInfo = null;
        let cachedDeduplication = null;
        let batchResultsCache = [];
        let classChartInstance = null;
        let campaignChartInstance = null;
        let uploadedBatchCsvText = null;
        let uploadedBatchFileName = null;
        let uploadedBatchColumns = [];
        let lastEnrichedCsv = null;

        // Switch active view
        function switchView(viewName, el) {
            if (window.event) {
                try {
                    window.event.preventDefault();
                    window.event.stopPropagation();
                } catch(e) {}
            }
            currentView = viewName;

            // Synchronize active state across both Sidebar AND Top Navigation Bar
            document.querySelectorAll('.view-nav-link').forEach(link => {
                if (link.getAttribute('data-view') === viewName) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });

            const views = ['dashboard', 'scanner', 'batch', 'comparison', 'dataset', 'robustness'];
            views.forEach(v => {
                const dom = document.getElementById('view-' + v);
                if (dom) {
                    dom.style.display = (v === viewName) ? 'block' : 'none';
                }
            });

            // Scroll to top of content
            window.scrollTo({ top: 0, behavior: 'instant' });

            if (viewName === 'comparison' && !cachedMetrics) loadMetrics();
            if (viewName === 'dataset') loadDatasetDetails();
            if (viewName === 'robustness' && !cachedPerturbations) loadPerturbations();
        }

        // Global Keyboard Shortcut Navigation (Keys 1-6 or D,S,B,C,T,R)
        window.addEventListener('keydown', function(e) {
            const tag = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toLowerCase() : '';
            if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

            const keyMap = {
                '1': 'dashboard', 'd': 'dashboard', 'D': 'dashboard',
                '2': 'scanner', 's': 'scanner', 'S': 'scanner',
                '3': 'batch', 'b': 'batch', 'B': 'batch',
                '4': 'comparison', 'c': 'comparison', 'C': 'comparison',
                '5': 'dataset', 't': 'dataset', 'T': 'dataset',
                '6': 'robustness', 'r': 'robustness', 'R': 'robustness',
            };

            if (keyMap[e.key]) {
                e.preventDefault();
                switchView(keyMap[e.key]);
            }
        });

        function updateCutoffs() {
            document.getElementById('lowerCutoffVal').innerText = document.getElementById('lowerCutoffInput').value;
            document.getElementById('upperCutoffVal').innerText = document.getElementById('upperCutoffInput').value;
        }

        // --- View 2: Scanner Logic ---
        const presets = {
            lock: {
                subj: "Account Suspension Warning - Immediate Verification Required",
                body: "Dear customer, your corporate access has been temporarily suspended due to anomalous activity. You must verify your credentials within 2 hours at hxxps://secure-verify[.]net or your account will be permanently deactivated.",
                channel: "email"
            },
            pwd: {
                subj: "Security Notification: Password Reset Requested",
                body: "Hello Alex, we received a request to reset your password for your account. If you initiated this request, use code 829-194. If you did not request this, you can safely ignore this email; no changes have been made.",
                channel: "email"
            },
            urgent: {
                subj: "P1 Incident #9481 - Production Latency Spike",
                body: "URGENT: Database latency exceeded 1200ms on us-east cluster. Core payment service degradation reported. Incident commander on bridge: please join zoom.us/j/92841924 priority review.",
                channel: "email"
            },
            smish: {
                subj: "",
                body: "USPS: Your package #US-9821 has an incorrect delivery address and cannot be delivered. Update your address at bit[.]ly/usps-track-99 to avoid return to sender.",
                channel: "sms"
            }
        };

        function loadPreset(key) {
            const p = presets[key];
            if (!p) return;
            document.getElementById('subjectInput').value = p.subj;
            document.getElementById('bodyInput').value = p.body;
            document.getElementById('channelSelect').value = p.channel;
        }

        function clearScannerForm() {
            document.getElementById('subjectInput').value = '';
            document.getElementById('bodyInput').value = '';
            document.getElementById('scanResultsCard').style.display = 'none';
            document.getElementById('scannerIdleCard').style.display = 'block';
        }

        async function analyzeSingleMessage() {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Analyzing...`;

            const payload = {
                body: document.getElementById('bodyInput').value,
                subject: document.getElementById('subjectInput').value,
                channel: document.getElementById('channelSelect').value,
                model: document.getElementById('modelSelect').value,
                lower_cutoff: document.getElementById('lowerCutoffInput').value,
                upper_cutoff: document.getElementById('upperCutoffInput').value
            };

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                renderScannerResults(data);
            } catch (err) {
                alert('Analysis error: ' + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-magnifying-glass me-1"></i> Scan Message Risk`;
            }
        }

        function renderScannerResults(data) {
            document.getElementById('scannerIdleCard').style.display = 'none';
            document.getElementById('scanResultsCard').style.display = 'block';

            const badgeBox = document.getElementById('badgeBox');
            const cat = data.predicted_category;
            if (cat === 'Potential Phishing') {
                badgeBox.innerHTML = `<div class="badge-danger-custom"><i class="fa-solid fa-triangle-exclamation me-2"></i> Potential Phishing</div>`;
            } else if (cat === 'Likely Legitimate') {
                badgeBox.innerHTML = `<div class="badge-success-custom"><i class="fa-solid fa-circle-check me-2"></i> Likely Legitimate</div>`;
            } else {
                badgeBox.innerHTML = `<div class="badge-warning-custom"><i class="fa-solid fa-shield-halved me-2"></i> Uncertain — Manual Review Recommended</div>`;
            }

            const riskPct = data.risk_score_percent.toFixed(1) + '%';
            document.getElementById('riskScoreLabel').innerText = riskPct;
            const bar = document.getElementById('riskProgressBar');
            bar.style.width = riskPct;
            bar.className = 'progress-bar ' + (data.risk_score_percent > 65 ? 'bg-danger' : (data.risk_score_percent < 40 ? 'bg-success' : 'bg-warning'));
            document.getElementById('confidenceDesc').innerText = `Confidence Tier: ${data.confidence_level} | Model: ${data.model_name}`;

            const suspBox = document.getElementById('suspiciousCuesList');
            if (data.suspicious_cues && data.suspicious_cues.length > 0) {
                suspBox.innerHTML = data.suspicious_cues.map(c => `<div class="cue-box cue-danger">• ${c}</div>`).join('');
            } else {
                suspBox.innerHTML = `<div class="small text-muted">No explicit high-risk cues triggered.</div>`;
            }

            const legitBox = document.getElementById('legitCuesList');
            if (data.legitimate_cues && data.legitimate_cues.length > 0) {
                legitBox.innerHTML = data.legitimate_cues.map(c => `<div class="cue-box cue-success">• ${c}</div>`).join('');
            } else {
                legitBox.innerHTML = `<div class="small text-muted">No countervailing legitimate operational signals identified.</div>`;
            }

            const stepsBox = document.getElementById('recommendedStepsList');
            stepsBox.innerHTML = data.recommended_steps.map(s => `<li>${s}</li>`).join('');

            document.getElementById('scannerDisclaimerText').innerText = data.mandatory_disclaimer;
        }

        // --- View 3: Batch CSV File Upload & Scanner Logic ---
        function onBatchSourceChange() {
            const src = document.getElementById('batchSourceSelect').value;
            const uploadContainer = document.getElementById('batchUploadContainer');
            const colSelectors = document.getElementById('batchColumnSelectors');
            if (src === 'custom') {
                uploadContainer.style.display = 'block';
                if (uploadedBatchColumns.length > 0) {
                    colSelectors.style.display = 'flex';
                }
            } else {
                uploadContainer.style.display = 'none';
                colSelectors.style.display = 'none';
            }
        }

        function handleBatchFileSelect(event) {
            const files = event.target.files || (event.dataTransfer ? event.dataTransfer.files : null);
            if (!files || !files.length) return;
            const file = files[0];

            if (!file.name.toLowerCase().endsWith('.csv')) {
                alert('Please select a valid .csv file.');
                return;
            }

            uploadedBatchFileName = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedBatchCsvText = e.target.result;
                const sizeKb = (file.size / 1024).toFixed(1);

                // Update UI badge
                const infoDiv = document.getElementById('batchFileInfo');
                infoDiv.style.display = 'block';
                infoDiv.innerHTML = `<i class="fa-solid fa-file-csv me-1 text-info"></i> <strong>${escapeHtml(file.name)}</strong> (${sizeKb} KB) loaded. Ready to scan!`;

                // Switch dropdown to custom
                document.getElementById('batchSourceSelect').value = 'custom';
                onBatchSourceChange();

                // Parse columns from first line
                const lines = uploadedBatchCsvText.split(String.fromCharCode(10));
                if (lines.length > 0 && lines[0].trim()) {
                    uploadedBatchColumns = parseCsvHeader(lines[0]);
                    populateColumnSelectors(uploadedBatchColumns);
                }
            };
            reader.readAsText(file);
        }

        function parseCsvHeader(headerLine) {
            const cols = [];
            let cur = '';
            let inQuotes = false;
            for (let i = 0; i < headerLine.length; i++) {
                const c = headerLine[i];
                if (c === '"') {
                    inQuotes = !inQuotes;
                } else if (c === ',' && !inQuotes) {
                    cols.push(cur.trim().replace(/^["']|["']$/g, ''));
                    cur = '';
                } else {
                    cur += c;
                }
            }
            if (cur) cols.push(cur.trim().replace(/^["']|["']$/g, ''));
            return cols.filter(c => c.length > 0);
        }

        function populateColumnSelectors(cols) {
            const textSelect = document.getElementById('batchTextColSelect');
            const labelSelect = document.getElementById('batchLabelColSelect');
            if (!textSelect || !labelSelect) return;

            textSelect.innerHTML = '<option value="">-- Auto-Detect Message Column (Recommended) --</option>';
            labelSelect.innerHTML = '<option value="">-- None / Auto-Detect Ground Truth --</option>';

            cols.forEach(c => {
                const opt1 = document.createElement('option');
                opt1.value = c;
                opt1.innerText = c;
                textSelect.appendChild(opt1);

                const opt2 = document.createElement('option');
                opt2.value = c;
                opt2.innerText = c;
                labelSelect.appendChild(opt2);
            });

            // Smart heuristics to pre-select
            const lowerCols = cols.map(c => c.toLowerCase());
            const textCandidates = ['body_text', 'message', 'text', 'sms', 'content', 'v2', 'body', 'email_text', 'mail', 'msg'];
            for (const cand of textCandidates) {
                const idx = lowerCols.indexOf(cand);
                if (idx !== -1) {
                    textSelect.value = cols[idx];
                    break;
                }
            }

            const labelCandidates = ['label', 'v1', 'category', 'class', 'target', 'spam', 'is_spam', 'type'];
            for (const cand of labelCandidates) {
                const idx = lowerCols.indexOf(cand);
                if (idx !== -1) {
                    labelSelect.value = cols[idx];
                    break;
                }
            }

            document.getElementById('batchColumnSelectors').style.display = 'flex';
        }

        function setupBatchDropZone() {
            const zone = document.getElementById('batchDropZone');
            if (!zone) return;

            ['dragenter', 'dragover'].forEach(name => {
                zone.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.style.borderColor = '#38bdf8';
                    zone.style.backgroundColor = 'rgba(56, 189, 248, 0.1)';
                }, false);
            });

            ['dragleave', 'drop'].forEach(name => {
                zone.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.style.borderColor = 'var(--accent-cyan)';
                    zone.style.backgroundColor = 'var(--panel-bg)';
                }, false);
            });

            zone.addEventListener('drop', (e) => {
                handleBatchFileSelect(e);
            }, false);
        }

        async function runBatchScan() {
            const btn = document.getElementById('runBatchScanBtn') || event.target;
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Scanning batch...`;

            const source = document.getElementById('batchSourceSelect').value;
            if (source === 'custom' && !uploadedBatchCsvText) {
                alert('Please click to browse or drag & drop a CSV file first before scanning.');
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-play me-1"></i> Execute Batch Scan`;
                return;
            }

            const textColEl = document.getElementById('batchTextColSelect');
            const labelColEl = document.getElementById('batchLabelColSelect');

            const payload = {
                dataset_type: source,
                csv_data: (source === 'custom') ? uploadedBatchCsvText : null,
                model: document.getElementById('batchModelSelect').value,
                limit: parseInt(document.getElementById('batchLimitSelect').value, 10),
                text_column: (textColEl && textColEl.value) ? textColEl.value : null,
                label_column: (labelColEl && labelColEl.value) ? labelColEl.value : null,
            };

            try {
                const res = await fetch('/api/batch-scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok || data.error) {
                    throw new Error(data.error || 'Batch scan failed on the server.');
                }
                batchResultsCache = data.records || [];
                lastEnrichedCsv = data.enriched_csv || null;
                renderBatchResults(data);
            } catch (err) {
                alert('Batch scan error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-play me-1"></i> Execute Batch Scan`;
            }
        }

        function renderBatchResults(data) {
            document.getElementById('batchResultsSection').style.display = 'block';
            const s = data.summary || {};
            const total = s.total_scanned || data.total_scanned || 0;
            const phish = s.potential_phishing_count || 0;
            const legit = s.likely_legitimate_count || 0;
            const unc = s.uncertain_review_count || 0;

            const phishPct = total > 0 ? ((phish / total) * 100).toFixed(1) : 0;
            const legitPct = total > 0 ? ((legit / total) * 100).toFixed(1) : 0;
            const uncPct = total > 0 ? ((unc / total) * 100).toFixed(1) : 0;

            document.getElementById('batchTotalCount').innerText = total.toLocaleString();
            document.getElementById('batchPhishPct').innerText = `${phish} (${phishPct}%)`;
            document.getElementById('batchLegitPct').innerText = `${legit} (${legitPct}%)`;
            document.getElementById('batchUncertainPct').innerText = `${unc} (${uncPct}%)`;

            // Ground truth benchmark banner
            const benchDiv = document.getElementById('batchBenchmarkBanner');
            if (s.ground_truth_metrics && s.ground_truth_metrics.evaluated_rows > 0) {
                const m = s.ground_truth_metrics;
                benchDiv.style.display = 'block';
                benchDiv.innerHTML = `
                    <div class="p-3 mb-3 rounded border border-info cyber-card d-flex flex-wrap align-items-center justify-content-between gap-2">
                        <div>
                            <i class="fa-solid fa-award text-warning me-2"></i>
                            <strong>Ground-Truth Label Evaluation:</strong>
                            <span class="text-muted ms-1">${m.evaluated_rows} labeled rows verified</span>
                        </div>
                        <div class="d-flex flex-wrap gap-3 small">
                            <span>Precision: <strong class="text-info">${(m.precision * 100).toFixed(1)}%</strong></span>
                            <span>Recall: <strong class="text-success">${(m.recall * 100).toFixed(1)}%</strong></span>
                            <span>F1-Score: <strong class="text-warning">${(m.f1_score * 100).toFixed(1)}%</strong></span>
                            <span>FPR: <strong class="text-danger">${(m.false_positive_rate * 100).toFixed(1)}%</strong></span>
                        </div>
                    </div>
                `;
            } else if (benchDiv) {
                benchDiv.style.display = 'none';
            }

            // Table header
            const theadRow = document.getElementById('batchTableHeadRow');
            const labelCol = data.label_column;
            theadRow.innerHTML = `
                <th>Message Text</th>
                ${labelCol ? `<th>Ground Truth (${escapeHtml(labelCol)})</th>` : ''}
                <th>Category</th>
                <th>Risk Score</th>
                <th>Confidence</th>
                <th>Suspicious Cues</th>
            `;

            // Table body
            const tbody = document.getElementById('batchTableBody');
            tbody.innerHTML = '';
            const textCol = data.text_column || 'body_text';

            (data.records || []).forEach(r => {
                const tr = document.createElement('tr');
                const cat = r.predicted_category || 'Unknown';
                const catClass = cat === 'Potential Phishing' ? 'text-danger' : (cat === 'Likely Legitimate' ? 'text-success' : 'text-warning');
                const rawText = String(r[textCol] || '');
                const snippet = rawText.length > 90 ? rawText.substring(0, 90) + '...' : rawText;
                const cues = String(r.suspicious_cues || 'None');
                const cuesSnippet = cues.length > 60 ? cues.substring(0, 60) + '...' : cues;

                tr.innerHTML = `
                    <td><code>${escapeHtml(snippet)}</code></td>
                    ${labelCol && r[labelCol] !== undefined ? `<td><span class="badge bg-secondary">${escapeHtml(String(r[labelCol]))}</span></td>` : ''}
                    <td><strong class="${catClass}">${escapeHtml(cat)}</strong></td>
                    <td>${r.risk_score_percent}%</td>
                    <td><span class="badge bg-secondary">${escapeHtml(r.confidence_level || '')}</span></td>
                    <td><small class="text-muted">${escapeHtml(cuesSnippet)}</small></td>
                `;
                tbody.appendChild(tr);
            });
        }

        function downloadBatchCSV() {
            if (lastEnrichedCsv) {
                const blob = new Blob([lastEnrichedCsv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                const prefix = uploadedBatchFileName ? uploadedBatchFileName.replace(/\\.csv$/i, '') : 'scanned';
                link.download = `phishshift_${prefix}_enriched.csv`;
                link.click();
                return;
            }

            if (!batchResultsCache.length) return alert('No batch results available to download.');
            const nl = String.fromCharCode(10);
            const headers = Object.keys(batchResultsCache[0]);
            let csv = headers.join(',') + nl;
            batchResultsCache.forEach(row => {
                csv += headers.map(h => `"${String(row[h] || '').replace(/"/g, '""')}"`).join(',') + nl;
            });
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'phishshift_scanned_batch.csv';
            link.click();
        }

        // --- View 4: Model Comparison Logic ---
        async function loadMetrics() {
            try {
                const res = await fetch('/api/metrics');
                cachedMetrics = await res.json();
                renderMetricsTable(cachedMetrics);
            } catch (err) {
                console.error(err);
            }
        }

        function renderMetricsTable(data) {
            const tbody = document.getElementById('comparisonTableBody');
            tbody.innerHTML = '';

            const models = [
                { key: 'rule_based', name: 'Rule-Based Heuristic Baseline' },
                { key: 'logistic_regression', name: 'TF-IDF + Logistic Regression' },
                { key: 'linear_svm', name: 'TF-IDF + Calibrated Linear SVM' }
            ];

            models.forEach(m => {
                const d = data[m.key];
                if (!d) return;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${m.name}</strong></td>
                    <td class="text-info">${(d.precision * 100).toFixed(1)}%</td>
                    <td class="text-success">${(d.recall * 100).toFixed(1)}%</td>
                    <td class="text-warning">${(d.f1_score * 100).toFixed(1)}%</td>
                    <td class="text-danger">${(d.false_positive_rate * 100).toFixed(1)}%</td>
                    <td>${d.true_positives}</td>
                    <td>${d.false_positives}</td>
                    <td>${d.true_negatives}</td>
                    <td>${d.false_negatives}</td>
                    <td>${d.total_samples}</td>
                `;
                tbody.appendChild(tr);
            });

            // Urgent Legit Performance
            const u = data.urgent_legit_performance || {};
            document.getElementById('ruleUrgentFpr').innerText = `${((u.rule_urgent_fpr || 0)*100).toFixed(1)}%`;
            document.getElementById('ruleUrgentDesc').innerText = `${u.rule_false_positives || 0} False Alarms`;
            document.getElementById('lrUrgentFpr').innerText = `${((u.lr_urgent_fpr || 0)*100).toFixed(1)}%`;
            document.getElementById('lrUrgentDesc').innerText = `${u.lr_false_positives || 0} False Alarms`;
            document.getElementById('svmUrgentFpr').innerText = `${((u.svm_urgent_fpr || 0)*100).toFixed(1)}%`;
            document.getElementById('svmUrgentDesc').innerText = `${u.svm_false_positives || 0} False Alarms`;
        }

        // --- View 5: Dataset & Deduplication Logic ---
        async function loadDatasetDetails() {
            const dsType = document.getElementById('datasetSelect').value;
            try {
                const res = await fetch(`/api/dataset-info?type=${dsType}`);
                cachedDatasetInfo = await res.json();
                renderDatasetView(cachedDatasetInfo);
            } catch (err) {
                console.error(err);
            }
        }

        function renderDatasetView(info) {
            document.getElementById('dsTotalRows').innerText = info.total_rows.toLocaleString();
            document.getElementById('dsTotalCols').innerText = info.columns.length;

            let phishRatio = '0.0%';
            if (info.label_counts) {
                const phishCount = info.label_counts['phishing'] || info.label_counts['spam'] || 0;
                phishRatio = ((phishCount / info.total_rows) * 100).toFixed(1) + '%';
            }
            document.getElementById('dsPhishRatio').innerText = phishRatio;
            document.getElementById('dsCampaigns').innerText = Object.keys(info.campaign_counts || {}).length || '-';

            // Schema Table
            const tbody = document.getElementById('schemaTableBody');
            tbody.innerHTML = '';
            info.columns_summary.forEach(c => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${c.name}</code></td>
                    <td><span class="badge bg-secondary">${c.type}</span></td>
                    <td>${c.missing}</td>
                    <td>${c.fill_rate}%</td>
                    <td><span class="text-success"><i class="fa-solid fa-circle-check me-1"></i> Complete</span></td>
                `;
                tbody.appendChild(tr);
            });

            // Render Chart.js
            renderDatasetCharts(info);
            recalculateDeduplication();
        }

        function renderDatasetCharts(info) {
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            const tickColor = isLight ? '#475569' : '#cbd5e1';
            const gridColor = isLight ? '#e2e8f0' : '#1e3a8a';
            const campBarColor = isLight ? '#0284c7' : '#38bdf8';

            // Chart 1: Class Balance
            const ctx1 = document.getElementById('classBalanceChart').getContext('2d');
            if (classChartInstance) classChartInstance.destroy();
            const labels = Object.keys(info.label_counts);
            const values = Object.values(info.label_counts);
            classChartInstance = new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: labels.map(l => l.toUpperCase()),
                    datasets: [{
                        label: 'Messages',
                        data: values,
                        backgroundColor: labels.map(l => l.includes('phish') || l.includes('spam') ? '#ef4444' : '#10b981'),
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: tickColor }, grid: { display: false } },
                        y: { ticks: { color: tickColor }, grid: { color: gridColor } }
                    }
                }
            });

            // Chart 2: Campaigns
            const ctx2 = document.getElementById('campaignChart').getContext('2d');
            if (campaignChartInstance) campaignChartInstance.destroy();
            const campLabels = Object.keys(info.campaign_counts);
            const campValues = Object.values(info.campaign_counts);
            campaignChartInstance = new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: campLabels.length ? campLabels : ['Standard Group'],
                    datasets: [{
                        label: 'Volume',
                        data: campValues.length ? campValues : [info.total_rows],
                        backgroundColor: campBarColor,
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: tickColor }, grid: { color: gridColor } },
                        y: { ticks: { color: tickColor, font: { size: 9 } }, grid: { display: false } }
                    }
                }
            });
        }

        // --- Theme Switching Logic ---
        function applyTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            try {
                localStorage.setItem('phishshift_theme', theme);
            } catch(e) {}

            const isLight = (theme === 'light');

            const themeIcon = document.getElementById('themeIcon');
            const themeText = document.getElementById('themeText');
            if (themeIcon && themeText) {
                themeIcon.className = isLight ? 'fa-solid fa-sun me-2 text-warning' : 'fa-solid fa-moon me-2 text-warning';
                themeText.innerText = isLight ? 'Light Mode' : 'Dark Mode';
            }

            const topIcon = document.getElementById('themeTopIcon');
            const topText = document.getElementById('themeTopText');
            if (topIcon && topText) {
                topIcon.className = isLight ? 'fa-solid fa-moon text-info' : 'fa-solid fa-sun text-warning';
                topText.innerText = isLight ? 'Dark Mode' : 'Light Mode';
            }

            // If charts are initialized, re-render with matching colors
            if (cachedDatasetInfo && classChartInstance && campaignChartInstance) {
                renderDatasetCharts(cachedDatasetInfo);
            }
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const nextTheme = (current === 'light') ? 'dark' : 'light';
            applyTheme(nextTheme);
        }

        // Initialize on page load
        window.onload = function() {
            let savedTheme = 'dark';
            try {
                savedTheme = localStorage.getItem('phishshift_theme');
            } catch(e) {}
            if (!savedTheme) {
                savedTheme = (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
            }
            applyTheme(savedTheme);
            setupBatchDropZone();
            switchView('dashboard');
        };

        async function recalculateDeduplication() {
            const dsType = document.getElementById('datasetSelect').value;
            const thresh = document.getElementById('dedupThreshInput').value;
            try {
                const res = await fetch('/api/deduplicate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_type: dsType, threshold: thresh })
                });
                cachedDeduplication = await res.json();
                renderDeduplicationView(cachedDeduplication);
            } catch (err) {
                console.error(err);
            }
        }

        function renderDeduplicationView(data) {
            const rep = data.report;
            document.getElementById('dedupRaw').innerText = rep.raw_count.toLocaleString();
            document.getElementById('dedupExact').innerText = rep.exact_duplicates_count.toLocaleString();
            document.getElementById('dedupNear').innerText = rep.near_duplicate_instances_removed.toLocaleString();
            document.getElementById('dedupRetained').innerText = rep.retained_count.toLocaleString();
            document.getElementById('dedupRatio').innerText = `${(rep.removal_ratio * 100).toFixed(1)}%`;

            const picker = document.getElementById('clusterPicker');
            picker.innerHTML = '';
            data.duplicate_clusters.forEach((c, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                opt.innerText = `Template Cluster #${idx+1} (${c.size} message variants)`;
                picker.appendChild(opt);
            });
            renderSelectedCluster();
        }

        function renderSelectedCluster() {
            if (!cachedDeduplication || !cachedDeduplication.duplicate_clusters.length) return;
            const selIdx = document.getElementById('clusterPicker').value;
            const cluster = cachedDeduplication.duplicate_clusters[selIdx];
            if (!cluster) return;

            const row = document.getElementById('clusterInspectionRow');
            row.innerHTML = `
                <div class="col-md-6">
                    <div class="p-2 border border-success rounded">
                        <small class="text-success fw-bold">Retained Master Message (Row #${cluster.retained_row_index}):</small>
                        <p class="small text-light mb-1 mt-1">${escapeHtml(cluster.retained_text)}</p>
                        <span class="badge bg-secondary">Label: ${cluster.label}</span>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-2 border border-warning rounded">
                        <small class="text-warning fw-bold">Removed Near-Duplicate Variants (${cluster.total_variants_count}):</small>
                        ${cluster.sample_variants.map(v => `<pre class="p-1 mb-1 bg-black text-muted small rounded" style="white-space: pre-wrap;">${escapeHtml(v)}</pre>`).join('')}
                    </div>
                </div>
            `;
        }

        // --- View 6: Robustness & Perturbation Logic ---
        async function loadPerturbations() {
            try {
                const res = await fetch('/api/perturbations');
                cachedPerturbations = await res.json();
                renderPerturbations(cachedPerturbations);
            } catch (err) {
                console.error(err);
            }
        }

        function renderPerturbations(data) {
            const tbody = document.getElementById('pertTableBody');
            tbody.innerHTML = '';

            const accordion = document.getElementById('pertAccordion');
            accordion.innerHTML = '';

            data.tests.forEach((t, idx) => {
                const ev = t.model_evaluations || {};
                const lr = ev.logistic_regression || {};
                const svm = ev.linear_svm || {};
                const rule = ev.rule_based || {};

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${t.test_id}</strong></td>
                    <td>${t.perturbation_name}</td>
                    <td><span class="badge bg-info text-dark">${t.expected_intent}</span></td>
                    <td>${(lr.probability_drift || 0) > 0 ? '+' : ''}${(lr.probability_drift || 0).toFixed(3)}</td>
                    <td>${(svm.probability_drift || 0) > 0 ? '+' : ''}${(svm.probability_drift || 0).toFixed(3)}</td>
                    <td>${(rule.probability_drift || 0) > 0 ? '+' : ''}${(rule.probability_drift || 0).toFixed(3)}</td>
                    <td>${lr.label_flipped ? '<span class="badge bg-danger">Flipped</span>' : '<span class="badge bg-success">Stable</span>'}</td>
                    <td>${svm.label_flipped ? '<span class="badge bg-danger">Flipped</span>' : '<span class="badge bg-success">Stable</span>'}</td>
                `;
                tbody.appendChild(tr);

                // Accordion
                const item = document.createElement('div');
                item.className = 'accordion-item bg-dark border-navy mb-2';
                item.innerHTML = `
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed bg-dark text-light border-0" type="button" data-bs-toggle="collapse" data-bs-target="#pertCollapse${idx}">
                            <strong>${t.test_id}: ${t.perturbation_name}</strong>
                        </button>
                    </h2>
                    <div id="pertCollapse${idx}" class="accordion-collapse collapse" data-bs-parent="#pertAccordion">
                        <div class="accordion-body border-top border-navy">
                            <div class="row g-2">
                                <div class="col-md-6">
                                    <small class="text-muted fw-bold">Original Message:</small>
                                    <div class="p-2 border rounded bg-black small text-light mt-1">${escapeHtml(t.original_text)}</div>
                                </div>
                                <div class="col-md-6">
                                    <small class="text-warning fw-bold">Harmless Perturbed Message:</small>
                                    <div class="p-2 border rounded bg-black small text-warning mt-1">${escapeHtml(t.perturbed_text)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                accordion.appendChild(item);
            });
        }

        function escapeHtml(text) {
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
    <!-- Bootstrap 5 JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(FULL_WEBSITE_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
