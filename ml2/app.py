"""
PhishShift Message Risk Scanner (ML-07)
A modern, professional cybersecurity web dashboard built with Streamlit.
Provides triage decision support, model comparison, campaign holdout evaluation,
and evidence-based risk explanations with strict safety safeguards.
"""

from __future__ import annotations
import json
import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Setup system path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, validate_schema
from src.deduplication import deduplicate_dataset
from src.predict import PhishShiftPredictor, UNCERTAINTY_LOWER, UNCERTAINTY_UPPER
from src.explanations import generate_explanation_advisory
from src.rule_detector import RuleBasedDetector
from src.batch_scanner import detect_columns, run_csv_batch_scan


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & DARK NAVY CYBERSECURITY THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishShift | Message Risk Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS_DARK = """
<style>
    /* Dark Navy & Blue Cybersecurity Dashboard Theme */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Completely hide Streamlit Deploy button and header deploy elements */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    header .stAppDeployButton,
    header [data-testid="stAppDeployButton"],
    button[kind="header"]:has(span:contains("Deploy")),
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    
    /* Top Header Bar */
    .cyber-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border: 1px solid #1e3a8a;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .cyber-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cyber-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Cards */
    .metric-card {
        background: #111e38;
        border: 1px solid #1e3a8a;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Risk Status Badges */
    .risk-badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fde68a;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Cue list items */
    .cue-item {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 13.5px;
        color: #e2e8f0;
    }
    .cue-item-danger {
        border-left-color: #ef4444;
    }
    .cue-item-success {
        border-left-color: #10b981;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(30, 58, 138, 0.2);
        border: 1px solid #1e40af;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 12.5px;
        color: #cbd5e1;
        margin-top: 20px;
        line-height: 1.5;
    }
</style>
"""

CUSTOM_CSS_LIGHT = """
<style>
    /* Clean High-Contrast Light Cybersecurity Dashboard Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Completely hide Streamlit Deploy button and header deploy elements */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    header .stAppDeployButton,
    header [data-testid="stAppDeployButton"],
    button[kind="header"]:has(span:contains("Deploy")),
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    
    /* Top Header Bar */
    .cyber-header {
        background: linear-gradient(135deg, #e0f2fe 0%, #e2e8f0 50%, #bae6fd 100%);
        border: 1px solid #cbd5e1;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    }
    .cyber-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0369a1;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cyber-subtitle {
        color: #475569;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0284c7;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Risk Status Badges */
    .risk-badge-danger {
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        color: #b91c1c;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-badge-success {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        color: #047857;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-badge-warning {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        color: #b45309;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Cue list items */
    .cue-item {
        background: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 13.5px;
        color: #0f172a;
    }
    .cue-item-danger {
        border-left-color: #ef4444;
    }
    .cue-item-success {
        border-left-color: #10b981;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 12.5px;
        color: #1e40af;
        margin-top: 20px;
        line-height: 1.5;
    }
</style>
"""

# -----------------------------------------------------------------------------
# CACHED DATA & ARTIFACT LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    return PhishShiftPredictor(models_dir=models_dir)


@st.cache_data
def get_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic_messages.csv")
    if os.path.exists(data_path):
        return load_dataset(data_path)
    return None, {}


@st.cache_data
def get_evaluation_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_perturbation_results():
    pert_path = os.path.join(PROJECT_ROOT, "results", "perturbation_results.json")
    if os.path.exists(pert_path):
        with open(pert_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def get_cached_deduplication(
    df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    text_col: str | None = None,
    label_col: str | None = None
):
    return deduplicate_dataset(
        df,
        similarity_threshold=similarity_threshold,
        text_col=text_col,
        label_col=label_col
    )


def plot_dataset_distributions(df: pd.DataFrame, label_col: str | None, campaign_col: str | None, channel_col: str | None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#0b1120")
    for ax in axes:
        ax.set_facecolor("#111e38")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#1e3a8a")

    # Plot 1: Class Balance
    ax1 = axes[0]
    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts()
        colors = ["#ef4444" if any(w in str(idx).lower() for w in ["phish", "spam"]) else "#10b981" for idx in counts.index]
        bars = ax1.bar([str(x).capitalize() for x in counts.index], counts.values, color=colors, edgecolor="#1e293b", width=0.45)
        ax1.set_title("Class Balance Distribution", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Message Count", color="#94a3b8", fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#38bdf8")
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + (max(counts.values) * 0.02), f"{int(h):,} ({h/len(df)*100:.1f}%)", ha="center", va="bottom", color="#f1f5f9", fontsize=9, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No label column found", ha="center", va="center", color="#94a3b8")
        ax1.set_title("Class Balance", color="#f1f5f9")

    # Plot 2: Campaign or Channel Distribution
    ax2 = axes[1]
    if campaign_col and campaign_col in df.columns and df[campaign_col].nunique() > 1:
        camp_counts = df[campaign_col].value_counts().head(10)
        bars2 = ax2.barh(list(reversed(camp_counts.index)), list(reversed(camp_counts.values)), color="#38bdf8", edgecolor="#1e3a8a", height=0.6)
        ax2.set_title(f"Campaign Groups (Top {len(camp_counts)})", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Message Count", color="#94a3b8", fontsize=10)
        ax2.grid(axis="x", linestyle="--", alpha=0.25, color="#38bdf8")
        for bar in bars2:
            w = bar.get_width()
            ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2.0, f"{int(w)}", ha="left", va="center", color="#f1f5f9", fontsize=8.5)
    elif channel_col and channel_col in df.columns:
        ch_counts = df[channel_col].value_counts()
        bars2 = ax2.bar([str(x).upper() for x in ch_counts.index], ch_counts.values, color=["#6366f1", "#06b6d4"], width=0.45)
        ax2.set_title("Channel Distribution", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
        ax2.set_ylabel("Message Count", color="#94a3b8", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#38bdf8")
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{int(h):,} ({h/len(df)*100:.1f}%)", ha="center", va="bottom", color="#f1f5f9", fontsize=9, fontweight="bold")
    else:
        text_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype).startswith('string')]
        if len(text_cols) > 0:
            lengths = df[text_cols[0]].astype(str).str.len()
            ax2.hist(lengths, bins=25, color="#0ea5e9", edgecolor="#0369a1")
            ax2.set_title("Message Character Length Distribution", color="#f1f5f9", fontsize=12, fontweight="bold", pad=10)
            ax2.set_xlabel("Character Count", color="#94a3b8", fontsize=10)
            ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#38bdf8")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Theme Mode")
    theme_choice = st.radio(
        "Appearance:",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("### 🛡️ PhishShift Navigation")
    nav_selection = st.radio(
        "Select View:",
        [
            "📊 Dashboard",
            "🔍 Message Scanner",
            "📑 Batch CSV Spam Scanner",
            "⚖️ Model Comparison",
            "📁 Dataset & Deduplication",
            "🧪 Robustness & Perturbation Lab",
        ],
        index=0,
    )
    st.divider()

    st.markdown("#### ⚙️ Uncertainty Rejection Thresholds")
    st.caption("Configurable uncertainty band for manual review triggers.")
    unc_lower = st.slider("Lower Threshold (Likely Legit Cutoff)", 0.20, 0.49, UNCERTAINTY_LOWER, 0.05)
    unc_upper = st.slider("Upper Threshold (Phishing Cutoff)", 0.51, 0.85, UNCERTAINTY_UPPER, 0.05)

# Apply dynamic theme CSS
if theme_choice == "☀️ Light Mode":
    st.markdown(CUSTOM_CSS_LIGHT, unsafe_allow_html=True)
else:
    st.markdown(CUSTOM_CSS_DARK, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 1: DASHBOARD
# -----------------------------------------------------------------------------
if nav_selection == "📊 Dashboard":
    st.markdown(
        """
        <div class="cyber-header">
            <h1 class="cyber-title">🛡️ PhishShift Message Risk Scanner</h1>
            <p class="cyber-subtitle">
                Educational cybersecurity triage tool for analyzing email and SMS communications.
                Engineered to detect subtle phishing cues while minimizing false positives on legitimate urgent business notifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, stats = get_dataset()
    metrics = get_evaluation_metrics()

    # Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total_msgs = stats.get("total_rows", 0) if stats else 0
    phish_count = stats.get("label_distribution", {}).get("phishing", 0) if stats else 0
    legit_count = stats.get("label_distribution", {}).get("legitimate", 0) if stats else 0
    campaign_count = stats.get("campaign_count", 0) if stats else 0
    
    rule_fpr = metrics.get("rule_based", {}).get("false_positive_rate", "N/A") if metrics else "N/A"
    if isinstance(rule_fpr, float):
        rule_fpr = f"{rule_fpr*100:.1f}%"

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Messages</div><div class="metric-value">{total_msgs}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Phishing Examples</div><div class="metric-value">{phish_count}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Legitimate Examples</div><div class="metric-value">{legit_count}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Campaigns</div><div class="metric-value">{campaign_count}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rule-Based FPR</div><div class="metric-value">{rule_fpr}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Executive Summary & System Status")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 🎯 Core Capabilities")
        st.markdown("""
        - **Template Deduplication**: Strips identical copies and abstracts dynamic variables (ticket IDs, names, dates, amounts) before model evaluation.
        - **Multi-Model Evaluation**: Evaluates transparent rules, TF-IDF Logistic Regression, and calibrated Linear SVM.
        - **Campaign Holdout Generalization**: Evaluates performance strictly against unseen phishing campaigns (`GroupShuffleSplit`).
        - **Uncertainty Abstention**: Routes borderline classifications to manual review instead of forcing hazardous false certainty.
        - **Urgent Legitimate False Positive Control**: Explicitly balances operational incident language against coercive threat markers.
        """)

    with col_right:
        st.markdown("#### ⚙️ Active Models Status")
        predictor = get_predictor()
        st.write("• **Rule-Based Baseline**: ✅ Active (Heuristic rules loaded)")
        st.write(f"• **TF-IDF Logistic Regression**: {'✅ Loaded' if predictor.lr_pipeline else '⚠️ Pending Training'}")
        st.write(f"• **TF-IDF Calibrated Linear SVM**: {'✅ Loaded' if predictor.svm_pipeline else '⚠️ Pending Training'}")
        
        if metrics:
            st.info(f"**Latest Evaluation**: Evaluated on holdout campaigns: `{', '.join(metrics.get('campaigns_evaluated', []))}`")
        else:
            st.warning("Model evaluation metrics pending. Run training pipeline to view holdout results.")

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ MANDATORY CYBERSECURITY DISCLAIMER:</strong>
            PhishShift is an educational triage decision-support tool. Predictions are probabilistic assessments
            based on trained patterns and explicit heuristic signals. They are NOT guarantees of safety or malice.
            Do not assume an email is harmless solely because a model assigns it a low risk score.
            Always confirm unexpected financial or credential requests via verified out-of-band communication channels.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# VIEW 2: MESSAGE SCANNER
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Message Scanner":
    st.markdown("## 🔍 Message Risk Scanner")
    st.caption("Paste or type a message below to analyze risk cues, inspect legitimate balancing factors, and receive triage guidance.")

    # Initialize session state variables for Scanner
    if "scanner_body" not in st.session_state:
        st.session_state["scanner_body"] = ""
    if "scanner_subject" not in st.session_state:
        st.session_state["scanner_subject"] = ""
    if "scanner_channel" not in st.session_state:
        st.session_state["scanner_channel"] = "Email"
    if "scanner_url_features" not in st.session_state:
        st.session_state["scanner_url_features"] = "none"
    if "scanner_sender_domain" not in st.session_state:
        st.session_state["scanner_sender_domain"] = "unknown"
    if "scanner_model" not in st.session_state:
        st.session_state["scanner_model"] = "Rule-Based Baseline"
    if "scanner_last_result" not in st.session_state:
        st.session_state["scanner_last_result"] = None

    def apply_scenario(body: str, subject: str, channel: str, url_f: str, sender: str):
        st.session_state["scanner_body"] = body
        st.session_state["scanner_subject"] = subject
        st.session_state["scanner_channel"] = channel
        st.session_state["scanner_url_features"] = url_f
        st.session_state["scanner_sender_domain"] = sender
        # Immediately compute risk score so it renders instantly
        p = get_predictor()
        p.uncertainty_lower = unc_lower
        p.uncertainty_upper = unc_upper
        st.session_state["scanner_last_result"] = p.predict_message(
            body_text=body,
            subject=subject,
            sanitized_url_features=url_f,
            sender_domain_type=sender,
            channel=channel.lower(),
            model_name=st.session_state.get("scanner_model", "Rule-Based Baseline")
        )

    # Preset message buttons
    st.markdown("##### ⚡ Quick Load Demonstration Scenarios:")
    b1, b2, b3, b4 = st.columns(4)

    if b1.button("🚨 Phishing: Account Lock", use_container_width=True):
        apply_scenario(
            body="CRITICAL ALERT: Your Microsoft 365 organization profile has been temporarily restricted due to anomalous billing records. You must verify identity at hxxps://auth[.]secure-portal[.]net within 24 hours to avoid permanent account termination.",
            subject="FINAL NOTICE: Account Suspension Pending",
            channel="Email",
            url_f="contains_defanged_url",
            sender="lookalike_spoofed"
        )
        st.rerun()

    if b2.button("🟢 Legit: Password Reset", use_container_width=True):
        apply_scenario(
            body="We received a request to reset the password for your internal account. Use the following link to choose a new password. If you did not make this request, you can safely ignore this email. We will never ask you for your password via email.",
            subject="Password Reset Verification",
            channel="Email",
            url_f="none",
            sender="trusted_internal"
        )
        st.rerun()

    if b3.button("⚡ Legit: Urgent P1 Incident", use_container_width=True):
        apply_scenario(
            body="URGENT P1 INCIDENT #INC-8492: Payment API latency spike observed across EU clusters. SRE incident response war room active on Slack #incident-p1. No end-user credential action or password change required.",
            subject="URGENT: Incident INC-8492 - Payment Latency",
            channel="Email",
            url_f="none",
            sender="trusted_internal"
        )
        st.rerun()

    if b4.button("📱 Phishing: SMS Smishing", use_container_width=True):
        apply_scenario(
            body="USPS Alert: Package tracking #US948194 cannot be delivered due to missing apartment unit number. Confirm delivery address within 12h: hxxps://bit[.]ly/usps-track-938",
            subject="",
            channel="SMS",
            url_f="contains_defanged_url;contains_shortener",
            sender="defanged_shortlink"
        )
        st.rerun()

    st.markdown("### ✍️ Message Input & Parameters")
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c1:
        chan_opts = ["Email", "SMS"]
        default_chan_idx = chan_opts.index(st.session_state["scanner_channel"]) if st.session_state["scanner_channel"] in chan_opts else 0
        selected_chan = st.selectbox("Channel", chan_opts, index=default_chan_idx)
        st.session_state["scanner_channel"] = selected_chan

    with col_c2:
        model_opts = ["Rule-Based Baseline", "TF-IDF + Logistic Regression", "TF-IDF + Linear SVM (Calibrated)"]
        default_mod_idx = model_opts.index(st.session_state["scanner_model"]) if st.session_state["scanner_model"] in model_opts else 0
        selected_mod = st.selectbox("Detection Model", model_opts, index=default_mod_idx)
        st.session_state["scanner_model"] = selected_mod

    with col_c3:
        sender_opts = ["unknown", "trusted_internal", "external_freemail", "lookalike_spoofed", "defanged_shortlink", "telecom_shortcode"]
        default_send_idx = sender_opts.index(st.session_state["scanner_sender_domain"]) if st.session_state["scanner_sender_domain"] in sender_opts else 0
        selected_send = st.selectbox("Sender Domain Type (Optional)", sender_opts, index=default_send_idx)
        st.session_state["scanner_sender_domain"] = selected_send

    if st.session_state["scanner_channel"] == "Email":
        subj_val = st.text_input("Email Subject (Optional)", value=st.session_state["scanner_subject"])
        st.session_state["scanner_subject"] = subj_val
    else:
        st.session_state["scanner_subject"] = ""

    body_val = st.text_area(
        "Message Body (Required)",
        value=st.session_state["scanner_body"],
        height=140,
        placeholder="Paste raw email text or SMS message body here..."
    )
    st.session_state["scanner_body"] = body_val

    url_val = st.text_input(
        "Sanitized URL Features (Optional)",
        value=st.session_state["scanner_url_features"],
        help="E.g., 'contains_defanged_url', 'contains_shortener', 'contains_ip_url', or 'none'"
    )
    st.session_state["scanner_url_features"] = url_val

    btn_scan_col, btn_clear_col = st.columns([1, 4])
    
    with btn_scan_col:
        scan_clicked = st.button("🛡️ Scan Message Risk", type="primary", use_container_width=True)

    with btn_clear_col:
        if st.button("🧹 Clear Input", use_container_width=False):
            st.session_state["scanner_body"] = ""
            st.session_state["scanner_subject"] = ""
            st.session_state["scanner_url_features"] = "none"
            st.session_state["scanner_sender_domain"] = "unknown"
            st.session_state["scanner_last_result"] = None
            st.rerun()

    if scan_clicked:
        if not st.session_state["scanner_body"].strip():
            st.error("Please enter or paste a message body to analyze.")
            st.session_state["scanner_last_result"] = None
        else:
            predictor = get_predictor()
            predictor.uncertainty_lower = unc_lower
            predictor.uncertainty_upper = unc_upper

            try:
                st.session_state["scanner_last_result"] = predictor.predict_message(
                    body_text=st.session_state["scanner_body"],
                    subject=st.session_state["scanner_subject"],
                    sanitized_url_features=st.session_state["scanner_url_features"],
                    sender_domain_type=st.session_state["scanner_sender_domain"],
                    channel=st.session_state["scanner_channel"].lower(),
                    model_name=st.session_state["scanner_model"]
                )
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                st.session_state["scanner_last_result"] = None

    # Render analysis results if available
    if st.session_state["scanner_last_result"] is not None:
        result = st.session_state["scanner_last_result"]
        advisory = generate_explanation_advisory(result)

        st.markdown("---")
        st.markdown("### 📋 Analysis Results & Risk Assessment")

        # Category Badge
        cat = result["category"]
        prob = result["phishing_probability"]
        conf = result["confidence_level"]

        if cat == "Potential Phishing":
            st.markdown(f'<div class="risk-badge-danger">🚨 {cat} • Risk Score: {prob*100:.1f}% ({conf} Confidence)</div>', unsafe_allow_html=True)
        elif cat == "Likely Legitimate":
            st.markdown(f'<div class="risk-badge-success">✅ {cat} • Risk Score: {prob*100:.1f}% ({conf} Confidence)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-badge-warning">⚠️ {cat} • Risk Score: {prob*100:.1f}% ({conf} Confidence)</div>', unsafe_allow_html=True)

        st.progress(float(prob))
        st.caption(f"**Score Interpretation**: {result['score_type_explanation']}")

        # Explanation Columns
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("#### 🚨 Suspicious Cues Detected")
            if advisory["suspicious_cues"]:
                for cue in advisory["suspicious_cues"]:
                    st.markdown(f'<div class="cue-item cue-item-danger">• {cue}</div>', unsafe_allow_html=True)
            else:
                st.info("No explicit high-risk heuristic triggers identified.")

        with col_exp2:
            st.markdown("#### 🛡️ Countervailing Legitimate Signals")
            if advisory["legitimate_cues"]:
                for cue in advisory["legitimate_cues"]:
                    st.markdown(f'<div class="cue-item cue-item-success">• {cue}</div>', unsafe_allow_html=True)
            else:
                st.info("No countervailing legitimate operational signals identified.")

        st.markdown("#### 🧭 Recommended Safe Next Steps")
        for step in advisory["recommended_steps"]:
            st.write(f"• {step}")

        with st.expander("ℹ️ Model Limitations & Extraction Details"):
            st.write(f"**Selected Model**: {result['model_name']}")
            st.write(f"**Model Limitations**: {advisory['model_limitations']}")
            st.json(result["extracted_features"])

        st.markdown(f'<div class="disclaimer-box">{advisory["mandatory_disclaimer"]}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 2B: BATCH CSV SPAM SCANNER
# -----------------------------------------------------------------------------
elif nav_selection == "📑 Batch CSV Spam Scanner":
    st.markdown("## 📑 Batch CSV Spam & Phishing Scanner")
    st.caption("Upload or select any CSV file containing spam, ham, or email messages for high-throughput batch risk assessment, uncertainty routing, and enriched export.")

    # Data Source Selection
    source_choice = st.radio(
        "Choose Data Source:",
        [
            "📂 Bundled SMS Spam Collection (5,574 messages: ham & spam)",
            "📂 Bundled Synthetic Demonstration (420 messages: email & sms)",
            "📤 Upload Custom CSV File",
        ],
        horizontal=True,
    )

    batch_df = None
    source_filename = ""

    if "SMS Spam Collection" in source_choice:
        spam_path = os.path.join(PROJECT_ROOT, "data", "sample_spam_collection.csv")
        if os.path.exists(spam_path):
            try:
                batch_df = pd.read_csv(spam_path, encoding="latin1")
                source_filename = "sample_spam_collection.csv"
            except Exception as e:
                st.error(f"Error loading bundled SMS spam dataset: {e}")
        else:
            st.error("Bundled sample_spam_collection.csv not found.")

    elif "Synthetic Demonstration" in source_choice:
        synth_path = os.path.join(PROJECT_ROOT, "data", "synthetic_messages.csv")
        if os.path.exists(synth_path):
            try:
                batch_df = pd.read_csv(synth_path, encoding="utf-8")
                source_filename = "synthetic_messages.csv"
            except Exception as e:
                st.error(f"Error loading synthetic dataset: {e}")
        else:
            st.error("Bundled synthetic_messages.csv not found.")

    else:
        uploaded_csv = st.file_uploader("Upload CSV file containing messages to scan", type=["csv"])
        if uploaded_csv is not None:
            try:
                # Try UTF-8 then Latin-1
                try:
                    batch_df = pd.read_csv(uploaded_csv, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded_csv.seek(0)
                    batch_df = pd.read_csv(uploaded_csv, encoding="latin1")
                source_filename = uploaded_csv.name
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if batch_df is not None:
        source_key = f"{source_choice}_{source_filename}_{len(batch_df)}"
        if st.session_state.get("batch_active_source_key") != source_key:
            st.session_state["batch_active_source_key"] = source_key
            st.session_state["batch_results"] = None
            st.session_state["batch_summary"] = None

        total_rows = len(batch_df)
        st.success(f"Loaded **{source_filename}** with **{total_rows:,} rows** and columns: `{', '.join(batch_df.columns)}`")

        # Column Auto-detection
        det_text, det_label, det_subject = detect_columns(batch_df)
        col_list = list(batch_df.columns)

        st.markdown("### ⚙️ Column Mapping & Scan Configuration")
        c_map1, c_map2, c_map3 = st.columns(3)

        with c_map1:
            text_col_selected = st.selectbox(
                "Message Body / Text Column (Required)",
                options=col_list,
                index=col_list.index(det_text) if det_text in col_list else 0,
                help="Column containing the primary message body or SMS text."
            )
        with c_map2:
            label_options = ["None (Unlabeled)"] + col_list
            default_label_idx = (label_options.index(det_label)) if det_label in label_options else 0
            label_col_selected = st.selectbox(
                "Ground-Truth Label Column (Optional)",
                options=label_options,
                index=default_label_idx,
                help="Optional column containing spam/ham or phishing/legitimate labels for accuracy evaluation."
            )
        with c_map3:
            subject_options = ["None"] + col_list
            default_subj_idx = (subject_options.index(det_subject)) if det_subject in subject_options else 0
            subject_col_selected = st.selectbox(
                "Subject Column (Optional)",
                options=subject_options,
                index=default_subj_idx,
                help="Optional email subject line."
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            batch_model = st.selectbox(
                "Detection Model for Batch Scan",
                ["TF-IDF + Logistic Regression", "Rule-Based Baseline", "TF-IDF + Linear SVM (Calibrated)"]
            )
        with col_opt2:
            batch_channel = st.selectbox("Message Channel", ["sms", "email"], index=0 if "sms" in source_filename.lower() else 1)
        with col_opt3:
            max_scan = st.slider(
                "Rows to Scan",
                min_value=10,
                max_value=total_rows,
                value=min(150, total_rows),
                step=10 if total_rows <= 500 else 50,
                help="Limit row count for rapid interactive triage, or slide to the right to scan the full dataset."
            )

        if st.button("🚀 Run Batch CSV Risk Scan", type="primary", use_container_width=True):
            predictor = get_predictor()
            predictor.uncertainty_lower = unc_lower
            predictor.uncertainty_upper = unc_upper

            with st.spinner(f"Scanning {max_scan:,} messages using {batch_model}..."):
                progress_bar = st.progress(0.0)
                
                # Execute batch scan
                enriched_results, batch_stats = run_csv_batch_scan(
                    df=batch_df,
                    text_col=text_col_selected,
                    predictor=predictor,
                    model_name=batch_model,
                    subject_col=None if subject_col_selected == "None" else subject_col_selected,
                    label_col=None if label_col_selected == "None (Unlabeled)" else label_col_selected,
                    channel=batch_channel,
                    max_rows=max_scan
                )
                progress_bar.progress(1.0)
                st.session_state["batch_scan_results"] = (enriched_results, batch_stats)

        # Render Batch Results if available in session
        if "batch_scan_results" in st.session_state:
            enriched_results, batch_stats = st.session_state["batch_scan_results"]

            st.markdown("---")
            st.markdown("### 📊 Batch Scan Summary & Risk Triage Breakdown")

            m1, m2, m3, m4, m5 = st.columns(5)
            scanned_n = batch_stats["total_scanned"]
            phish_n = batch_stats["potential_phishing_count"]
            legit_n = batch_stats["likely_legitimate_count"]
            unc_n = batch_stats["uncertain_review_count"]
            mean_r = batch_stats["mean_risk_percent"]

            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Scanned</div><div class="metric-value">{scanned_n}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Potential Phishing</div><div class="metric-value" style="color: #ef4444;">{phish_n} ({phish_n/scanned_n*100:.1f}%)</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Likely Legitimate</div><div class="metric-value" style="color: #10b981;">{legit_n} ({legit_n/scanned_n*100:.1f}%)</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Uncertain / Review</div><div class="metric-value" style="color: #f59e0b;">{unc_n} ({unc_n/scanned_n*100:.1f}%)</div></div>', unsafe_allow_html=True)
            with m5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Risk Score</div><div class="metric-value">{mean_r:.1f}%</div></div>', unsafe_allow_html=True)

            # Ground Truth Evaluation Card (if label column was evaluated)
            if "ground_truth_metrics" in batch_stats:
                gt = batch_stats["ground_truth_metrics"]
                cm = gt["confusion_matrix"]
                st.markdown("#### 🎯 Ground-Truth Benchmark Against File Labels")
                st.info(
                    f"**Benchmark on {gt['evaluated_rows']} Ground-Truth Rows**: "
                    f"Precision: **{gt['precision']:.4f}** | Recall: **{gt['recall']:.4f}** | "
                    f"F1-Score: **{gt['f1_score']:.4f}** | False Positive Rate: **{gt['false_positive_rate']*100:.2f}%**"
                )
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("True Positives (Spam/Phish Caught)", cm["tp"])
                g2.metric("False Positives (Legit Flagged)", cm["fp"])
                g3.metric("True Negatives (Legit Clean)", cm["tn"])
                g4.metric("False Negatives (Spam Missed)", cm["fn"])

            # Filterable Results Table
            st.markdown("#### 📋 Scanned Messages Inspection")
            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                filter_cat = st.selectbox(
                    "Filter by Predicted Risk Category",
                    ["All Categories", "Potential Phishing Only", "Likely Legitimate Only", "Uncertain Review Only"]
                )
            with filter_col2:
                search_query = st.text_input("🔍 Search within message text", placeholder="Type a keyword to filter...")

            filtered_df = enriched_results.copy()
            active_text_col = text_col_selected if text_col_selected in enriched_results.columns else (
                "message" if "message" in enriched_results.columns else (
                    "body_text" if "body_text" in enriched_results.columns else enriched_results.columns[0]
                )
            )

            if filter_cat == "Potential Phishing Only":
                filtered_df = filtered_df[filtered_df["predicted_category"] == "Potential Phishing"]
            elif filter_cat == "Likely Legitimate Only":
                filtered_df = filtered_df[filtered_df["predicted_category"] == "Likely Legitimate"]
            elif filter_cat == "Uncertain Review Only":
                filtered_df = filtered_df[filtered_df["predicted_category"] == "Uncertain — Manual Review Recommended"]

            if search_query.strip() and active_text_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[active_text_col].astype(str).str.contains(search_query, case=False)]

            display_cols = [c for c in [active_text_col, "predicted_category", "risk_score_percent", "confidence_level", "suspicious_cues", "ground_truth_standardized"] if c in filtered_df.columns]

            st.dataframe(filtered_df[display_cols], width="stretch", height=320)

            # Download Enriched CSV
            csv_export = enriched_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Enriched CSV Results",
                data=csv_export,
                file_name=f"phishshift_scanned_{source_filename}",
                mime="text/csv",
                help="Exports complete dataset with appended risk scores, predicted categories, and detected cues."
            )

            # Individual Row Drill-down Inspector
            with st.expander("🔍 Deep-Dive Message Inspector"):
                row_idx = st.number_input("Select Row Index to Inspect Details", min_value=0, max_value=len(enriched_results)-1, value=0)
                selected_row = enriched_results.iloc[row_idx]
                
                st.markdown(f"**Selected Message (Row #{row_idx}):**")
                st.text(selected_row[active_text_col] if active_text_col in selected_row else str(selected_row.iloc[0]))
                
                det_c1, det_c2 = st.columns(2)
                with det_c1:
                    st.write(f"**Predicted Category**: `{selected_row['predicted_category']}`")
                    st.write(f"**Risk Score**: `{selected_row['risk_score_percent']:.1f}%` ({selected_row['confidence_level']})")
                    st.write(f"**Suspicious Cues**: `{selected_row['suspicious_cues']}`")
                with det_c2:
                    st.write(f"**Legitimate Signals**: `{selected_row['legitimate_signals']}`")
                    if "ground_truth_standardized" in selected_row:
                        st.write(f"**Ground Truth Label**: `{selected_row['ground_truth_standardized']}`")


# -----------------------------------------------------------------------------
# VIEW 3: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif nav_selection == "⚖️ Model Comparison":
    st.markdown("## ⚖️ Model Comparison & Campaign Holdout Evaluation")
    st.caption("Rigorous side-by-side evaluation across all three detection models on held-out campaigns.")

    metrics = get_evaluation_metrics()
    if not metrics:
        st.warning("⚠️ Evaluation metrics have not been generated yet. Please run `python scripts/train_and_evaluate.py` to evaluate models.")
    else:
        st.markdown("### 📊 Performance Metrics Table")

        comparison_data = []
        for key, display_name in [
            ("rule_based", "1. Rule-Based Baseline"),
            ("logistic_regression", "2. TF-IDF + Logistic Regression"),
            ("linear_svm", "3. TF-IDF + Calibrated Linear SVM"),
        ]:
            m = metrics.get(key, {})
            if "precision" in m:
                comparison_data.append({
                    "Model": display_name,
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1_score']:.4f}",
                    "False Positive Rate (FPR)": f"{m['false_positive_rate']*100:.2f}%",
                    "TP": m["true_positives"],
                    "FP": m["false_positives"],
                    "TN": m["true_negatives"],
                    "FN": m["false_negatives"],
                    "Total Evaluated": m["total_samples"],
                })
            else:
                comparison_data.append({
                    "Model": display_name,
                    "Precision": "Pending",
                    "Recall": "Pending",
                    "F1-Score": "Pending",
                    "False Positive Rate (FPR)": "Pending",
                    "TP": "-", "FP": "-", "TN": "-", "FN": "-", "Total Evaluated": "-",
                })

        st.dataframe(pd.DataFrame(comparison_data), width="stretch")

        # Urgent Legitimate Performance
        urgent_perf = metrics.get("urgent_legit_performance", {})
        if urgent_perf:
            st.markdown("### ⚡ False Positive Impact on Legitimate Urgent Messages")
            st.info(
                f"**Specialized Stress Test**: Evaluated on **{urgent_perf.get('total_urgent_legitimate_samples')} legitimate business messages** "
                "containing words such as 'urgent', 'P1 incident', 'immediate', or 'priority'. "
                "Standard keywords can lead to catastrophic false alarms in operational settings."
            )
            uc1, uc2, uc3 = st.columns(3)
            with uc1:
                st.metric("Rule-Based Urgent FPR", f"{urgent_perf.get('rule_urgent_fpr', 0)*100:.1f}%", f"{urgent_perf.get('rule_false_positives')} False Alarms")
            with uc2:
                st.metric("Logistic Regression Urgent FPR", f"{urgent_perf.get('lr_urgent_fpr', 0)*100:.1f}%", f"{urgent_perf.get('lr_false_positives')} False Alarms")
            with uc3:
                st.metric("Calibrated Linear SVM Urgent FPR", f"{urgent_perf.get('svm_urgent_fpr', 0)*100:.1f}%", f"{urgent_perf.get('svm_false_positives')} False Alarms")

        # Visualizations
        st.markdown("### 📈 Evaluation Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        
        cm_img_path = os.path.join(PROJECT_ROOT, "reports", "confusion_matrices.png")
        comp_img_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.png")

        with chart_col1:
            st.markdown("#### 🧩 Confusion Matrices (Holdout Campaigns)")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, width="stretch")
            else:
                st.info("Confusion matrix chart pending.")

        with chart_col2:
            st.markdown("#### 📊 Metric Benchmark Comparison")
            if os.path.exists(comp_img_path):
                st.image(comp_img_path, width="stretch")
            else:
                st.info("Comparison chart pending.")


# -----------------------------------------------------------------------------
# VIEW 4: DATASET & DEDUPLICATION
# -----------------------------------------------------------------------------
elif nav_selection == "📁 Dataset & Deduplication":
    st.markdown("## 📁 Dataset Discovery, Schema Validation & Deduplication")
    st.caption("Inspects dataset size, label distribution, missing values, near-duplicate template clustering, and schema compliance.")

    st.markdown("### 🗂️ Select Active Dataset")
    dataset_source = st.radio(
        "Choose dataset to inspect, validate, and deduplicate:",
        [
            "📂 Bundled Synthetic Demonstration Dataset (420 messages, 13 campaigns)",
            "📂 Bundled SMS Spam Collection (5,574 messages: ham & spam)",
            "📤 Upload Custom CSV File",
        ],
        horizontal=True,
    )

    active_df = None
    active_source_name = ""
    is_strict_valid = False
    validation_errors = []
    summary_stats = {}

    if "Synthetic Demonstration" in dataset_source:
        active_df, summary_stats = get_dataset()
        active_source_name = "synthetic_messages.csv"
        if active_df is not None:
            is_strict_valid, validation_errors, _ = validate_schema(active_df)

    elif "SMS Spam Collection" in dataset_source:
        spam_path = os.path.join(PROJECT_ROOT, "data", "sample_spam_collection.csv")
        if os.path.exists(spam_path):
            try:
                active_df = pd.read_csv(spam_path, encoding="latin1")
                active_source_name = "sample_spam_collection.csv"
                is_strict_valid, validation_errors, summary_stats = validate_schema(active_df)
            except Exception as e:
                st.error(f"Error loading SMS spam collection: {e}")
        else:
            st.error("Bundled sample_spam_collection.csv not found.")

    else:
        uploaded_file = st.file_uploader("Upload CSV formatted per standard or custom schema", type=["csv"])
        if uploaded_file is not None:
            try:
                try:
                    active_df = pd.read_csv(uploaded_file, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    active_df = pd.read_csv(uploaded_file, encoding="latin1")
                active_source_name = uploaded_file.name
                is_strict_valid, validation_errors, summary_stats = validate_schema(active_df)
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

    if active_df is not None and len(active_df) > 0:
        det_text, det_label, det_subject = detect_columns(active_df)
        active_text_col = det_text if det_text in active_df.columns else active_df.columns[0]
        active_label_col = det_label if det_label in active_df.columns else (
            "label" if "label" in active_df.columns else None
        )
        active_campaign_col = "campaign_group" if "campaign_group" in active_df.columns else None
        active_channel_col = "channel" if "channel" in active_df.columns else None

        # Schema Validation Banner
        if is_strict_valid:
            st.success(f"✅ **Strict Schema Validation Passed**: All 9 required fields present with valid categories in `{active_source_name}`.")
        else:
            with st.expander("ℹ️ Schema Inspection & Compatibility Report", expanded=False):
                st.markdown(f"**Dataset**: `{active_source_name}` | **Text Column**: `{active_text_col}` | **Label Column**: `{active_label_col or 'Unlabeled'}`")
                if validation_errors:
                    st.write("**Schema Notices / Missing Fields for Multi-Campaign Holdout:**")
                    for err in validation_errors:
                        st.write(f"- {err}")
                st.caption("Standard deduplication and distribution analysis remain fully operational across all text columns.")

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Messages", f"{len(active_df):,}")
        m2.metric("Total Columns", len(active_df.columns))
        if active_label_col and active_label_col in active_df.columns:
            phish_pct = (active_df[active_label_col].astype(str).str.lower().isin(["phishing", "spam"])).mean() * 100
            m3.metric("Phishing / Spam Ratio", f"{phish_pct:.1f}%")
        else:
            m3.metric("Labels", "Unlabeled")

        if active_campaign_col and active_campaign_col in active_df.columns:
            m4.metric("Campaign Groups", active_df[active_campaign_col].nunique())
        elif active_channel_col and active_channel_col in active_df.columns:
            sms_pct = (active_df[active_channel_col].astype(str).str.lower() == "sms").mean() * 100
            m4.metric("SMS Ratio", f"{sms_pct:.1f}%")
        else:
            m4.metric("Avg Character Length", f"{active_df[active_text_col].astype(str).str.len().mean():.0f} chars")

        st.divider()

        # Visualizations (Class Balance & Campaign/Channel Distribution)
        st.markdown("### 📊 Distribution Visualizations")
        fig_dist = plot_dataset_distributions(active_df, active_label_col, active_campaign_col, active_channel_col)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        # Missing Values & Schema Summary
        st.markdown("### 🔍 Missing Values & Column Completeness")
        col_summary_data = []
        for c in active_df.columns:
            null_count = int(active_df[c].isna().sum())
            fill_rate = ((len(active_df) - null_count) / len(active_df)) * 100
            col_summary_data.append({
                "Column Name": c,
                "Data Type": str(active_df[c].dtype),
                "Missing Values": null_count,
                "Fill Rate (%)": f"{fill_rate:.1f}%",
                "Status": "Complete (100%)" if null_count == 0 else f"{null_count} Missing",
            })
        st.dataframe(pd.DataFrame(col_summary_data), width="stretch")

        st.divider()

        # Deduplication Section
        st.markdown("### 🧹 Template Deduplication Engine")
        st.caption(
            "Detects exact duplicate messages and near-identical template variations by masking dynamic variables "
            "(ticket IDs, amounts, dates, usernames, URLs) and computing Jaccard shingle similarity."
        )

        dedup_ctrl1, dedup_ctrl2 = st.columns([2, 1])
        with dedup_ctrl1:
            similarity_thresh = st.slider(
                "Near-Duplicate Similarity Threshold (Jaccard)",
                min_value=0.60,
                max_value=0.95,
                value=0.85,
                step=0.05,
                help="Higher threshold requires closer template match. Standard recommended cutoff is 0.85."
            )
        with dedup_ctrl2:
            st.write("")
            st.write("")
            recalc = st.button("🔄 Recalculate Deduplication", use_container_width=True)

        with st.spinner("Analyzing message signatures and computing near-duplicate clusters..."):
            deduped_df, dedup_rep = get_cached_deduplication(
                active_df,
                similarity_threshold=similarity_thresh,
                text_col=active_text_col,
                label_col=active_label_col,
            )

        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
        dc1.metric("Raw Samples", f"{dedup_rep['raw_count']:,}")
        dc2.metric("Exact Duplicates", f"{dedup_rep['exact_duplicates_count']:,}")
        dc3.metric("Near-Duplicate Templates Removed", f"{dedup_rep['near_duplicate_instances_removed']:,}")
        dc4.metric("Retained Unique Messages", f"{dedup_rep['retained_count']:,}")
        dc5.metric("Template Removal Ratio", f"{dedup_rep['removal_ratio']*100:.1f}%")

        st.info(
            f"**Deduplication Strategy Applied**: Abstracted dynamic entities (tokens, amounts, dates, URLs) followed by "
            f"inverted-index Jaccard clustering at a **{similarity_thresh:.2f} threshold**. "
            f"This prevents template memorization between training and evaluation splits."
        )

        # Template Cluster Inspector
        dup_clusters = dedup_rep.get("duplicate_clusters", [])
        if dup_clusters:
            with st.expander(f"🔍 Inspect Detected Template Clusters & Message Variants ({len(dup_clusters)} Multi-Message Template Groups)", expanded=True):
                st.caption("Inspect how messages with dynamic variations (ticket numbers, employee names, dates) were grouped into single template clusters:")
                cluster_options = [f"Cluster #{idx+1} ({len(c)} messages in group)" for idx, c in enumerate(dup_clusters)]
                sel_cluster_label = st.selectbox("Select Template Group to Inspect:", cluster_options)
                sel_idx = cluster_options.index(sel_cluster_label)
                chosen_cluster = dup_clusters[sel_idx]

                rep_idx = chosen_cluster[0]
                rep_row = active_df.iloc[rep_idx]

                c_ins1, c_ins2 = st.columns(2)
                with c_ins1:
                    st.markdown(f"**Retained Primary Template (Row #{rep_idx}):**")
                    st.success(rep_row[active_text_col])
                    if active_label_col and active_label_col in rep_row:
                        st.caption(f"Ground Truth Label: `{rep_row[active_label_col]}`")
                with c_ins2:
                    st.markdown(f"**Near-Duplicate Template Variations Removed ({len(chosen_cluster)-1} messages):**")
                    for v_idx in chosen_cluster[1:6]:
                        v_row = active_df.iloc[v_idx]
                        st.code(str(v_row[active_text_col])[:220] + ("..." if len(str(v_row[active_text_col])) > 220 else ""), language="text")
                    if len(chosen_cluster) > 6:
                        st.caption(f"... plus {len(chosen_cluster) - 6} more near-duplicate messages grouped in this template cluster.")

        # Download Deduplicated CSV
        dedup_csv = deduped_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Deduplicated Clean Dataset (CSV)",
            data=dedup_csv,
            file_name=f"deduplicated_{active_source_name}",
            mime="text/csv",
            help="Exports clean dataset with exact duplicates and near-identical template variants removed."
        )

        st.divider()

        # Dataset Sample Preview
        st.markdown("### 📄 Dataset Preview (First 10 Rows)")
        st.dataframe(active_df.head(10), width="stretch")

    else:
        st.info("Please select or upload a dataset to begin discovery, schema validation, and deduplication.")

    # Dataset Source & Licensing Card
    st.divider()
    st.markdown("### 📜 Dataset Source, Provenance & Licensing")
    st.markdown(
        """
        - **Synthetic Demonstration Callset**: 420 messages across 13 campaign clusters. Locally generated under the MIT License specifically for safe educational cybersecurity evaluation.
        - **SMS Spam Collection**: 5,574 mobile messages compiled from the UCI Machine Learning Repository (Tiago A. Almeida and José María Gómez Hidalgo). Open Access for academic and educational research.
        - **Privacy & Safety Safeguards**: All test messages use defanged URL indicators (e.g., `hxxps://`, `[.]`), synthetic phone numbers, and placeholder domains. No external cloud endpoints are contacted, and zero sensitive enterprise credentials or personal emails are collected or stored.
        """
    )


# -----------------------------------------------------------------------------
# VIEW 5: ROBUSTNESS & PERTURBATION LAB
# -----------------------------------------------------------------------------
elif nav_selection == "🧪 Robustness & Perturbation Lab":
    st.markdown("## 🧪 Robustness, Perturbation & Error Analysis")
    st.caption("Inspects model resilience against harmless text variations, defanged URLs, punctuation changes, and representative error root causes.")

    pert_summary = get_perturbation_results()
    metrics = get_evaluation_metrics()

    st.markdown("### 🔬 8 Controlled Harmless Perturbation Tests")
    if pert_summary:
        st.markdown("Each test applies a harmless syntax or punctuation alteration to measure classification drift and model brittleness:")
        
        pert_table = []
        for t in pert_summary.get("tests", []):
            m_eval = t.get("model_evaluations", {})
            lr_eval = m_eval.get("logistic_regression", {})
            rule_eval = m_eval.get("rule_based", {})
            svm_eval = m_eval.get("linear_svm", {})

            pert_table.append({
                "Test ID": t["test_id"],
                "Perturbation Name": t["perturbation_name"],
                "Base Type": t["base_type"],
                "Expected": t["expected_intent"],
                "LR Drift": f"{lr_eval.get('probability_drift', 0):+.3f}",
                "SVM Drift": f"{svm_eval.get('probability_drift', 0):+.3f}",
                "Rule Drift": f"{rule_eval.get('probability_drift', 0):+.3f}",
                "LR Flipped?": "⚠️ Yes" if lr_eval.get("label_flipped") else "No",
                "SVM Flipped?": "⚠️ Yes" if svm_eval.get("label_flipped") else "No",
            })
        
        st.dataframe(pd.DataFrame(pert_table), width="stretch")

        with st.expander("🔍 Inspect Full Perturbation Texts and Predictions"):
            for t in pert_summary.get("tests", []):
                st.markdown(f"**{t['test_id']}: {t['perturbation_name']}**")
                st.text(f"Original:  {t['original_text']}")
                st.text(f"Perturbed: {t['perturbed_text']}")
                st.write(t.get("model_evaluations", {}))
                st.divider()

        st.markdown("#### 💡 Empirical Insights from Perturbation Study")
        for c in pert_summary.get("conclusions", []):
            st.write(f"• {c}")

    else:
        st.warning("Perturbation study results pending. Run `python scripts/train_and_evaluate.py`.")

    st.markdown("### 🔎 Representative Model Error Analysis")
    if metrics and "error_analysis" in metrics:
        errors = metrics["error_analysis"]
        st.caption(f"Inspection of {len(errors)} representative holdout prediction errors across the models:")
        
        for err in errors:
            with st.container():
                st.markdown(f"**Message ID: `{err.get('message_id')}` ({err.get('channel').upper()}) — Campaign: `{err.get('campaign_group')}`**")
                st.markdown(f"*Snippet*: `{err.get('body_snippet')}`")
                
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.write(f"**Actual**: `{err.get('actual_label')}`")
                c_b.write(f"**Rule**: `{err.get('rule_prediction')}`")
                c_c.write(f"**LogReg**: `{err.get('lr_prediction')}`")
                c_d.write(f"**SVM**: `{err.get('svm_prediction')}`")

                st.markdown(f"• **Error Type**: {err.get('error_type')}")
                st.markdown(f"• **Root Cause**: {err.get('root_cause_analysis')}")
                st.markdown(f"• **Recommended Mitigation**: {err.get('recommended_mitigation')}")
                st.divider()
    else:
        st.info("No error analysis records available.")
