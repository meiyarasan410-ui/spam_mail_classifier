# PhishShift: 3-to-5 Minute Demonstration Plan

This walkthrough script guides evaluators, educators, and security analysts through the key features of the PhishShift Message Risk Scanner.

---

## ⏱️ Step-by-Step Walkthrough (Estimated Duration: 4 Minutes)

### Minute 1: Executive Dashboard & Dataset Overview
1. **Launch**: Start the application with `streamlit run app.py` and open `http://localhost:8501`.
2. **Dashboard Review**:
   - Highlight the **420 total messages** distributed evenly across **13 campaign groups** (210 phishing, 210 legitimate; 65.5% email, 34.5% SMS).
   - Point out the active model status badges and latest evaluation metrics.
   - Emphasize the prominent **Cybersecurity Triage Disclaimer**: *PhishShift provides decision-support triage, not a guarantee of safety.*

### Minute 2: Interactive Message Risk Scanner & Quick Scenarios
1. **Navigate**: Click on `🔍 Message Scanner` in the sidebar.
2. **Scenario A — Blatant Phishing**:
   - Click the preset button: `🚨 Phishing: Account Suspension`.
   - Click `🛡️ Scan Message Risk`.
   - **Observe**: Crimson `Potential Phishing` alert badge (>90% risk score).
   - Show the detected suspicious cues: coercive threat, credential prompt, defanged URL (`hxxps://auth[.]secure-portal[.]net`).
   - Review the actionable next steps (do not click links, report to internal SOC).
3. **Scenario B — Legitimate Urgent Incident (FR-05)**:
   - Click the preset button: `⚡ Legit: Urgent P1 Incident`.
   - Click `🛡️ Scan Message Risk`.
   - **Observe**: Even though the message contains words like *"URGENT"*, *"P1 INCIDENT"*, and *"IMMEDIATELY"*, the scanner classifies it as `Likely Legitimate`.
   - Show the countervailing legitimate signals: absence of credential solicitation, standard incident ticketing structure, internal sender context.

### Minute 3: Uncertainty Abstention (FR-04) & Smishing
1. **Scenario C — Smishing Detection**:
   - Click the preset button: `📱 Phishing: SMS Smishing`.
   - **Observe**: Identifies SMS shortener (`bit[.]ly`), package delivery lure, and suggests carrier reporting (forward to 7726).
2. **Scenario D — Uncertainty Rejection Policy**:
   - Demonstrate the sidebar uncertainty sliders (`[0.40, 0.65]`).
   - Enter an ambiguous or subtly phrased message.
   - **Observe**: The amber `⚠️ Uncertain — Manual Review Recommended` badge triggers, routing the message to human analysts rather than making a dangerous binary guess.

### Minute 4: Model Comparison & Generalization (FR-02, FR-03)
1. **Navigate**: Switch to `⚖️ Model Comparison`.
2. **Review Comparison Table**:
   - Compare **Rule-Based Baseline** vs **TF-IDF Logistic Regression** vs **Calibrated Linear SVM**.
   - Show that on the campaign-separated holdout set, the Rule-Based detector achieved **0.0% False Positive Rate on urgent messages**, whereas standard linear ML models exhibited higher false alarm rates.
   - Inspect the side-by-side **Confusion Matrices** and benchmark chart.

### Minute 5: Deduplication Proof (FR-01) & Perturbation Lab (FR-05)
1. **Navigate**: Switch to `📁 Dataset & Deduplication`.
   - Show the FR-01 deduplication statistics: 180 exact duplicates and 331 near-duplicate template instances clustered and removed.
2. **Navigate**: Switch to `🧪 Robustness & Perturbation Lab`.
   - Review the 8 controlled harmless perturbation tests (bracket defanging, punctuation intensity, spaced letters).
   - Review the representative model error analysis records detailing root causes and mitigations.
