# Dataset Provenance Sheet: PhishShift (ML-07)

## 1. Dataset Overview

- **Dataset Identifier**: `phishshift-synthetic-messages-v1`
- **Location**: `data/synthetic_messages.csv`
- **Total Record Count**: 420 messages
- **Collection / Generation Date**: 2026-01-01 through 2026-05-30
- **Primary Channels**: Email (65.5%), SMS (34.5%)
- **Class Balance**: 210 Phishing (50.0%), 210 Legitimate (50.0%)
- **Number of Independent Campaign Groups**: 13

## 2. Ethical Sourcing & Synthetic Creation Methodology

In strict compliance with educational cybersecurity standards, privacy preservation regulations, and the ML-07 project specifications:
- **No Private or Confidential Real-World Communications**: The dataset contains zero proprietary corporate emails, real PII (Personally Identifiable Information), real passwords, or authentic telephone numbers.
- **Controlled Synthetic Authoring**: All 420 examples were synthetically authored based on real-world threat intelligence attack taxonomies documented in APWG (Anti-Phishing Working Group) and CISA alerts.
- **Defanged Links Only**: All URLs, hostnames, and IP addresses are systematically defanged (`hxxps://`, `[.]`, `bit[.]ly`) to prevent accidental browser navigation, web scraping, or malware execution.
- **Intentional Template Clustering**: Designed with intentional parameter variations (names, ticket IDs, amounts, dates) across 13 distinct campaigns to enable empirical validation of **FR-01** (Template Deduplication) and **FR-03** (Campaign-Separated Holdout Evaluation).

## 3. Schema Definition & Field Annotations

| Field Name | Type | Description | Allowed Values / Format |
| :--- | :--- | :--- | :--- |
| `message_id` | String | Unique record identifier | `MSG_0001` - `MSG_0420` |
| `channel` | String | Transmission medium | `email`, `sms` |
| `subject_or_title` | String | Message subject line or notification title | Free text (may be empty for SMS) |
| `body_text` | String | Primary message content body | Free text (defanged indicators preserved) |
| `sanitized_url_features`| String | Defanged structural link indicators | `none`, `contains_defanged_url`, `contains_shortener`, `contains_ip_url` |
| `sender_domain_type` | String | Categorized origin of the sender | `trusted_internal`, `external_freemail`, `lookalike_spoofed`, `defanged_shortlink`, `telecom_shortcode`, `unknown` |
| `label` | String | Ground-truth triage category | `phishing`, `legitimate` |
| `campaign_group` | String | Campaign identifier cluster | 13 unique campaign identifiers |
| `collected_date` | String | ISO-8601 synthetic generation timestamp | `YYYY-MM-DD` |

## 4. Campaign Distribution Breakdown

### Phishing Campaigns (7 Groups, 210 Messages)
1. `phish_urgent_suspension_v1` (35 msgs): Blatant account suspension threats and coercive credential prompts.
2. `phish_payroll_directdeposit_v2` (30 msgs): Spoofed payroll direct deposit updates and W-2 amendments.
3. `phish_password_reset_spoof_v3` (25 msgs): Spoofed unauthorized password reset warning lures.
4. `phish_unfamiliar_wording_cloud_v4` (30 msgs): Subtle collaborative workspace asset approval requests with unfamiliar phrasing (**Holdout Test Campaign**).
5. `phish_sms_package_delivery_v5` (35 msgs): SMS package delivery smishing with customs payment links.
6. `phish_sms_bank_fraud_v6` (30 msgs): SMS fraudulent bank alert lures with shorteners (**Holdout Test Campaign**).
7. `phish_exec_impersonation_v7` (25 msgs): CEO/executive wire transfer and gift card purchase requests.

### Legitimate Campaigns (6 Groups, 210 Messages)
8. `legit_pwd_reset_routine` (35 msgs): Automated password resets with routine non-coercive safety disclaimers (**Holdout Test Campaign**).
9. `legit_payroll_regular` (35 msgs): Scheduled monthly pay stubs and annual benefits open enrollment.
10. `legit_urgent_incident` (35 msgs): Legitimate high-priority production P1 outages and maintenance notices (**Holdout Test Campaign**).
11. `legit_sms_2fa_codes` (40 msgs): Routine SMS OTP authentication codes without URLs.
12. `legit_team_collaboration` (35 msgs): GitHub PR review mentions, Jira issues, Google Meet calendar invites.
13. `legit_it_announcements` (30 msgs): Corporate VPN maintenance and hardware refresh announcements.

## 5. Licensing and Usage Constraints

- **License**: CC-BY-4.0 (Creative Commons Attribution 4.0 International)
- **Permitted Use**: Academic research, educational triage training, defensive ML benchmarking.
- **Prohibited Use**: Malicious phishing payload generation, adversarial evasion benchmarking for offensive bypass.
