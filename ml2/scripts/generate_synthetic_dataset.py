"""
Synthetic Demonstration Dataset Generator for PhishShift (ML-07).
Generates 420+ realistic, ethically sanitized email and SMS cybersecurity examples
covering subtle phishing lures, legitimate urgent notices, password resets, payroll,
and intentional near-duplicate template clusters for deduplication evaluation.
"""

from __future__ import annotations
import csv
import os
import random

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_messages.csv")


def generate_messages() -> list[dict]:
    random.seed(42)
    messages = []
    msg_id = 1

    # =========================================================================
    # PHISHING CAMPAIGNS (Target: ~210 messages)
    # =========================================================================

    # Campaign 1: Urgent Account Suspension (High urgency, threats, defanged links)
    suspension_templates = [
        ("FINAL NOTICE: Your {service} account is scheduled for immediate suspension within {hours} hours due to unverified billing details. Restore access now: {url}",
         "Account Suspension Notice", "contains_defanged_url;brand_spoof", "lookalike_spoofed", "email"),
        ("CRITICAL ALERT: Unauthorized access detected from IP {ip}. Your {service} profile has been temporarily restricted. Verify identity at {url} to prevent termination.",
         "Security Alert: Unauthorized Access", "contains_defanged_url;ip_based_url", "lookalike_spoofed", "email"),
        ("Your {service} mailbox has exceeded storage capacity ({pct}%). Inbound messages are held. Expand quota immediately at {url} or messages will be permanently deleted.",
         "Mailbox Quota Exceeded - Action Required", "contains_defanged_url", "lookalike_spoofed", "email"),
    ]
    services = ["Microsoft 365", "Google Workspace", "Okta SSO", "AWS Console", "Workday Portal", "Dropbox Enterprise"]
    urls_defanged = [
        "hxxps://secure-auth[.]portal-verify[.]net/login?token=9284",
        "hxxps://login[.]microsoft-support-team[.]com/verify",
        "hxxp://198[.]51[.]100[.]42/auth/session-renew",
        "hxxps://okta-auth[.]cloud-identity-check[.]org/sso",
        "hxxps://account-update[.]helpdesk-portal[.]info/resolve"
    ]

    for i in range(35):
        tmpl, subj, url_f, s_type, chan = suspension_templates[i % len(suspension_templates)]
        body = tmpl.format(
            service=random.choice(services),
            hours=random.choice([12, 24, 48]),
            ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            pct=random.randint(97, 99),
            url=random.choice(urls_defanged)
        )
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "phishing",
            "campaign_group": "phish_urgent_suspension_v1",
            "collected_date": f"2026-02-{random.randint(1, 28):02d}"
        })
        msg_id += 1

    # Campaign 2: Payroll Direct Deposit Spoof (Payroll lure with urgent credential capture)
    payroll_phish_templates = [
        ("ATTENTION: An error occurred during your bi-weekly direct deposit calculation. Update your banking routing credentials at {url} before 5:00 PM today to avoid delayed payroll funds.",
         "Urgent: Direct Deposit Discrepancy", "contains_defanged_url;credential_solicitation", "lookalike_spoofed", "email"),
        ("Year-end tax withholding statement (W-2/1099) amendment required. View your revised compensation breakdown and confirm SSN and password at {url}.",
         "Payroll Compliance: Tax Form Update", "contains_defanged_url", "external_freemail", "email"),
        ("Executive Compensation Advisory: Your bonus allocation adjustment has been finalized. Sign in with organizational credentials to approve payment distribution: {url}",
         "Payroll Notice: Executive Bonus Approval", "contains_defanged_url", "lookalike_spoofed", "email"),
    ]
    for i in range(30):
        tmpl, subj, url_f, s_type, chan = payroll_phish_templates[i % len(payroll_phish_templates)]
        body = tmpl.format(url=random.choice(urls_defanged))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "phishing",
            "campaign_group": "phish_payroll_directdeposit_v2",
            "collected_date": f"2026-03-{random.randint(1, 30):02d}"
        })
        msg_id += 1

    # Campaign 3: Password Reset Spoof (Spoofed password change with fake cancel link)
    pwd_phish_templates = [
        ("Someone recently requested a password reset for your corporate account from Kiev, Ukraine. If this was NOT you, click immediately to cancel the request and secure your credentials: {url}",
         "Unauthorized Password Reset Alert", "contains_defanged_url;brand_spoof", "lookalike_spoofed", "email"),
        ("Security token expired: Your SSO single-sign-on session password will expire in 12 hours. Renew existing password and security questions here: {url}",
         "Password Expiration Warning", "contains_defanged_url", "lookalike_spoofed", "email"),
    ]
    for i in range(25):
        tmpl, subj, url_f, s_type, chan = pwd_phish_templates[i % len(pwd_phish_templates)]
        body = tmpl.format(url=random.choice(urls_defanged))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "phishing",
            "campaign_group": "phish_password_reset_spoof_v3",
            "collected_date": f"2026-03-{random.randint(1, 30):02d}"
        })
        msg_id += 1

    # Campaign 4: Unfamiliar Wording / Subtle Phishing (Holdout Candidate - Unfamiliar phrasing)
    subtle_phish_templates = [
        ("Colleague collaboration invitation: A shared cloud document repository requires organizational clearance prior to synchronization. Please review the collaborative manifest: {url}",
         "Asset Collaboration Request", "contains_defanged_url;subtle_urgency", "external_freemail", "email"),
        ("Compliance archival survey: To preserve your electronic communication retention privileges, execute validation sequence via our delegated external portal: {url}",
         "Electronic Communication Record Retention", "contains_defanged_url", "lookalike_spoofed", "email"),
        ("Vendor contract amendment: We noticed anomalous parameters regarding milestone {num}. Kindly verify the digital signature ledger at {url} before scheduled dispatch.",
         "Contract Milestone Verification", "contains_defanged_url", "external_freemail", "email"),
        ("Corporate travel reconciliation: Discrepancies noted in itinerary ref #{num}. Please substantiate your corporate authorization token: {url}",
         "Travel Authorization Reconciliation", "contains_defanged_url", "lookalike_spoofed", "email"),
    ]
    for i in range(30):
        tmpl, subj, url_f, s_type, chan = subtle_phish_templates[i % len(subtle_phish_templates)]
        body = tmpl.format(num=random.randint(10000, 99999), url=random.choice(urls_defanged))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "phishing",
            "campaign_group": "phish_unfamiliar_wording_cloud_v4",
            "collected_date": f"2026-04-{random.randint(1, 25):02d}"
        })
        msg_id += 1

    # Campaign 5: SMS Delivery Smishing (Package detained, customs fee)
    sms_shorteners = [
        "hxxps://bit[.]ly/usps-track-938",
        "hxxps://tinyurl[.]com/dhl-customs-fee",
        "hxxps://t[.]co/fedex-pkg-hold",
        "hxxps://rb[.]gy/package-redelivery-601"
    ]
    sms_package_templates = [
        "USPS Alert: Package tracking #US{num} cannot be delivered due to missing apartment information. Confirm delivery details within 12h: {url}",
        "DHL Express: Your shipment #DH{num} has an unpaid customs clearance fee ($2.85). Pay online to avoid parcel return: {url}",
        "FedEx Notification: Parcel arrived at local distribution center but requires signature verification. Reschedule at {url}",
    ]
    for i in range(35):
        tmpl = sms_package_templates[i % len(sms_package_templates)]
        body = tmpl.format(num=random.randint(100000, 999999), url=random.choice(sms_shorteners))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": "sms",
            "subject_or_title": "",
            "body_text": body,
            "sanitized_url_features": "contains_defanged_url;contains_shortener",
            "sender_domain_type": "defanged_shortlink",
            "label": "phishing",
            "campaign_group": "phish_sms_package_delivery_v5",
            "collected_date": f"2026-04-{random.randint(1, 28):02d}"
        })
        msg_id += 1

    # Campaign 6: SMS Bank Alert Smishing (Fraud alert lure)
    sms_bank_templates = [
        "Chase Alert: Suspicious debit charge of ${amt}.00 at TARGET STORES detected. If this was not you, lock your card immediately at: {url}",
        "Wells Fargo Fraud Dept: Unrecognized wire transfer of ${amt}.00 initiated. Reply NO to decline or verify credentials at {url}",
        "Bank of America Notice: Temporary lock placed on online banking due to anomalous sign-in. Unlock securely via {url}",
    ]
    for i in range(30):
        tmpl = sms_bank_templates[i % len(sms_bank_templates)]
        body = tmpl.format(amt=random.randint(450, 1850), url=random.choice(sms_shorteners))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": "sms",
            "subject_or_title": "",
            "body_text": body,
            "sanitized_url_features": "contains_defanged_url;contains_shortener",
            "sender_domain_type": "defanged_shortlink",
            "label": "phishing",
            "campaign_group": "phish_sms_bank_fraud_v6",
            "collected_date": f"2026-05-{random.randint(1, 20):02d}"
        })
        msg_id += 1

    # Campaign 7: Executive Impersonation / CEO Wire Fraud
    exec_templates = [
        ("Are you at your desk right now? I need you to process an urgent wire transfer for an acquisition closing today. Keep this strictly confidential until press release.",
         "Confidential Request", "none", "external_freemail", "email"),
        ("I am stuck in an executive board meeting and cannot take calls. Please purchase {qty} Apple / Steam gift cards ($100 each) for the client presentation and send me photos of codes.",
         "Quick Favor for Board Meeting", "none", "external_freemail", "email"),
    ]
    for i in range(25):
        tmpl, subj, url_f, s_type, chan = exec_templates[i % len(exec_templates)]
        body = tmpl.format(qty=random.choice([5, 10, 15]))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "phishing",
            "campaign_group": "phish_exec_impersonation_v7",
            "collected_date": f"2026-05-{random.randint(1, 28):02d}"
        })
        msg_id += 1

    # =========================================================================
    # LEGITIMATE CAMPAIGNS (Target: ~210 messages)
    # =========================================================================

    # Campaign 8: Legitimate Password Resets (Explicitly mentions password, token, no coercion)
    legit_pwd_templates = [
        ("We received a request to reset the password for your internal account. Use the following button to reset it. If you did not make this request, you can safely ignore this email. We will never ask you for your password.",
         "Reset your password", "none", "trusted_internal", "email"),
        ("Your temporary password reset code is: {token}. This code expires in 15 minutes. If you did not request this change, please report this incident to IT Security at extension 4400.",
         "Password Reset Verification Code", "none", "trusted_internal", "email"),
        ("Password changed successfully for user ID #{uid}. If you made this modification, no further action is required. Review recent logins on the intranet security dashboard.",
         "Password change confirmation", "none", "trusted_internal", "email"),
    ]
    for i in range(35):
        tmpl, subj, url_f, s_type, chan = legit_pwd_templates[i % len(legit_pwd_templates)]
        body = tmpl.format(token=random.randint(100000, 999999), uid=random.randint(1000, 9999))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "legitimate",
            "campaign_group": "legit_pwd_reset_routine",
            "collected_date": f"2026-01-{random.randint(1, 31):02d}"
        })
        msg_id += 1

    # Campaign 9: Legitimate Payroll and Account Statements (Mentions payroll, tax, scheduled)
    legit_payroll_templates = [
        ("Your monthly payroll statement for pay period ending {date} is now available for review in Workday. Please log into the company portal to view deductions and withholdings.",
         "Your Pay Statement is Ready", "none", "trusted_internal", "email"),
        ("Benefits Open Enrollment begins next Monday. Review medical, dental, and retirement plan contribution limits on the PeopleOps internal knowledge base. No changes required if satisfied.",
         "Annual Benefits Open Enrollment Notice", "none", "trusted_internal", "email"),
        ("Form W-2 wage and tax statements for tax year 2025 are available electronically on the ADP employee portal. Access via company single-sign-on.",
         "Form W-2 Tax Statement Availability", "none", "trusted_internal", "email"),
    ]
    for i in range(35):
        tmpl, subj, url_f, s_type, chan = legit_payroll_templates[i % len(legit_payroll_templates)]
        body = tmpl.format(date=f"2026-02-{random.choice([15, 28])}")
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "legitimate",
            "campaign_group": "legit_payroll_regular",
            "collected_date": f"2026-02-{random.randint(1, 28):02d}"
        })
        msg_id += 1

    # Campaign 10: Legitimate Urgent Business Notices (Mentions urgent, P1, priority, but legitimate)
    legit_urgent_templates = [
        ("URGENT P1 INCIDENT #INC-{num}: Payment API latency spike observed across EU clusters. SRE incident response bridge active on Slack #incident-p1. No end-user credential action required.",
         "URGENT: Incident INC-{} - Production Latency".format(random.randint(1000, 9999)), "none", "trusted_internal", "email"),
        ("Urgent notice: Scheduled database failover maintenance will take place tonight at 02:00 UTC. Secondary replicas will promote automatically. Expected downtime under 2 minutes.",
         "Urgent: Scheduled Maintenance Tonight", "none", "trusted_internal", "email"),
        ("Priority 1 Alert: Build pipeline blocked due to failing integration test in repository core-service. All engineers please hold master merges until resolved by DevOps.",
         "P1 Alert: CI/CD Build Pipeline Stalled", "none", "trusted_internal", "email"),
        ("URGENT: Building evacuation drill scheduled for today at 14:00. Please familiarize yourself with emergency stairwells and designated outdoor muster points.",
         "Urgent Facility Notice: Fire Evacuation Drill", "none", "trusted_internal", "email"),
    ]
    for i in range(35):
        tmpl, subj, url_f, s_type, chan = legit_urgent_templates[i % len(legit_urgent_templates)]
        body = tmpl.format(num=random.randint(1000, 9999))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "legitimate",
            "campaign_group": "legit_urgent_incident",
            "collected_date": f"2026-03-{random.randint(1, 30):02d}"
        })
        msg_id += 1

    # Campaign 11: Legitimate SMS 2FA Codes (Telecom shortcode, OTP, no links)
    legit_sms_templates = [
        "{service}: Your security verification code is {code}. It expires in 10 minutes. Never share this code with anyone.",
        "Your {service} verification code: {code}. If you did not initiate this request, contact account security.",
        "{code} is your one-time authentication passcode for {service}. Valid for 5 minutes.",
    ]
    sms_services = ["Google", "GitHub", "Microsoft", "Bank of America", "Uber", "Slack", "Amazon"]
    for i in range(40):
        tmpl = legit_sms_templates[i % len(legit_sms_templates)]
        body = tmpl.format(
            service=random.choice(sms_services),
            code=random.randint(100000, 999999)
        )
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": "sms",
            "subject_or_title": "",
            "body_text": body,
            "sanitized_url_features": "none",
            "sender_domain_type": "telecom_shortcode",
            "label": "legitimate",
            "campaign_group": "legit_sms_2fa_codes",
            "collected_date": f"2026-04-{random.randint(1, 28):02d}"
        })
        msg_id += 1

    # Campaign 12: Legitimate Team Collaboration (Jira, GitHub, Calendar)
    collab_templates = [
        ("@{user} mentioned you in pull request #{pr}: 'Could you check the database migration script before release?' Review changes in GitHub.",
         "GitHub: Mention in PR #{}".format(random.randint(100, 999)), "none", "trusted_internal", "email"),
        ("New ticket assigned to you: [PROJ-{num}] Optimize Redis caching for session tokens. Due date: Friday. View on Jira.",
         "Jira: Issue Assigned to You", "none", "trusted_internal", "email"),
        ("Invitation: Architecture Review Sync - Tomorrow at 10:00 AM. Location: Google Meet room 'arch-weekly'. Agenda attached.",
         "Calendar: Architecture Review Sync", "none", "trusted_internal", "email"),
    ]
    users = ["alice", "bob", "carol", "dave", "elena", "frank"]
    for i in range(35):
        tmpl, subj, url_f, s_type, chan = collab_templates[i % len(collab_templates)]
        body = tmpl.format(user=random.choice(users), pr=random.randint(100, 999), num=random.randint(1000, 4999))
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": body,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "legitimate",
            "campaign_group": "legit_team_collaboration",
            "collected_date": f"2026-05-{random.randint(1, 25):02d}"
        })
        msg_id += 1

    # Campaign 13: Legitimate IT Announcements
    it_announce_templates = [
        ("IT Operations Notice: Scheduled VPN gateway firmware upgrades will occur this Saturday from 22:00 to 23:00 EST. Remote users may experience brief disconnections.",
         "IT Notification: Weekend VPN Maintenance", "none", "trusted_internal", "email"),
        ("Laptop hardware refresh eligibility: Employees who joined prior to Q1 2024 are eligible for standard hardware replacement. Order through the internal IT service desk portal.",
         "IT Hardware Refresh Program 2026", "none", "trusted_internal", "email"),
    ]
    for i in range(30):
        tmpl, subj, url_f, s_type, chan = it_announce_templates[i % len(it_announce_templates)]
        messages.append({
            "message_id": f"MSG_{msg_id:04d}",
            "channel": chan,
            "subject_or_title": subj,
            "body_text": tmpl,
            "sanitized_url_features": url_f,
            "sender_domain_type": s_type,
            "label": "legitimate",
            "campaign_group": "legit_it_announcements",
            "collected_date": f"2026-05-{random.randint(1, 30):02d}"
        })
        msg_id += 1

    return messages


def save_dataset_csv(messages: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(messages)
    print(f"Generated {len(messages)} synthetic messages saved to: {path}")


if __name__ == "__main__":
    msgs = generate_messages()
    save_dataset_csv(msgs, OUTPUT_PATH)
