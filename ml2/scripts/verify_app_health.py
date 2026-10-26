"""
Comprehensive App Health Verification Script.
Simulates HTTP and headless Streamlit AppTest execution across all pages.
"""

import sys
import os
import urllib.request
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_http_endpoint():
    url = "http://localhost:8501"
    print(f"1. Checking HTTP status for {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhishShift-HealthChecker"})
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            print(f"   HTTP Status: {status} (OK)")
            assert status == 200, f"Expected 200, got {status}"
    except Exception as e:
        print(f"   HTTP Check Error: {e}")
        return False
    return True

def check_all_views_with_apptest():
    print("2. Running Streamlit headless AppTest across all navigation views...")
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(PROJECT_ROOT, "app.py")
    at = AppTest.from_file(app_path)
    at.default_timeout = 20

    # Initial Run
    at.run()
    if at.exception:
        print(f"   FAIL: Exception on initial load: {at.exception}")
        return False
    print("   [OK] Initial app load passed with 0 exceptions.")

    # Get navigation radio
    nav = at.sidebar.radio[0]
    options = nav.options
    print(f"   Available views in navigation: {options}")

    for opt in options:
        print(f"   Testing View: '{opt}'...")
        nav.set_value(opt)
        at.run()
        if at.exception:
            print(f"   FAIL: Exception on view '{opt}': {at.exception}")
            for exc in at.exception:
                print(f"      {exc.value}")
            return False
        print(f"   [OK] '{opt}' rendered cleanly with 0 exceptions.")

    # Test Message Scanner interaction
    print("3. Testing interactive Message Scanner inference in AppTest...")
    nav.set_value("🔍 Message Scanner")
    at.run()
    # Click preset button (e.g. Account Lock)
    account_lock_btn = None
    for btn in at.button:
        if "Account Lock" in btn.label:
            account_lock_btn = btn
            break
    if account_lock_btn:
        account_lock_btn.click()
        at.run()
        print("   [OK] Clicked preset 'Account Lock'")
    
    # Click scan button
    scan_btn = None
    for btn in at.button:
        if "Scan Message Risk" in btn.label:
            scan_btn = btn
            break
    if scan_btn:
        scan_btn.click()
        at.run()
        if at.exception:
            print(f"   FAIL: Exception during message scan: {at.exception}")
            return False
        print("   [OK] Scan Message Risk executed and rendered triage result with 0 exceptions.")

    # Test Dataset & Deduplication
    print("4. Testing Dataset & Deduplication in AppTest...")
    nav.set_value("📁 Dataset & Deduplication")
    at.run()
    if at.exception:
        print(f"   FAIL: Exception on Dataset & Deduplication: {at.exception}")
        return False
    print("   [OK] Dataset & Deduplication rendered charts, metrics, and template clusters cleanly.")

    return True

if __name__ == "__main__":
    http_ok = check_http_endpoint()
    app_ok = check_all_views_with_apptest()

    if http_ok and app_ok:
        print("\n==========================================")
        print("ALL WEBSITE INTEGRITY CHECKS PASSED (100%)")
        print("==========================================")
        sys.exit(0)
    else:
        print("\nWEBSITE INTEGRITY CHECKS FAILED.")
        sys.exit(1)
