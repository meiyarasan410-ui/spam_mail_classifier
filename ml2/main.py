"""
PhishShift Entrypoint Launcher for VS Code & Local Environments.
Runs Streamlit Dashboard or Flask API depending on configuration.
"""

import sys
import os
import subprocess

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    mode = sys.argv[1] if len(sys.argv) > 1 else "streamlit"

    if mode in ["--flask", "--api", "flask", "api"]:
        print("[*] Launching PhishShift Flask Web Application on http://localhost:5000 ...")
        from api.index import app
        app.run(host="127.0.0.1", port=5000, debug=True)
    else:
        print("[*] Launching PhishShift Streamlit Cybersecurity Dashboard on http://localhost:8501 ...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
