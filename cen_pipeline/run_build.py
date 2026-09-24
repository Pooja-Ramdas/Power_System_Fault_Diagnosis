"""
Launch build_dataset.py as a detached process and then regenerate the manifest.
This script itself runs in a subprocess detached from the agent session,
so the build survives agent termination.

Usage: python run_build.py
"""
import os
import sys
import subprocess
import time
from datetime import datetime

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PIPELINE_DIR, "..", ".venv", "Scripts", "python.exe")
BUILD_SCRIPT = os.path.join(PIPELINE_DIR, "build_dataset.py")
MANIFEST_SCRIPT = os.path.join(PIPELINE_DIR, "regenerate_manifest.py")
LOG_FILE = os.path.join(PIPELINE_DIR, "build_log.txt")


def main():
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-4]
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 46}\n")
        log.write(f"Build started at {timestamp}\n")
        log.write(f"{'=' * 46}\n")
        log.flush()

    print(f"[run_build] Starting build at {timestamp}")
    print(f"[run_build] Log: {LOG_FILE}")
    print(f"[run_build] Python: {VENV_PYTHON}")

    # Run build_dataset.py with unbuffered output, appending to log
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        result = subprocess.run(
            [VENV_PYTHON, "-u", BUILD_SCRIPT],
            cwd=PIPELINE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    end_timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-4]
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 46}\n")
        log.write(f"Build finished at {end_timestamp} (exit code {result.returncode})\n")
        log.write(f"{'=' * 46}\n")
        log.flush()

    print(f"[run_build] Build finished with exit code {result.returncode}")

    # Regenerate manifest from all meta.json files
    print("[run_build] Regenerating complete manifest.csv ...")
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write("\n[run_build] Regenerating complete manifest.csv ...\n")
        result2 = subprocess.run(
            [VENV_PYTHON, "-u", MANIFEST_SCRIPT],
            cwd=PIPELINE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[run_build] Manifest regeneration exit code: {result2.returncode}\n")
        log.write(f"[run_build] ALL DONE.\n")

    print("[run_build] ALL DONE.")


if __name__ == "__main__":
    main()
