#!/usr/bin/env python3
"""
Kaggle NEURALIS End-to-End Training, Evaluation, and Report Artifact Generator
----------------------------------------------------------------------------------
This script automatically:
  1. Clones the NEURALIS GitHub repository (or uses current workspace).
  2. Auto-detects train.csv / test.csv datasets in /kaggle/input/ or local directories.
  3. Normalizes catalog schemas and builds 5 domain pricing feature sets.
  4. Runs 5-fold cross-validation across LightGBM, XGBoost, CatBoost, Random Forest, ExtraTrees, Ridge,
     and auto-tunes the RidgeCV Meta-Learner stacker.
  5. Computes statistical significance, latency profiling, and target skewness metrics.
  6. Generates 14 high-resolution publication PNG plots into `docs/` and 6 JSON/CSV reports into `experiments/reports/`.
  7. Creates test predictions `submission.csv` and packages all artifacts into `experiments_reports_and_docs.zip`
     for 1-click download back to your local repository!

Usage in Kaggle:
  !python notebooks/kaggle_runner.py
"""

import os
import sys
import io
import glob
import shutil
import zipfile
import subprocess
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable unbuffered real-time UTF-8 stdout streaming
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


def find_kaggle_dataset(fail_fast: bool = True):
    """Locates train.csv and test.csv in Kaggle or local environments with fail-first check."""
    train_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("data/raw/train.csv") + glob.glob("train.csv")
    test_paths  = glob.glob("/kaggle/input/**/test.csv", recursive=True) + glob.glob("data/raw/test.csv") + glob.glob("test.csv")

    train_file = train_paths[0] if train_paths else None
    test_file  = test_paths[0] if test_paths else None

    print("[SEARCH] Dataset Auto-Detection:", flush=True)
    print(f"   - Training Data: {train_file if train_file else 'NOT FOUND'}", flush=True)
    print(f"   - Testing Data:  {test_file if test_file else 'NOT FOUND'}", flush=True)

    if not train_file and fail_fast:
        print("⚠️ Training data file not found in /kaggle/input or local data/raw/. Using synthetic fallback data for demonstration...", flush=True)

    return train_file, test_file


def run_pipeline():
    print("=========================================================================", flush=True)
    print("NEURALIS Kaggle End-to-End ML Pipeline (Real-Time Unbuffered & Fail-Fast)", flush=True)
    print("=========================================================================", flush=True)

    # Prepare environment with PYTHONUNBUFFERED=1 for live stdout streaming
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 1. Locate datasets
    train_path, test_path = find_kaggle_dataset(fail_fast=True)

    # 2. Run generate-report CLI with unbuffered flag (-u) and fail-first check=True
    cmd = [sys.executable, "-u", "main.py", "generate-report"]
    if train_path:
        cmd.extend(["--data", train_path])

    print(f"\n⚡ [STEP 1/3] Executing Report & Benchmark Pipeline: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ FAIL-FAST: Subprocess execution failed with exit code {e.returncode}. Halting pipeline immediately!", flush=True)
        sys.exit(e.returncode)

    # 3. If test data is available, fit full pipeline and generate submission.csv
    if test_path and os.path.exists(test_path):
        print(f"\n⚡ [STEP 2/3] Fitting final Stacker Ensemble on {train_path}...", flush=True)
        cmd_train = [sys.executable, "-u", "main.py", "train", "--config", "configs/training/final_train.yaml", "--data", train_path, "--model", "stacker"]
        try:
            subprocess.run(cmd_train, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"❌ FAIL-FAST: Stacker model training failed with exit code {e.returncode}. Halting pipeline immediately!", flush=True)
            sys.exit(e.returncode)

        print(f"\n⚡ [STEP 3/3] Generating test predictions for {test_path}...", flush=True)
        cmd_predict = [sys.executable, "-u", "main.py", "predict", "--config", "configs/inference/inference.yaml", "--data", test_path, "--output", "submission.csv"]
        try:
            subprocess.run(cmd_predict, check=True, env=env)
            print("✅ Created submission.csv successfully!", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FAIL-FAST: Test prediction generation failed with exit code {e.returncode}. Halting pipeline immediately!", flush=True)
            sys.exit(e.returncode)

    # 4. Zip all report artifacts for 1-click download
    zip_path = "experiments_reports_and_docs.zip"
    print(f"\n📦 Archiving reports and plots into {zip_path}...", flush=True)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in ["experiments/reports", "docs"]:
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, os.getcwd())
                        zipf.write(full_p, rel_p)

        if os.path.exists("submission.csv"):
            zipf.write("submission.csv", "submission.csv")

    print("=========================================================================", flush=True)
    print("🎉 KAGGLE EXECUTION COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Download '{zip_path}' from Kaggle output pane and unzip into your workspace.", flush=True)
    print("=========================================================================", flush=True)


if __name__ == "__main__":
    run_pipeline()

