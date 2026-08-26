#!/usr/bin/env python3
"""
Kaggle PrismPrice End-to-End Training, Evaluation, and Report Artifact Generator
----------------------------------------------------------------------------------
This script automatically:
  1. Clones the A_ML_25 GitHub repository (or uses current workspace).
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
import glob
import shutil
import zipfile
import subprocess
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_kaggle_dataset():
    """Locates train.csv and test.csv in Kaggle or local environments."""
    train_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("data/raw/train.csv") + glob.glob("train.csv")
    test_paths  = glob.glob("/kaggle/input/**/test.csv", recursive=True) + glob.glob("data/raw/test.csv") + glob.glob("test.csv")

    train_file = train_paths[0] if train_paths else None
    test_file  = test_paths[0] if test_paths else None

    print(f"[SEARCH] Dataset Auto-Detection:")
    print(f"   - Training Data: {train_file if train_file else 'NOT FOUND (Using Synthetic Fallback)'}")
    print(f"   - Testing Data:  {test_file if test_file else 'NOT FOUND (Skipping Submission Generation)'}")

    return train_file, test_file


def run_pipeline():
    print("=========================================================================")
    print("PrismPrice Kaggle End-to-End Multimodal ML & Report Pipeline")
    print("=========================================================================")

    # 1. Locate datasets
    train_path, test_path = find_kaggle_dataset()

    # 2. Run generate-report CLI
    cmd = [sys.executable, "main.py", "generate-report"]
    if train_path:
        cmd.extend(["--data", train_path])

    print(f"\nExecuting command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3. If test data is available, fit full pipeline and generate submission.csv
    if test_path and os.path.exists(test_path):
        print(f"\nGenerating test predictions for {test_path}...")
        df_test = pd.read_csv(test_path)
        
        # Fit train and predict test via trainer
        cmd_train = [sys.executable, "main.py", "train", "--data", train_path, "--model", "stacker"]
        subprocess.run(cmd_train, check=True)

        cmd_predict = [sys.executable, "main.py", "predict", "--data", test_path, "--output", "submission.csv"]
        subprocess.run(cmd_predict, check=True)
        print("Generated submission.csv successfully!")

    # 4. Zip all report artifacts for 1-click download
    zip_path = "experiments_reports_and_docs.zip"
    print(f"\nArchiving reports and plots into {zip_path}...")
    
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

    print("=========================================================================")
    print("KAGGLE EXECUTION COMPLETE SUCCESSFULLY!")
    print(f"Download '{zip_path}' from Kaggle and unzip into your local repository.")
    print("=========================================================================")


if __name__ == "__main__":
    run_pipeline()
