#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# create_project_structure.sh
# ---------------------------
# Creates the directory tree and files for:
#   price_prediction_competition/
# Intended to be run from the parent directory that already contains price_prediction_competition/.
#
# Usage:
#   chmod +x create_project_structure.sh
#   ./create_project_structure.sh
# ---------------------------

PROJECT_DIR="price_prediction_competition"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Error: directory '$PROJECT_DIR' not found in current working dir: $(pwd)"
  echo "Please create it or run this script from the correct parent directory."
  exit 1
fi

cd "$PROJECT_DIR"
echo "Working in $(pwd)"

# Helper to create files only if they don't exist
create_file() {
  local path="$1"
  local content="$2"
  if [ -f "$path" ]; then
    echo " - exists: $path"
  else
    mkdir -p "$(dirname "$path")"
    printf "%s\n" "$content" > "$path"
    echo " + created: $path"
  fi
}

# Directories & files
echo "Creating directories and files..."

# configs
create_file "configs/data/base_data.yaml" \
"## base_data.yaml
dataset:
  name: base
  path: data/raw
  target: price
preprocessing:
  fill_na: true
  normalise: false
"

create_file "configs/data/augmented_data.yaml" \
"## augmented_data.yaml
dataset:
  name: augmented
  path: data/processed/augmented
augmentation:
  enable: true
  methods:
    - noise
    - synthetic_features
"

create_file "configs/data/external_data.yaml" \
"## external_data.yaml
external:
  provider: example_provider
  path: data/external
"

create_file "configs/model/lgbm.yaml" \
"## lgbm.yaml
model:
  name: lightgbm
  params:
    num_leaves: 128
    learning_rate: 0.03
    n_estimators: 10000
"

create_file "configs/model/xgb.yaml" \
"## xgb.yaml
model:
  name: xgboost
  params:
    max_depth: 8
    eta: 0.05
    nrounds: 2000
"

create_file "configs/model/bert.yaml" \
"## bert.yaml
model:
  name: bert_regressor
  pretrained: bert-base-uncased
  fine_tune: true
"

create_file "configs/model/fusion_nn.yaml" \
"## fusion_nn.yaml
model:
  name: fusion_nn
  arch:
    tabular_layers: [256, 128]
    text_pooling: mean
"

create_file "configs/model/ensemble.yaml" \
"## ensemble.yaml
ensemble:
  method: stacking
  base_models:
    - lgbm
    - xgb
    - bert_regressor
  meta_model: lgbm
"

create_file "configs/features/text_features.yaml" \
"## text_features.yaml
text:
  use_bert: true
  max_len: 128
"

create_file "configs/features/numeric_features.yaml" \
"## numeric_features.yaml
numeric:
  normalise: true
  impute_strategy: median
"

create_file "configs/features/parsed_features.yaml" \
"## parsed_features.yaml
parsed:
  extractors:
    - count_tokens
    - count_numbers
"

create_file "configs/features/all_features.yaml" \
"## all_features.yaml
features:
  include: [numeric, text, parsed]
"

create_file "configs/training/cv_stratified.yaml" \
"## cv_stratified.yaml
cv:
  type: StratifiedKFold
  n_splits: 5
  shuffle: true
  random_state: 42
"

create_file "configs/training/params_sweep.yaml" \
"## params_sweep.yaml
optuna:
  n_trials: 100
  direction: minimize
"

create_file "configs/training/final_train.yaml" \
"## final_train.yaml
train:
  use_all_data: true
  save_model: true
"

create_file "configs/inference/inference.yaml" \
"## inference.yaml
inference:
  batch_size: 512
  ensemble_weights:
    lgbm: 0.4
    xgb: 0.3
    bert: 0.3
"

# src python structure
create_file "src/__init__.py" "# src package"

create_file "src/data/__init__.py" "# data package"

create_file "src/data/dataset_loader.py" \
"\"\"\"Dataset loader utilities.\"\"\"
import os
import pandas as pd

def load_csv(path):
    return pd.read_csv(path)
"

create_file "src/data/text_cleaning.py" \
"\"\"\"Text cleaning utilities.\"\"\"
import re
def clean_text(s: str) -> str:
    if not s: return ''
    s = re.sub(r'\\s+', ' ', s)
    return s.strip()
"

create_file "src/data/parse_features.py" \
"\"\"\"Feature parsing helpers (counts, weights).\"\"\"
def parse_price_string(s: str):
    try:
        return float(s)
    except:
        return None
"

create_file "src/data/augmentations.py" \
"\"\"\"Data augmentation helpers.\"\"\"
def add_noise(series, scale=0.01):
    import numpy as np
    return series * (1 + np.random.normal(0, scale, size=len(series)))
"

# features
create_file "src/features/__init__.py" "# features package"

create_file "src/features/text_embeddings.py" \
"\"\"\"Text embedding wrappers (BERT / TF-IDF / SBERT).\"\"\"
def embed_texts(texts):
    # placeholder
    return [[0.0]] * len(texts)
"

create_file "src/features/numeric_features.py" \
"\"\"\"Numeric transformations.\"\"\"
def scale_numeric(df, cols):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df
"

create_file "src/features/categorical_features.py" \
"\"\"\"Categorical feature encoders.\"\"\"
def one_hot_encode(df, cols):
    return df.join(pd.get_dummies(df[cols], drop_first=True))
"

create_file "src/features/feature_selector.py" \
"\"\"\"Select features using importance / statistical methods.\"\"\"
def select_top_k(features_df, k=50):
    return features_df.columns[:k]
"

create_file "src/features/build_features.py" \
"\"\"\"Combine all feature groups into a matrix.\"\"\"
def build_all(df):
    # placeholder pipeline
    return df
"

# models
create_file "src/models/__init__.py" "# models package"

create_file "src/models/base_model.py" \
"\"\"\"BaseModel interface.\"\"\"
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): pass
    @abstractmethod
    def predict(self, X): pass
"

create_file "src/models/lgbm_model.py" \
"\"\"\"LightGBM wrapper.\"\"\"
import lightgbm as lgb
class LGBMModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
"

create_file "src/models/xgb_model.py" \
"\"\"\"XGBoost wrapper.\"\"\"
import xgboost as xgb
class XGBModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
"

create_file "src/models/catboost_model.py" \
"\"\"\"CatBoost wrapper.\"\"\"
from catboost import CatBoostRegressor
class CatModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = CatBoostRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
"

create_file "src/models/bert_regressor.py" \
"\"\"\"Placeholder BERT regressor wrapper.\"\"\"
def train_bert(...):
    raise NotImplementedError
"

create_file "src/models/fusion_nn.py" \
"\"\"\"Fusion NN (text + tabular) placeholder.\"\"\"
def build_fusion_model():
    pass
"

create_file "src/models/ensemble.py" \
"\"\"\"Stacking / blending utilities.\"\"\"
def blend(preds_list, weights=None):
    import numpy as np
    if weights is None:
        weights = [1/len(preds_list)] * len(preds_list)
    return sum(w * p for w,p in zip(weights, preds_list))
"

# training
create_file "src/training/__init__.py" "# training package"

create_file "src/training/trainer.py" \
"\"\"\"Generic training pipeline.\"\"\"
def run_training(cfg):
    print('Running training with', cfg)
"

create_file "src/training/evaluator.py" \
"\"\"\"Evaluation and CV helpers.\"\"\"
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    import numpy as np
    return float(mean_squared_error(y_true, y_pred, squared=False))
"

create_file "src/training/utils_cv.py" \
"\"\"\"Cross-validation utilities (Stratified/Group CV).\"\"\"
def get_cv(cfg):
    return None
"

create_file "src/training/metrics.py" \
"\"\"\"Common metrics (RMSE, MAE, RMSLE).\"\"\"
def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)
"

create_file "src/training/hyperparam_opt.py" \
"\"\"\"Optuna / RayTune wrapper (placeholder).\"\"\"
def run_optuna(study_name, objective):
    pass
"

# inference
create_file "src/inference/__init__.py" "# inference package"

create_file "src/inference/predict.py" \
"\"\"\"Unified prediction pipeline.\"\"\"
def predict_pipeline(model, X):
    return model.predict(X)
"

create_file "src/inference/postprocess.py" \
"\"\"\"Postprocessing outputs.\"\"\"
def clamp_preds(preds, lower=0.0):
    import numpy as np
    return np.clip(preds, lower, None)
"

# utils
create_file "src/utils/__init__.py" "# utils package"

create_file "src/utils/io.py" \
"\"\"\"IO helpers for models, OOF, and data.\"\"\"
import joblib
def save_obj(obj, path):
    joblib.dump(obj, path)
def load_obj(path):
    return joblib.load(path)
"

create_file "src/utils/logging_utils.py" \
"\"\"\"Logging & experiment tracking helpers.\"\"\"
def get_logger(name='project'):
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
"

create_file "src/utils/seed_everything.py" \
"\"\"\"Seed helper for reproducibility.\"\"\"
import random, os
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
"

create_file "src/utils/timer.py" \
"\"\"\"Simple timer context manager.\"\"\"
import time
class Timer:
    def __init__(self, name='timer'):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print(f\"{self.name} took {time.time()-self.start:.2f}s\")
"

create_file "src/utils/visualization.py" \
"\"\"\"Visualization helpers.\"\"\"
def plot_metric(history):
    pass
"

# pipelines & experiments
create_file "src/pipelines/train_pipeline.py" \
"\"\"\"High-level training pipeline that orchestrates everything.\"\"\"
def main(cfg_path):
    print('Training pipeline reading', cfg_path)
"

create_file "src/pipelines/feature_pipeline.py" \
"\"\"\"Feature-only pipeline.\"\"\"
def build_features(cfg):
    print('Building features from', cfg)
"

create_file "src/pipelines/ensemble_pipeline.py" \
"\"\"\"Create stack / blend artifacts.\"\"\"
def build_ensemble():
    pass
"

create_file "src/pipelines/inference_pipeline.py" \
"\"\"\"End-to-end inference pipeline.\"\"\"
def run_inference(cfg):
    print('Running inference with', cfg)
"

create_file "src/experiments/exp_lgbm_baseline.py" \
"\"\"\"Example experiment: LightGBM baseline.\"\"\"
if __name__ == '__main__':
    print('Run LightGBM baseline experiment')
"

create_file "src/experiments/exp_bert_finetune.py" \
"\"\"\"Example experiment: BERT fine-tuning.\"\"\"
if __name__ == '__main__':
    print('Run BERT finetune experiment')
"

create_file "src/experiments/exp_fusion_nn.py" \
"\"\"\"Example experiment: Fusion NN.\"\"\"
if __name__ == '__main__':
    print('Run fusion NN experiment')
"

create_file "src/experiments/exp_stacking_v1.py" \
"\"\"\"Experiment: stacking v1.\"\"\"
if __name__ == '__main__':
    print('Run stacking v1')
"

create_file "src/experiments/exp_final_submission.py" \
"\"\"\"Experiment: final submission builder.\"\"\"
if __name__ == '__main__':
    print('Build final submission')
"

# notebooks
mkdir -p notebooks
for nb in "01_eda.ipynb" "02_text_parsing.ipynb" "03_feature_analysis.ipynb" "04_model_diagnostics.ipynb"; do
  if [ -f "notebooks/$nb" ]; then
    echo " - exists: notebooks/$nb"
  else
    # create tiny notebook skeleton
    printf '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}' > "notebooks/$nb"
    echo " + created: notebooks/$nb"
  fi
done

# data folders
mkdir -p data/{raw,interim,processed,external}

# experiments folder
mkdir -p experiments/{oof_predictions,models,logs,submissions,reports}

# requirements, main, run script, README
create_file "requirements.txt" \
"# Put pinned deps here
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
joblib
pytest
"

create_file "main.py" \
"\"\"\"Entry point for CLI: python main.py train --config configs/model/lgbm.yaml\"\"\"
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','inference'], help='mode')
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    print('Mode:', args.mode, 'Config:', args.config)

if __name__ == '__main__':
    main()
"

create_file "run_experiment.sh" \
"#!/usr/bin/env bash
# Example runner to dispatch multiple experiments
python main.py train --config configs/model/lgbm.yaml
"

chmod +x run_experiment.sh

create_file "README.md" \
"# Price Prediction Competition\n\nProject scaffold created by create_project_structure.sh\n\nUsage:\n- Edit YAML files in `configs/`\n- Run experiments via `python main.py train --config <path>` or the `src/experiments/` scripts\n"

# CI/CD: GitHub Actions workflow
create_file ".github/workflows/ci.yml" \
"name: CI\n\non:\n  push:\n    branches: [ main, master ]\n  pull_request:\n    branches: [ main, master ]\n\njobs:\n  lint-and-test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - name: Set up Python\n        uses: actions/setup-python@v4\n        with:\n          python-version: '3.10'\n      - name: Install dependencies\n        run: |\n          python -m pip install --upgrade pip\n          pip install -r requirements.txt\n      - name: Run tests\n        run: |\n          pytest -q\n\n  build-docker:\n    runs-on: ubuntu-latest\n    needs: lint-and-test\n    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'\n    steps:\n      - uses: actions/checkout@v4\n      - name: Build Docker image\n        run: |\n          docker build -t price-prediction:latest ./docker\n"

# docker files
create_file "docker/Dockerfile" \
"FROM python:3.10-slim\nWORKDIR /app\nCOPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . /app\nCMD [\"python\", \"main.py\"]\n"

create_file "docker/.dockerignore" \
"__pycache__\n*.pyc\n.data\n.env\n"

# tests folder
mkdir -p ci_cd/tests
create_file "ci_cd/tests/test_smoke.py" \
"def test_imports():\n    import importlib\n    importlib.import_module('src')\n    assert True\n"

# linting + pipeline + mlflow placeholders
mkdir -p ci_cd/linting
create_file "ci_cd/linting/.pylintrc" \
"[MASTER]\nignore=venv\n"

mkdir -p ci_cd/docker
create_file "ci_cd/docker/Dockerfile.ci" \
"FROM alpine:3.18\n# CI helper image placeholder\n"

create_file "ci_cd/pipelines.yml" \
"# CI/CD pipeline config placeholder for other CI systems\n"

mkdir -p ci_cd/mlflow
create_file "ci_cd/mlflow/mlflow.yml" \
"# mlflow server config placeholder\n"

# Add simple .gitignore
create_file ".gitignore" \
"__pycache__/\n*.pyc\n.env\ndata/\nnotebooks/.ipynb_checkpoints\n"

echo "All files created. Summary:"
tree -a -I 'venv|__pycache__' -L 3 || true

echo "Done. You can now commit the changes."
