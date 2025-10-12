# Price Prediction Competition - Skeleton

Baseline pipeline: TF-IDF (title + description) -> LightGBM in CV mode.

Usage:
- Prepare `data/raw/train.csv` with columns: unique_identifier, Description, Price
- Install: `pip install -r requirements.txt`
- Train: `python main.py train --config configs/model/lgbm.yaml`
- Inference (demo): `python main.py infer --config configs/model/lgbm.yaml`

Project layout:
- `src/` : code modules for data, features, models, training, pipelines.
- `configs/` : YAML-driven experiments.
- `experiments/` : artifacts and outputs.

This skeleton is intentionally minimal. Replace modules with more advanced versions (BERT, CLIP, fusion networks, Optuna sweeps) as you iterate.
