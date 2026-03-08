# src/experiments/exp_full_pipeline.py
"""
End-to-end experiment:
 - load data
 - parse features
 - build embeddings (text + image)
 - reduce dims (UMAP preferred, PCA fallback)
 - run K-fold CV for multiple base models (Linear, RF, LGBM, XGB, CatBoost)
 - collect OOFs, compute metrics (RMSE, MAE, R2, SMAPE)
 - save OOFs and per-model artifacts for stacking
"""
import os
import sys
from typing import List, Dict
import numpy as np
import pandas as pd
from ..data.dataset_loader import DatasetLoader
from ..data.parse_features import Parser
from ..features.build_features import FeatureBuilder
from ..features.dimensionality import DimReducer
from ..training.trainer import Trainer
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..utils.metrics import rmse, mae, r2, smape
from ..models.linear_model import LinearModel
from ..models.rf_model import RandomForestModel
from ..models.lgb_model import LGBModel
# optional models: XGBModel, CatModel
try:
    from ..models.xgb_model import XGBModel
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from ..models.cat_model import CatModel
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

logger = LoggerFactory.get("exp_full_pipeline")

def ensure_dirs():
    for d in ["data/processed", "experiments/models", "experiments/oof", "experiments/reports"]:
        os.makedirs(d, exist_ok=True)

def run(config: Dict):
    """
    config keys used:
      - data_path
      - text_col, image_col, target_col
      - sample_frac (optional)
      - n_splits
      - dim_method ('umap' or 'pca')
      - dim_components
    """
    ensure_dirs()
    loader = DatasetLoader(path=config["data_path"])
    df = loader.sample(frac=config.get("sample_frac", 1.0))
    logger.info(f"Data shape after sampling: {df.shape}")

    # 1) Parse structured pieces
    df = Parser.add_parsed_features(df, text_col=config.get("text_col", "Description"))
    logger.info("Parsed textual numeric/unit features")

    # 2) Prepare target and build features (text + image + numeric)
    y = df[config.get("target_col", "Price")].values.astype(float)
    feature_builder = FeatureBuilder(
        text_cfg=config.get("text_cfg", {"method":"sbert", "cache_path":"data/processed/text_embeddings.joblib"}),
        image_cfg=config.get("image_cfg", {"cache_path":"data/processed/image_embeddings.joblib"}),
        numeric_cfg=config.get("numeric_cfg", {"scaler_path":"data/processed/numeric_scaler.joblib"}),
        selector_cfg=config.get("selector_cfg", {}),
        output_cache=config.get("feature_cache", "data/processed/features.joblib")
    )
    X_raw, meta = feature_builder.build(df,
                                       text_col=config.get("text_col", "Description"),
                                       image_col=config.get("image_col", "image_path"),
                                       force_rebuild=config.get("force_rebuild_features", False),
                                       y=y,
                                       mode="train")
    logger.info(f"Built raw feature matrix: shape={X_raw.shape if hasattr(X_raw,'shape') else 'sparse?'}; meta={meta}")

    # convert sparse to dense if needed for UMAP/PCA; keep copy of raw for tree models if you want
    is_sparse = hasattr(X_raw, "toarray") or hasattr(X_raw, "tocsr")
    if is_sparse:
        try:
            X_dense = X_raw.todense() if hasattr(X_raw, "todense") else X_raw.toarray()
            X_dense = np.asarray(X_dense)
        except MemoryError:
            logger.warning("Sparse -> dense conversion OOM; falling back to sparse-compatible flow (no UMAP).")
            X_dense = None
    else:
        X_dense = np.asarray(X_raw)

    # 3) Dimensionality reduction (UMAP preferred; fallback PCA)
    reducer = DimReducer(method=config.get("dim_method", "umap"),
                         n_components=config.get("dim_components", 50),
                         cache_path=config.get("dim_cache", "data/processed/dimred.joblib"))
    if X_dense is None:
        logger.warning("Skipping dimensionality (no dense matrix). Using raw features for modeling.")
        X_final = X_raw
        dim_meta = {}
    else:
        try:
            X_red, dim_meta = reducer.fit_transform(X_dense, use_cache=config.get("use_dim_cache", True), fingerprint=meta.get("feature_fingerprint"))
            X_final = X_red
            logger.info(f"Dim reduction applied. X_final shape {X_final.shape}")
        except Exception as e:
            logger.warning(f"Dim reducer failed ({e}). Falling back to PCA on dense matrix.")
            reducer_pca = DimReducer(method="pca", n_components=min(50, X_dense.shape[1]), cache_path="data/processed/pca_fallback.joblib")
            X_final, dim_meta = reducer_pca.fit_transform(X_dense, use_cache=False, fingerprint=meta.get("feature_fingerprint"))

    # 4) Target summary
    logger.info(f"Target loaded; n={len(y)}; stats: mean={y.mean():.3f} std={y.std():.3f}")

    # 5) Run CV across models
    n_splits = config.get("n_splits", 5)
    trainer = Trainer(output_dir=config.get("models_out", "experiments/models"), n_splits=n_splits, random_state=config.get("random_state", 42), stratify=config.get("stratify", False))

    model_inventory = []
    # Linear
    model_inventory.append(("Linear", LinearModel, {}))
    # RandomForest
    model_inventory.append(("RF", RandomForestModel, {"n_estimators": config.get("rf_n_estimators", 200)}))
    # LightGBM
    model_inventory.append(("LGBM", LGBModel, {"params": config.get("lgb_params", {"n_estimators": 200, "learning_rate":0.05})}))
    # XGBoost
    if XGB_AVAILABLE:
        model_inventory.append(("XGB", XGBModel, {"params": config.get("xgb_params", {"n_estimators":200, "learning_rate":0.05})}))
    else:
        logger.info("XGBoost not available; skipping XGB model.")
    # CatBoost
    if CAT_AVAILABLE:
        model_inventory.append(("Cat", CatModel, {"params": config.get("cat_params", {"iterations":500, "learning_rate":0.03})}))
    else:
        logger.info("CatBoost not available; skipping Cat model.")

    results = []
    oof_list = []
    model_names = []
    # Ensure X_final is indexable for trainer (np array or sparse)
    for name, ModelClass, ctor_params in model_inventory:
        logger.info(f"Running CV for model: {name}")
        try:
            models, oof, metrics_summary = trainer.run_cv(ModelClass, model_params=ctor_params, X=X_final, y=y, fit_params={})
            results.append({"name": name, "metrics": metrics_summary})
            oof_list.append(oof.reshape(-1,1))
            model_names.append(name)
        except Exception as e:
            logger.exception(f"Model {name} failed: {e}")
            continue

    # 6) Collate comparison table
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "model": r["name"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "smape": m["smape"]
        })
    df_report = pd.DataFrame(rows).sort_values("smape")
    print("\n=== Model Comparison ===")
    print(df_report.to_string(index=False))
    # save report
    IO.save_pickle(df_report, "experiments/reports/model_comparison.joblib")
    df_report.to_csv("experiments/reports/model_comparison.csv", index=False)

    # 7) Save OOF matrix for stacking
    if oof_list:
        OOF = np.hstack(oof_list)
        IO.save_pickle(OOF, "experiments/oof/oof_matrix.joblib")
        IO.save_pickle(np.array(model_names), "experiments/oof/model_names.joblib")
        logger.info(f"Saved OOF matrix shape {OOF.shape} and model names.")
    else:
        logger.warning("No OOFs to save (no successful model runs).")

    # 8) Save metadata
    IO.save_pickle({"dim_meta": dim_meta, "feature_meta": meta}, "experiments/reports/metadata.joblib")
    logger.info("Experiment complete.")
    return df_report

if __name__ == "__main__":
    # basic config; change paths as needed
    cfg = {
        "data_path": "data/raw/train.csv",
        "text_col": "Description",
        "image_col": "image_path",
        "target_col": "Price",
        "sample_frac": 0.01,            # small sample for quick runs; set to 1.0 for full run
        "n_splits": 3,
        "dim_method": "umap",
        "dim_components": 64,
        "force_rebuild_features": False,
        "use_dim_cache": True,
        "models_out": "experiments/models",
        "rf_n_estimators": 100,
        "lgb_params": {"n_estimators":200, "learning_rate":0.05},
        "xgb_params": {"n_estimators":200, "learning_rate":0.05},
        "cat_params": {"iterations":300, "learning_rate":0.03},
        "random_state": 42,
        "stratify": False
    }
    run(cfg)
    
    