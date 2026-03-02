# src/pipelines/train_pipeline.py
import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..data.dataset_loader import DatasetLoader
from ..data.parse_features import Parser
from ..features.build_features import FeatureBuilder
from ..features.dimensionality import DimReducer
from ..training.trainer import Trainer
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..models.lgb_model import LGBModel
from ..models.linear_model import LinearModel
from ..models.rf_model import RandomForestModel
from ..models.stacker import Stacker

# Because this is giving error at this time
try:
    from ..models.xgb_model import XGBModel
except Exception:
    XGBModel = None

try:
    from ..models.cat_model import CatModel
except Exception:
    CatModel = None


logger = LoggerFactory.get("train_pipeline")

def run_train_pipeline(cfg: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    End-to-end training pipeline using configuration dict.
    cfg should contain:
      - data_path, text_col, image_col, target_col, sample_frac
      - text_cfg, image_cfg, numeric_cfg, feature_cache
      - dim_method, dim_components, dim_cache
      - models_out, n_splits
      - model params as needed
    Returns summary dict with model comparison & paths.
    """
    # prepare directories
    os.makedirs(cfg.get("experiments_dir", "experiments"), exist_ok=True)
    os.makedirs(cfg.get("models_out", "experiments/models"), exist_ok=True)
    os.makedirs(cfg.get("oof_out", "experiments/oof"), exist_ok=True)

    # 1. Load data (optionally sample small fraction for quick dev)
    loader = DatasetLoader(cfg["data_path"])
    df = loader.sample(frac=cfg.get("sample_frac", 1.0), random_state=cfg.get("random_state", 42))
    logger.info(f"Loaded data with {len(df)} rows")

    # 2. Parse features
    df = Parser.add_parsed_features(df, text_col=cfg.get("text_col", "Description"))

    # 3. Build features (text + image + numeric)
    fb = FeatureBuilder(cfg.get("text_cfg", {}), cfg.get("image_cfg", {}), cfg.get("numeric_cfg", {}), output_cache=cfg.get("feature_cache", "data/processed/features.joblib"))
    X_raw, meta = fb.build(df, text_col=cfg.get("text_col", "Description"), image_col=cfg.get("image_col", "image_path"), force_rebuild=cfg.get("force_rebuild", False))
    logger.info(f"Feature matrix built. meta={meta}")

    # 4. Prepare dense for dim reduction
    X_dense = None
    if hasattr(X_raw, "toarray") or hasattr(X_raw, "todense"):
        try:
            X_dense = np.asarray(X_raw.todense() if hasattr(X_raw, "todense") else X_raw.toarray())
        except Exception:
            X_dense = None
    else:
        X_dense = np.asarray(X_raw)

    # 5. Dimensionality reduction
    reducer = DimReducer(method=cfg.get("dim_method", "umap"), n_components=cfg.get("dim_components", 50), cache_path=cfg.get("dim_cache", "data/processed/dimred.joblib"))
    if X_dense is not None:
        X_final, dim_meta = reducer.fit_transform(X_dense, use_cache=cfg.get("use_dim_cache", True))
    else:
        X_final = X_raw  # sparse, fall back
        dim_meta = {}
    IO.save_pickle(dim_meta, cfg.get("dim_meta_out", "experiments/reports/dim_meta.joblib"))

    # 6. Prepare target
    y = df[cfg.get("target_col", "Price")].values.astype(float)

    # 7. Run CV for each model
    trainer = Trainer(output_dir=cfg.get("models_out", "experiments/models"), n_splits=cfg.get("n_splits", 5), random_state=cfg.get("random_state", 42), stratify=cfg.get("stratify", False))
    results = []
    oof_list = []
    model_names = []

    # define models to run
    model_entries = [
        ("Linear", LinearModel, {}),
        ("RF", RandomForestModel, cfg.get("rf_params", {"n_estimators": cfg.get("rf_n_estimators", 200)})),
        ("LGBM", LGBModel, {"params": cfg.get("lgb_params", {"n_estimators": cfg.get("lgb_n_estimators",200), "learning_rate":0.05})})
    ]

    # append optional models if available
    if XGBModel is not None:
        model_entries.append(("XGB", XGBModel, cfg.get("xgb_params", {"params": {"n_estimators": 200}})))
    if CatModel is not None:
        model_entries.append(("Cat", CatModel, cfg.get("cat_params", {"params": {"iterations": 300}})))

    if model_name:
        selected = model_name.strip().lower()
        model_entries = [entry for entry in model_entries if entry[0].lower() == selected]
        if not model_entries:
            raise ValueError(f"Unsupported or unavailable model_name='{model_name}'. Available: Linear, RF, LGBM" + (", XGB" if XGBModel is not None else "") + (", Cat" if CatModel is not None else ""))

    for name, ModelClass, ctor_params in model_entries:
        try:
            logger.info(f"Training model: {name}")
            models, oof, metrics_summary = trainer.run_cv(ModelClass, model_params=ctor_params, X=X_final, y=y, fit_params={})
            results.append({"name": name, "metrics": metrics_summary})
            oof_list.append(oof.reshape(-1,1))
            model_names.append(name)
        except Exception as e:
            logger.exception(f"Model {name} failed: {e}")

    # 8. Save model comparison
    import pandas as pd
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"model": r["name"], "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "smape": m["smape"]})
    df_report = pd.DataFrame(rows).sort_values("smape")
    IO.save_pickle(df_report, cfg.get("report_out", "experiments/reports/model_comparison.joblib"))
    df_report.to_csv(cfg.get("report_csv", "experiments/reports/model_comparison.csv"), index=False)

    # 9. Save OOF matrix and model names for stacking
    if oof_list:
        OOF = np.hstack(oof_list)
        IO.save_pickle(OOF, os.path.join(cfg.get("oof_out", "experiments/oof"), "oof_matrix.joblib"))
        IO.save_pickle(np.array(model_names), os.path.join(cfg.get("oof_out", "experiments/oof"), "model_names.joblib"))
        logger.info(f"Saved OOF matrix with shape {OOF.shape}")

    # 10. Run Stacker (meta-level) if requested
    if cfg.get("run_stacker", True) and oof_list:
        # fit stacker on OOF
        meta_ooF = OOF
        stacker = Stacker(method=cfg.get("stacker_method", "ridge"), params=cfg.get("stacker_params", {"alpha":1.0}), n_splits=cfg.get("stacker_n_splits", 5), save_path=os.path.join(cfg.get("models_out", "experiments/models"), "stacker.joblib"))
        stacker_summary = stacker.fit_cv(meta_ooF, y, fit_final=True)
        IO.save_pickle(stacker_summary, os.path.join(cfg.get("reports_dir", "experiments/reports"), "stacker_summary.joblib"))
        logger.info(f"Stacker finished. summary: {stacker_summary}")

    summary = {
        "model_report": cfg.get("report_csv", "experiments/reports/model_comparison.csv"),
        "oof_path": os.path.join(cfg.get("oof_out", "experiments/oof"), "oof_matrix.joblib")
    }
    return summary
