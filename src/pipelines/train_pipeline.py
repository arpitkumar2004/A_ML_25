# src/pipelines/train_pipeline.py
import os
import time
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
from ..utils.run_registry import make_run_id, write_run_manifest
from ..registry.model_registry import register_run
from ..utils.mlflow_utils import MLflowTracker, mlflow_link
from ..utils.column_aliases import resolve_column_name

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
    run_id = cfg.get("run_id") or make_run_id(prefix="train")
    timings = {}
    t0_total = time.perf_counter()
    tracker = MLflowTracker(cfg=cfg, stage="train", run_id=run_id)
    tracker.start()
    tracker.log_config(cfg)
    tracker.set_tags({"selected_model_filter": model_name or "all"})

    os.makedirs(cfg.get("experiments_dir", "experiments"), exist_ok=True)
    os.makedirs(cfg.get("models_out", "experiments/models"), exist_ok=True)
    os.makedirs(cfg.get("oof_out", "experiments/oof"), exist_ok=True)

    try:
        # 1. Load data (optionally sample small fraction for quick dev)
        t_load = time.perf_counter()
        loader = DatasetLoader(cfg["data_path"])
        df = loader.sample(frac=cfg.get("sample_frac", 1.0), random_state=cfg.get("random_state", 42))
        logger.info(f"Loaded data with {len(df)} rows")
        timings["load_data"] = round(time.perf_counter() - t_load, 4)
        tracker.log_metrics({"train_rows": len(df)})

        # 2. Parse features
        t_parse = time.perf_counter()
        text_col = resolve_column_name(df.columns, cfg.get("text_col", "catalog_content"))
        image_col = resolve_column_name(df.columns, cfg.get("image_col", "image_link"))
        target_col = resolve_column_name(df.columns, cfg.get("target_col", "price"))
        id_col = resolve_column_name(df.columns, cfg.get("id_col", "sample_id"))
        cfg["text_col"] = text_col
        cfg["image_col"] = image_col
        cfg["target_col"] = target_col
        cfg["id_col"] = id_col

        df = Parser.add_parsed_features(df, text_col=text_col)
        timings["parse_features"] = round(time.perf_counter() - t_parse, 4)

        # 3. Prepare target before feature-selection-aware building
        y = df[target_col].values.astype(float)

        # 4. Build features (text + image + numeric + optional selection)
        t_features = time.perf_counter()
        fb = FeatureBuilder(
            cfg.get("text_cfg", {}),
            cfg.get("image_cfg", {}),
            cfg.get("numeric_cfg", {}),
            selector_cfg=cfg.get("selector_cfg", {}),
            output_cache=cfg.get("feature_cache", "data/processed/features.joblib"),
        )
        X_raw, meta = fb.build(
            df,
            text_col=text_col,
            image_col=image_col,
            force_rebuild=cfg.get("force_rebuild", False),
            y=y,
            mode="train",
        )
        logger.info(f"Feature matrix built. meta={meta}")
        timings["build_features"] = round(time.perf_counter() - t_features, 4)

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
        t_dim = time.perf_counter()
        reducer = DimReducer(method=cfg.get("dim_method", "umap"), n_components=cfg.get("dim_components", 50), cache_path=cfg.get("dim_cache", "data/processed/dimred.joblib"))
        if X_dense is not None:
            X_final, dim_meta = reducer.fit_transform(X_dense, use_cache=cfg.get("use_dim_cache", True), fingerprint=meta.get("feature_fingerprint"))
        else:
            X_final = X_raw  # sparse, fall back
            dim_meta = {}
        IO.save_pickle(dim_meta, cfg.get("dim_meta_out", "experiments/reports/dim_meta.joblib"))
        timings["dimensionality"] = round(time.perf_counter() - t_dim, 4)

        # 7. Run CV for each model
        t_train = time.perf_counter()
        trainer = Trainer(output_dir=cfg.get("models_out", "experiments/models"), n_splits=cfg.get("n_splits", 5), random_state=cfg.get("random_state", 42), stratify=cfg.get("stratify", False))
        results = []
        oof_list = []
        model_names = []
        trained_models = {}

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
                trained_models[name] = models[0] if models else None
                tracker.log_metrics(
                    {
                        f"model.{name}.rmse": metrics_summary.get("rmse", 0.0),
                        f"model.{name}.mae": metrics_summary.get("mae", 0.0),
                        f"model.{name}.r2": metrics_summary.get("r2", 0.0),
                        f"model.{name}.smape": metrics_summary.get("smape", 0.0),
                    }
                )
            except Exception as e:
                logger.exception(f"Model {name} failed: {e}")

        # 8. Save model comparison
        rows = []
        for r in results:
            m = r["metrics"]
            rows.append({"model": r["name"], "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "smape": m["smape"]})
        df_report = pd.DataFrame(rows).sort_values("smape")
        IO.save_pickle(df_report, cfg.get("report_out", "experiments/reports/model_comparison.joblib"))
        df_report.to_csv(cfg.get("report_csv", "experiments/reports/model_comparison.csv"), index=False)
        if not df_report.empty:
            best = df_report.iloc[0].to_dict()
            best_name = str(best.get("model", ""))
            registry_meta = tracker.register_model_if_possible(trained_models.get(best_name), model_key=best_name)
            tracker.log_metrics(
                {
                    "best.rmse": best.get("rmse", 0.0),
                    "best.mae": best.get("mae", 0.0),
                    "best.r2": best.get("r2", 0.0),
                    "best.smape": best.get("smape", 0.0),
                }
            )
            tracker.set_tags({"best_model": best.get("model", "")})
        else:
            registry_meta = {
                "enabled": False,
                "registered": False,
                "registered_model_name": "",
                "version": None,
                "model_uri": "",
                "artifact_path": "",
                "error": "",
            }

        # 9. Save OOF matrix and model names for stacking
        if oof_list:
            OOF = np.hstack(oof_list)
            IO.save_pickle(OOF, os.path.join(cfg.get("oof_out", "experiments/oof"), "oof_matrix.joblib"))
            IO.save_pickle(np.array(model_names), os.path.join(cfg.get("oof_out", "experiments/oof"), "model_names.joblib"))
            logger.info(f"Saved OOF matrix with shape {OOF.shape}")
            tracker.log_metrics({"oof_cols": OOF.shape[1], "oof_rows": OOF.shape[0]})

        stacker_summary_path = os.path.join(cfg.get("reports_dir", "experiments/reports"), "stacker_summary.joblib")

        # 10. Run Stacker (meta-level) if requested
        if cfg.get("run_stacker", True) and oof_list:
            # fit stacker on OOF
            meta_ooF = OOF
            stacker = Stacker(method=cfg.get("stacker_method", "ridge"), params=cfg.get("stacker_params", {"alpha":1.0}), n_splits=cfg.get("stacker_n_splits", 5), save_path=os.path.join(cfg.get("models_out", "experiments/models"), "stacker.joblib"))
            stacker_summary = stacker.fit_cv(meta_ooF, y, fit_final=True)
            IO.save_pickle(stacker_summary, stacker_summary_path)
            logger.info(f"Stacker finished. summary: {stacker_summary}")
            tracker.log_metrics({
                "stacker.rmse": stacker_summary.get("rmse", 0.0),
                "stacker.mae": stacker_summary.get("mae", 0.0),
                "stacker.r2": stacker_summary.get("r2", 0.0),
                "stacker.smape": stacker_summary.get("smape", 0.0),
            })
        timings["train_and_eval"] = round(time.perf_counter() - t_train, 4)
        timings["total"] = round(time.perf_counter() - t0_total, 4)

        manifest_outputs = {
            "model_report": cfg.get("report_csv", "experiments/reports/model_comparison.csv"),
            "oof_path": os.path.join(cfg.get("oof_out", "experiments/oof"), "oof_matrix.joblib"),
            "dim_cache": cfg.get("dim_cache", "data/processed/dimred.joblib"),
            "feature_cache": cfg.get("feature_cache", "data/processed/features.joblib"),
            "selector_path": cfg.get("selector_cfg", {}).get("save_path", "data/processed/feature_selector.joblib"),
            "numeric_scaler_path": cfg.get("numeric_cfg", {}).get("scaler_path", "data/processed/numeric_scaler.joblib"),
            "stacker_summary": stacker_summary_path,
            "mlflow": mlflow_link(tracker.mlflow_run_id, tracker.tracking_uri, tracker.experiment_name),
            "model_registry": registry_meta,
        }
        manifest_path = write_run_manifest(
            run_id=run_id,
            stage="train",
            cfg=cfg,
            outputs=manifest_outputs,
            timings=timings,
            registry_dir=cfg.get("registry_dir", "experiments/registry"),
        )
        register_run(
            run_id=run_id,
            manifest_path=manifest_path,
            stage="train",
            registry_dir=cfg.get("registry_dir", "experiments/registry"),
            status="staging",
            tracking={"mlflow": manifest_outputs.get("mlflow", {})},
        )

        tracker.log_metrics({f"timing.{k}": v for k, v in timings.items()})
        tracker.log_artifacts_if_exists(
            [
                cfg.get("report_csv", "experiments/reports/model_comparison.csv"),
                stacker_summary_path,
                os.path.join(cfg.get("oof_out", "experiments/oof"), "model_names.joblib"),
                manifest_path,
            ],
            artifact_path="artifacts",
        )

        summary = {
            "model_report": cfg.get("report_csv", "experiments/reports/model_comparison.csv"),
            "oof_path": os.path.join(cfg.get("oof_out", "experiments/oof"), "oof_matrix.joblib"),
            "run_id": run_id,
            "mlflow_run_id": tracker.mlflow_run_id,
            "manifest_path": manifest_path,
            "timings_seconds": timings,
        }
        tracker.end(status="FINISHED")
        return summary
    except Exception:
        tracker.end(status="FAILED")
        raise
