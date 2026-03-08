# src/pipelines/inference_pipeline.py
from typing import Dict, Any
import os
import time
import pandas as pd
from ..inference.predict import PredictPipeline
from ..inference.postprocess import Postprocessor
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..utils.run_registry import make_run_id, write_run_manifest, append_jsonl

logger = LoggerFactory.get("inference_pipeline")

def run_inference_pipeline(cfg: Dict[str, Any]) -> str:
    """
    Run inference on a CSV file and save predictions to output path.
    cfg:
      - input_path: path to CSV with same schema as training
      - output_path: where to save submission CSV
      - text_col, image_col
      - model artifacts paths (optional overrides)
    Returns: path to saved CSV
    """
    run_id = cfg.get("run_id") or make_run_id(prefix="infer")
    timings = {}
    t0_total = time.perf_counter()

    t_load = time.perf_counter()
    df_in = IO.read_csv(cfg["input_path"])
    timings["load_input"] = round(time.perf_counter() - t_load, 4)

    t_init = time.perf_counter()
    pp = PredictPipeline(
        text_cfg=cfg.get("text_cfg"),
        image_cfg=cfg.get("image_cfg"),
        numeric_cfg=cfg.get("numeric_cfg"),
        selector_cfg=cfg.get("selector_cfg"),
        feature_cache=cfg.get("feature_cache", "data/processed/features.joblib"),
        dim_cache=cfg.get("dim_cache", "data/processed/dimred.joblib"),
        models_dir=cfg.get("models_dir", "experiments/models"),
        oof_meta_path=cfg.get("oof_meta_path", "experiments/oof/model_names.joblib"),
        stacker_path=cfg.get("stacker_path", "experiments/models/stacker.joblib")
    )
    timings["pipeline_init"] = round(time.perf_counter() - t_init, 4)

    t_predict = time.perf_counter()
    preds = pp.predict(
        df_in,
        text_col=cfg.get("text_col", "Description"),
        image_col=cfg.get("image_col", "image_path"),
        force_rebuild_features=cfg.get("force_rebuild_features", False),
    )
    timings["predict"] = round(time.perf_counter() - t_predict, 4)
    t_post = time.perf_counter()
    # optional inverse transforms
    if cfg.get("target_transform") == "log1p":
        preds = Postprocessor.invert_log1p(preds)
    preds = Postprocessor.clip_min(preds, cfg.get("min_value", 0.0))
    if cfg.get("round", True):
        preds = Postprocessor.round_to_cents(preds)
    # prepare submission dataframe
    id_col = cfg.get("id_col", "unique_identifier")
    out_df = Postprocessor.to_submission_df(df_in[id_col].values, preds, id_col=id_col, pred_col=cfg.get("pred_col", "predicted_price"))
    out_path = cfg.get("output_path", "experiments/submissions/prediction.csv")
    IO.to_csv(out_df, out_path, index=False)
    timings["postprocess_and_save"] = round(time.perf_counter() - t_post, 4)
    timings["total"] = round(time.perf_counter() - t0_total, 4)

    n_rows = int(len(df_in))
    latency_record = {
        "run_id": run_id,
        "rows": n_rows,
        "total_seconds": timings["total"],
        "predict_seconds": timings["predict"],
        "seconds_per_row": (timings["total"] / max(n_rows, 1)),
        "output_path": out_path,
    }
    append_jsonl(cfg.get("latency_log_path", "experiments/monitoring/latency_events.jsonl"), latency_record)

    manifest_outputs = {
        "output_path": out_path,
        "models_dir": cfg.get("models_dir", "experiments/models"),
        "feature_cache": cfg.get("feature_cache", "data/processed/features.joblib"),
        "dim_cache": cfg.get("dim_cache", "data/processed/dimred.joblib"),
        "selector_path": cfg.get("selector_cfg", {}).get("save_path", "data/processed/feature_selector.joblib"),
        "latency_log_path": cfg.get("latency_log_path", "experiments/monitoring/latency_events.jsonl"),
        "rows": n_rows,
    }
    manifest_path = write_run_manifest(
        run_id=run_id,
        stage="inference",
        cfg=cfg,
        outputs=manifest_outputs,
        timings=timings,
        registry_dir=cfg.get("registry_dir", "experiments/registry"),
    )

    logger.info(f"Saved predictions to {out_path}")
    logger.info(f"Inference manifest saved to {manifest_path}")
    return out_path
