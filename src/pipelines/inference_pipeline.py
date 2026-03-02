# src/pipelines/inference_pipeline.py
from typing import Dict, Any
import os
import pandas as pd
from ..inference.predict import PredictPipeline
from ..inference.postprocess import Postprocessor
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO

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
    df_in = IO.read_csv(cfg["input_path"])
    pp = PredictPipeline(
        text_cfg=cfg.get("text_cfg"),
        image_cfg=cfg.get("image_cfg"),
        numeric_cfg=cfg.get("numeric_cfg"),
        feature_cache=cfg.get("feature_cache", "data/processed/features.joblib"),
        dim_cache=cfg.get("dim_cache", "data/processed/dimred.joblib"),
        models_dir=cfg.get("models_dir", "experiments/models"),
        oof_meta_path=cfg.get("oof_meta_path", "experiments/oof/model_names.joblib"),
        stacker_path=cfg.get("stacker_path", "experiments/models/stacker.joblib")
    )
    preds = pp.predict(df_in, text_col=cfg.get("text_col","Description"), image_col=cfg.get("image_col","image_path"))
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
    logger.info(f"Saved predictions to {out_path}")
    return out_path
