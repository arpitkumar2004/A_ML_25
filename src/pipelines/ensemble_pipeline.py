# src/pipelines/ensemble_pipeline.py
from typing import Dict, Any
import os
import numpy as np
from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory
from ..models.stacker import Stacker

logger = LoggerFactory.get("ensemble_pipeline")

def run_ensemble_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load OOF matrix and target y (or require y passed) and run Stacker meta-level CV and final fit.
    cfg should contain:
      - oof_path
      - y_path (or y can be passed as array)
      - stacker_method, stacker_params, n_splits
      - stacker_save_path
    """
    oof_path = cfg.get("oof_path", "experiments/oof/oof_matrix.joblib")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(oof_path)
    OOF = IO.load_pickle(oof_path)  # n x m

    # load target y: either from cfg y_path or expect passed in cfg['y_array']
    if cfg.get("y_array") is not None:
        y = cfg["y_array"]
    elif cfg.get("y_path") is not None:
        y_df = IO.read_csv(cfg["y_path"])
        target_col = cfg.get("target_col", "price")
        y = y_df[target_col].values
    else:
        raise ValueError("y_array or y_path must be provided in cfg to run ensemble pipeline.")

    # fit stacker
    stacker = Stacker(method=cfg.get("stacker_method", "ridge"),
                      params=cfg.get("stacker_params", {"alpha":1.0}),
                      n_splits=cfg.get("n_splits", 5),
                      random_state=cfg.get("random_state", 42),
                      save_path=cfg.get("stacker_save_path", "experiments/models/stacker.joblib"),
                      save_oof_path=cfg.get("meta_oof_out", "experiments/oof/meta_oof.joblib"))
    summary = stacker.fit_cv(OOF, y, fit_final=cfg.get("fit_final", True))
    IO.save_pickle(summary, cfg.get("summary_out", "experiments/reports/stacker_summary.joblib"))
    logger.info(f"Stacker summary saved to {cfg.get('summary_out', 'experiments/reports/stacker_summary.joblib')}")
    return summary
