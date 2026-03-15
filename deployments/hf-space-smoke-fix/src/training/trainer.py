# src/training/trainer.py
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..training.metrics import rmse, mae, r2, smape
from ..training.utils_cv import make_folds
import os
import joblib
from ..models.base_model import BaseModel

logger = LoggerFactory.get("trainer")


class Trainer:
    """
    Trainer for KFold CV with OOF predictions and per-fold model persistence.

    Example:
        trainer = Trainer(output_dir='experiments/models', n_splits=5, random_state=42)
        models, oof = trainer.run_cv(model_class, X, y, model_params)
    """

    def __init__(self, output_dir: str = "experiments/models", n_splits: int = 5, random_state: int = 42, stratify: bool = False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify = stratify

    def run_cv(self, model_class: Any, model_params: Optional[Dict] = None, X=None, y=None, fit_params: Optional[Dict] = None) -> Tuple[List[BaseModel], np.ndarray, Dict]:
        """
        Run CV with the provided model_class (class object implementing BaseModel interface)
        model_class: class or callable returning model instance (e.g. LGBModel)
        model_params: dict passed to constructor
        X: features (numpy array or sparse)
        y: target (numpy array)
        fit_params: optional dict passed to fit() (e.g., eval_set)
        Returns (models_list, oof_predictions, metrics_summary)
        """
        model_params = model_params or {}
        fit_params = fit_params or {}
        y = np.array(y)
        n = len(y)
        oof = np.zeros(n, dtype=float)
        fold_models = []
        fold_metrics = []

        for fold, (tr_idx, val_idx) in enumerate(make_folds(y, n_splits=self.n_splits, random_state=self.random_state, stratify=self.stratify)):
            logger.info(f"Starting fold {fold + 1}/{self.n_splits} — train {len(tr_idx)} val {len(val_idx)}")
            X_tr = X[tr_idx]
            X_val = X[val_idx]
            y_tr = y[tr_idx]
            y_val = y[val_idx]

            # instantiate model
            model: BaseModel = model_class(**model_params) if isinstance(model_class, type) else model_class
            # Fit with optional eval_set as (X_val, y_val) if supported
            try:
                if hasattr(model, "fit"):
                    if fit_params.get("use_eval_set", True):
                        # many wrappers accept eval_set as [(X_val, y_val)]
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                    else:
                        model.fit(X_tr, y_tr)
                else:
                    raise RuntimeError("Model has no fit()")
            except TypeError:
                # fallback: call fit without eval_set
                model.fit(X_tr, y_tr)

            preds_val = model.predict(X_val)
            oof[val_idx] = preds_val
            # save model
            model_path = os.path.join(self.output_dir, f"fold_{fold+1}_{model.__class__.__name__}.joblib")
            IO.save_pickle(model, model_path)
            logger.info(f"Saved fold model to {model_path}")
            fold_models.append(model)

            # metrics
            m = {
                "fold": fold + 1,
                "rmse": rmse(y_val, preds_val),
                "mae": mae(y_val, preds_val),
                "r2": r2(y_val, preds_val),
                "smape": smape(y_val, preds_val),
            }
            fold_metrics.append(m)
            logger.info(f"Fold {fold + 1} metrics: RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, R2={m['r2']:.6f}, SMAPE={m['smape']:.4f}%")

        # overall metrics
        metrics_summary = {
            "rmse": rmse(y, oof),
            "mae": mae(y, oof),
            "r2": r2(y, oof),
            "smape": smape(y, oof),
            "folds": fold_metrics
        }
        # save OOF
        IO.save_pickle(oof, os.path.join(self.output_dir, "oof_predictions.joblib"))
        IO.save_pickle(metrics_summary, os.path.join(self.output_dir, "cv_metrics.joblib"))
        logger.info(f"OOF saved and metrics summary saved. Overall SMAPE={metrics_summary['smape']:.4f}%")
        return fold_models, oof, metrics_summary
