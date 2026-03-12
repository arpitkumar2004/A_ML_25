# src/models/stacker.py
"""
Stacker module with meta-level KFold CV.
Supports Ridge and LightGBM as meta-learners.
Evaluates stability and avoids overfitting on OOF features.
"""
from typing import Optional, Any, Dict
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from ..utils.logging_utils import LoggerFactory
from ..training.metrics import rmse, mae, r2, smape
from ..utils.io import IO

logger = LoggerFactory.get("stacker")

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False


class Stacker:
    def __init__(
        self,
        method: str = "ridge",
        params: Optional[Dict] = None,
        n_splits: int = 5,
        random_state: int = 42,
        save_path: str = "experiments/models/stacker.joblib",
        save_oof_path: str = "experiments/oof/meta_oof.joblib"
    ):
        self.method = method.lower()
        self.params = params or {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.save_path = save_path
        self.save_oof_path = save_oof_path
        self.model: Any = None
        self._init_model()

    def _init_model(self):
        if self.method == "ridge":
            alpha = self.params.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha)
            logger.info(f"Initialized Ridge meta-learner (alpha={alpha})")
        elif self.method == "lgbm":
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM not available for stacking meta-learner.")
            p = self.params.copy()
            self.model = LGBMRegressor(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.05),
                random_state=p.get("random_state", self.random_state),
                num_leaves=p.get("num_leaves", 31)
            )
            logger.info(f"Initialized LGBM meta-learner with params={p}")
        else:
            raise ValueError(f"Unknown stacker method: {self.method}")

    def fit_cv(self, OOF: np.ndarray, y: np.ndarray, fit_final: bool = True) -> Dict[str, float]:
        """
        Fit meta-model using KFold CV on base OOF predictions.

        Args:
            OOF: np.ndarray of shape (n_samples, n_base_models)
            y: target array
            fit_final: if True, fits the model on full OOF after CV
        Returns:
            mean_metrics: dict with mean + std across folds
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_meta = np.zeros_like(y, dtype=float)
        metrics_list = []

        logger.info(f"Starting meta-level CV ({self.n_splits} folds) with method={self.method}")

        for fold, (tr_idx, val_idx) in enumerate(kf.split(OOF)):
            X_tr, X_val = OOF[tr_idx], OOF[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = self._get_model_instance()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            oof_meta[val_idx] = preds

            fold_metrics = {
                "rmse": rmse(y_val, preds),
                "mae": mae(y_val, preds),
                "r2": r2(y_val, preds),
                "smape": smape(y_val, preds)
            }
            metrics_list.append(fold_metrics)
            logger.info(f"Fold {fold+1}/{self.n_splits}: {fold_metrics}")

        IO.save_pickle(oof_meta, self.save_oof_path)
        logger.info(f"Saved meta OOF preds to {self.save_oof_path}")

        # aggregate metrics
        mean_metrics = {
            k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]
        }
        std_metrics = {
            k + "_std": float(np.std([m[k] for m in metrics_list])) for k in metrics_list[0]
        }
        summary = {**mean_metrics, **std_metrics}
        logger.info(f"Meta-level CV metrics: {summary}")

        # fit on full OOF for final model
        if fit_final:
            self.model.fit(OOF, y)
            IO.save_pickle(self.model, self.save_path)
            logger.info(f"Final meta-model trained and saved to {self.save_path}")

        return summary

    def _get_model_instance(self):
        """Creates a fresh copy of the base meta-model."""
        if self.method == "ridge":
            return Ridge(alpha=self.params.get("alpha", 1.0))
        elif self.method == "lgbm":
            return LGBMRegressor(
                n_estimators=self.params.get("n_estimators", 300),
                learning_rate=self.params.get("learning_rate", 0.05),
                random_state=self.random_state,
                num_leaves=self.params.get("num_leaves", 31)
            )

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        return self.model.predict(meta_X)

    def load(self, path: Optional[str] = None):
        p = path or self.save_path
        self.model = IO.load_pickle(p)
        logger.info(f"Loaded stacker model from {p}")
