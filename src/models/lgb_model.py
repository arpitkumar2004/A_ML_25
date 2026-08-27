# src/models/lgb_model.py
from typing import Optional, Dict
from lightgbm import LGBMRegressor
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("LGBModel")


class LGBModel(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        params = params or {}

        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            pass

        default_params = {
            "objective": params.get("objective", "regression"),
            "learning_rate": params.get("learning_rate", 0.05),
            "n_estimators": params.get("n_estimators", 1000),
            "num_leaves": params.get("num_leaves", 31),
            "max_depth": params.get("max_depth", -1),
            "subsample": params.get("subsample", 1.0),
            "colsample_bytree": params.get("colsample_bytree", 1.0),
            "random_state": params.get("random_state", 42),
            "n_jobs": params.get("n_jobs", -1),
            "verbose": -1,
        }

        if has_gpu:
            default_params["device"] = "gpu"
            logger.info("⚡ LGBModel GPU Acceleration Enabled (device='gpu')")
        else:
            logger.info("ℹ️ LGBModel running in CPU mode.")

        default_params.update(params)
        self.model = LGBMRegressor(**default_params)
        logger.info(f"LGBModel init params: {default_params}")

    def fit(self, X, y, eval_set=None):
        try:
            if eval_set:
                try:
                    self.model.set_params(early_stopping_rounds=50)
                    self.model.fit(X, y, eval_set=eval_set)
                except (TypeError, Exception):
                    import lightgbm as lgb
                    self.model.fit(X, y, eval_set=eval_set, callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                self.model.fit(X, y)
        except Exception as e:
            logger.warning(f"LightGBM fit failed on primary device ({e}). Retrying with CPU fallback...")
            self.model.set_params(device="cpu", n_jobs=-1)
            if eval_set:
                try:
                    self.model.fit(X, y, eval_set=eval_set)
                except Exception:
                    self.model.fit(X, y)
            else:
                self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

