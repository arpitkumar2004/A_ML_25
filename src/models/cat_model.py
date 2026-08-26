# src/models/cat_model.py
from typing import Optional, Dict
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("CatModel")

try:
    from catboost import CatBoostRegressor
except Exception as e:
    CAT_AVAILABLE = False
    CAT_ERROR = e
else:
    CAT_AVAILABLE = True


class CatModel(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        if not CAT_AVAILABLE:
            raise ImportError("catboost is not installed. Install catboost to use CatModel.") from CAT_ERROR
        params = params or {}

        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            pass

        default_params = {
            "iterations": params.get("iterations", 1000),
            "learning_rate": params.get("learning_rate", 0.03),
            "depth": params.get("depth", 6),
            "eval_metric": params.get("eval_metric", "RMSE"),
            "random_seed": params.get("random_seed", 42),
            "verbose": False,
        }

        if has_gpu:
            default_params["task_type"] = "GPU"
            logger.info("⚡ CatModel GPU Acceleration Enabled (task_type='GPU')")
        else:
            logger.info("ℹ️ CatModel running in CPU mode.")

        default_params.update(params)
        self.model = CatBoostRegressor(**default_params)
        logger.info(f"CatModel init params: {default_params}")

    def fit(self, X, y, eval_set=None):
        try:
            if eval_set:
                self.model.fit(X, y, eval_set=eval_set, use_best_model=True)
            else:
                self.model.fit(X, y)
        except Exception as e:
            logger.warning(f"CatBoost fit failed on GPU ({e}). Retrying with CPU fallback...")
            self.model.set_params(task_type="CPU")
            if eval_set:
                self.model.fit(X, y, eval_set=eval_set, use_best_model=True)
            else:
                self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

