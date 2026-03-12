# src/models/lgb_model.py
from typing import Optional, Dict
from lightgbm import LGBMRegressor
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("LGBModel")


class LGBModel(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        params = params or {}
        # Provide reasonable defaults; user can pass more in params
        self.model = LGBMRegressor(
            objective=params.get("objective", "regression"),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 1000),
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
        )
        logger.info(f"LGBModel init params: {params}")

    def fit(self, X, y, eval_set=None):
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
