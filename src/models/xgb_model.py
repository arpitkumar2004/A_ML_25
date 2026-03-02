# src/models/xgb_model.py
from typing import Optional, Dict
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("XGBModel")

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGB_AVAILABLE = False
    XGB_ERROR = e
else:
    XGB_AVAILABLE = True


class XGBModel(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is not installed. Install xgboost to use XGBModel.") from XGB_ERROR
        params = params or {}
        self.model = XGBRegressor(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            objective="reg:squarederror",
        )
        logger.info(f"XGBModel init params: {params}")

    def fit(self, X, y, eval_set=None):
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
