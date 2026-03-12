# src/models/rf_model.py
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("RFModel")


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 200, max_depth: int = None, random_state: int = 42, n_jobs: int = -1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
        )
        logger.info(f"RandomForestModel init n_estimators={n_estimators}, max_depth={max_depth}")

    def fit(self, X, y, eval_set=None):
        logger.info("Fitting RandomForest")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
