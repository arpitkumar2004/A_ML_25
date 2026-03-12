# src/models/linear_model.py
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("LinearModel")


class LinearModel(BaseModel):
    def __init__(self, fit_intercept: bool = True, normalize: bool = False):
        self.model = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
        self.normalize = normalize

    def fit(self, X, y, eval_set=None):
        logger.info("Fitting LinearRegression")
        self.model.fit(X, y)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

class RidgeRegression(BaseModel):
    def __init__(self, alpha: float = 1.0, normalize: bool = False):
        self.model = Ridge(alpha=alpha)
        self.normalize = normalize
        
    def fit(self, X, y, eval_set=None):
        logger.info("Fitting RidgeRegression")
        self.model.fit(X, y)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

class LassoRegression(BaseModel):
    def __init__(self, alpha: float = 1.0, normalize: bool = False):
        self.model = Lasso(alpha=alpha)
        self.normalize = normalize
        
    def fit(self, X, y, eval_set=None):
        logger.info("Fitting LassoRegression")
        self.model.fit(X, y)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds