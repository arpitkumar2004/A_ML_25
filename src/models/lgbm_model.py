import lightgbm as lgb
import numpy as np
import os
import joblib
from .base_model import BaseModel

MODEL_PATH = "experiments/models/lgbm_model.pkl"

class LGBMModel(BaseModel):
    def __init__(self, params=None, num_boost_round=1000, early_stopping_rounds=50):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y, valid_sets=None):
        dtrain = lgb.Dataset(X, label=y)
        valid = []
        if valid_sets:
            for Xv, yv in valid_sets:
                valid.append(lgb.Dataset(Xv, label=yv))
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round,
                               valid_sets=valid if valid else None,
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=False)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
