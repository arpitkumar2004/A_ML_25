"""LightGBM wrapper."""
import lightgbm as lgb
class LGBMModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

