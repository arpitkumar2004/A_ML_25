"""CatBoost wrapper."""
from catboost import CatBoostRegressor
class CatModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = CatBoostRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

