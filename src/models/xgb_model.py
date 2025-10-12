"""XGBoost wrapper."""
import xgboost as xgb
class XGBModel:
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params).fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

