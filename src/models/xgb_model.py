# placeholder - similar wrapper for xgboost
class XGBModel:
    def __init__(self, cfg):
        self.cfg = cfg
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
