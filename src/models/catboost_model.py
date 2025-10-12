# placeholder wrapper for CatBoost
class CatBoostModel:
    def __init__(self, cfg):
        self.cfg = cfg
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
