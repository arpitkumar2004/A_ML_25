# Minimal stub for future BERT regressor
class BertRegressor:
    def __init__(self, cfg):
        self.cfg = cfg
    def fit(self, df):
        print("BertRegressor.fit() placeholder")
    def predict(self, df):
        import numpy as np
        return np.zeros(len(df))
