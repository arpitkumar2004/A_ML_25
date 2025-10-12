# stub for text+image+tabular fusion neural net (PyTorch/TF)
class FusionNet:
    def __init__(self, cfg):
        self.cfg = cfg
    def fit(self, X_text, X_tab, y):
        print("FusionNet.fit() placeholder")
    def predict(self, X_text, X_tab):
        import numpy as np
        return np.zeros(X_tab.shape[0])
