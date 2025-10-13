import numpy as np

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # handle division by zero
    return np.mean(diff) * 100

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).mean())
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def rmsle(y_true, y_pred):
    # handle log1p target optionally
    eps = 1e-9
    return np.sqrt(((np.log1p(y_pred + eps) - np.log1p(y_true + eps))**2).mean())
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
def adjusted_r2_score(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2


