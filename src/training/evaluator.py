import numpy as np
from .metrics import rmse, rmsle, mae, smape, r2_score, adjusted_r2_score

def evaluate(y_true, y_pred):
    return {
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred)
        'adjusted_r2': adjusted_r2_score(y_true, y_pred, n_features=1)  # n_features should be set appropriately
    }


