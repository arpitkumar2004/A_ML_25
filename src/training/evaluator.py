import numpy as np
from .metrics import rmse, rmsle, mae

def evaluate(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred)
    }
