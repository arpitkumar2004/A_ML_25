"""Evaluation and CV helpers."""
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    import numpy as np
    return float(mean_squared_error(y_true, y_pred, squared=False))

