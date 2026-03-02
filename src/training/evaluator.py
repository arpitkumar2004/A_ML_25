# src/training/evaluator.py
from typing import Dict
from ..utils.metrics import rmse, mae, r2, smape
import numpy as np


def evaluate_preds(y_true, y_pred) -> Dict[str, float]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
