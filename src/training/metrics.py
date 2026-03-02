# src/utils/metrics.py
from typing import Union
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    return float(r2_score(y_true, y_pred))


def smape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Symmetric Mean Absolute Percentage Error (in percent)
      SMAPE = (100/N) * sum( |y_pred - y_true| / ((|y_true| + |y_pred|)/2) )
    We guard against zeros by adding a small epsilon in denominator.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    eps = 1e-8
    mask = denom < eps
    # For entries where denom is zero (both y_true and y_pred are 0), define error=0
    denom[mask] = eps
    diff = np.abs(y_pred - y_true) / denom
    return float(np.mean(diff) * 100.0)

def rmsle(y_true, y_pred):
    # handle log1p target optionally
    eps = 1e-9
    return np.sqrt(((np.log1p(y_pred + eps) - np.log1p(y_true + eps))**2).mean())

def adjusted_r2_score(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2


