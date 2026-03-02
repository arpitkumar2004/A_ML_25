# src/training/utils_cv.py
from typing import Optional
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def make_folds(y, n_splits: int = 5, random_state: int = 42, stratify: bool = False):
    """
    Returns a generator of (train_idx, val_idx).
    For regression, stratify option creates bins from y and does StratifiedKFold on bins.
    """
    if stratify:
        # create bins (10 bins by quantiles)
        n_bins = min(10, max(2, n_splits))
        bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        # digitize
        y_binned = np.digitize(y, bins[1:-1], right=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf.split(np.zeros(len(y)), y_binned)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return kf.split(y)
