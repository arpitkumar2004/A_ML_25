"""Stacking / blending utilities."""
def blend(preds_list, weights=None):
    import numpy as np
    if weights is None:
        weights = [1/len(preds_list)] * len(preds_list)
    return sum(w * p for w,p in zip(weights, preds_list))

