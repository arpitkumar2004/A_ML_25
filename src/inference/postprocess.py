"""Postprocessing outputs."""
def clamp_preds(preds, lower=0.0):
    import numpy as np
    return np.clip(preds, lower, None)

