import numpy as np
def invert_log1p(preds):
    return np.expm1(preds)
