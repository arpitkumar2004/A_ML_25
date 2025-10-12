import numpy as np
from sklearn.model_selection import KFold
from ..models.lgbm_model import LGBMModel
from ..utils.seed_everything import seed_everything
from ..training.metrics import rmsle
import os

def train_lgbm_cv(X, y, cfg):
    folds = cfg['training'].get('cv_folds', 3)
    seed = cfg['training'].get('random_seed', 42)
    seed_everything(seed)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model_cfg = cfg['model']
        model = LGBMModel(params=cfg["model"]["params"])
        model.fit(X_tr, y_tr, X_val, y_val)
        preds = model.predict(X_val)
        oof[val_idx] = preds
        print(f"Fold {fold} score: {rmsle(y_val, preds):.6f}")
        models.append(model)
    full_score = rmsle(y, oof)
    print(f"OOF Score: {full_score:.6f}")
    # save models directory
    os.makedirs("experiments/models", exist_ok=True)
    return models, oof
