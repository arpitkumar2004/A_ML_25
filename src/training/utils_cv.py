from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
def get_cv(cfg, y=None, groups=None):
    folds = cfg['training'].get('cv_folds', 5)
    return KFold(n_splits=folds, shuffle=True, random_state=cfg['training'].get('random_seed', 42))

