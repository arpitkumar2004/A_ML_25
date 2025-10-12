# placeholder for feature selection logic (L1, mutual_info, shap)
def select_features(X, y, names=None, k=100):
    # trivial: return all features and names
    return list(range(X.shape[1])), names
