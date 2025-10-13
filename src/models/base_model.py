import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


class BaseModel:
    """Base model linear regression class."""
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importances(self):
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        else:
            raise NotImplementedError("Feature importances not available for this model.")
    def save(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        import joblib
        self.model = joblib.load(filepath)
        return self
        
    def score(self, X, y):
        return self.model.score(X, y)
    
def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
    else:
        raise NotImplementedError("Feature importances not available for this model.")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
def evaluate_model(model, X, y, metric):
    """Evaluate model using the specified metric."""
    y_pred = model.predict(X)
    return metric(y, y_pred)

def cross_validate_model(model_class, X, y, cfg, metric):
    folds = cfg['training'].get('cv_folds', 3)
    kf = KFold(n_splits=folds, shuffle=True, random_state=cfg['training'].get('random_seed', 42))
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_class()
        model.fit(X_tr, y_tr)
        score = evaluate_model(model, X_val, y_val, metric)
        scores.append(score)
        print(f"Fold {fold} score: {score:.6f}")
    return np.mean(scores), np.std(scores)
