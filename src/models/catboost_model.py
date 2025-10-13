# # Catboost Model code 

# import catboost as cb
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.model_selection import KFold
# import numpy as np
# import pandas as pd
# from src.training.metrics import smape
# from src.models.base_model import BaseModel, evaluate_model, get_feature_importance
# from src.utils.seed_everything import seed_everything
# import joblib

# class CatBoostModel(BaseModel):
#     def __init__(self):
#         self.model = cb.CatBoostRegressor()
#         self.is_fitted = False
#     def fit(self, X, y, X_val=None, y_val=None):
#         self.model.fit(X, y)
#         self.is_fitted = True
#         return self
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("Model is not fitted yet.")
#         return self.model.predict(X)
#     def feature_importances(self):
#         if hasattr(self.model, 'feature_importances_'):
#             return self.model.feature_importances_
#         else:
#             raise NotImplementedError("Feature importances not available for this model.")
#     def save(self, filepath):
#         joblib.dump(self.model, filepath)
#     def load(self, filepath):
#         self.model = joblib.load(filepath)
#         self.is_fitted = True
#         return self
#     def score(self, X, y):
#         if not self.is_fitted:
#             raise ValueError("Model is not fitted yet.")
#         return smape(y, self.predict(X))
    
# def cross_validate_model(model_class, X, y, cfg, metric):
#     folds = cfg['training'].get('cv_folds', 3)
#     seed = cfg['training'].get('random_seed', 42)
#     seed_everything(seed)
#     kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
#     scores = []
#     for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
#         X_tr, X_val = X[tr_idx], X[val_idx]
#         y_tr, y_val = y[tr_idx], y[val_idx]
#         model = model_class()
#         model.fit(X_tr, y_tr, X_val, y_val)
#         score = evaluate_model(model, X_val, y_val, metric)
#         print(f"Fold {fold} score: {score:.6f}")
#         scores.append(score)
#     return scores, np.mean(scores), np.std(scores)
# def get_feature_importance(model, feature_names):
#     """Get feature importance from the model."""
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         importances = model.coef_
#     else:
#         raise NotImplementedError("Feature importances not available for this model.")
    
#     return pd.DataFrame({
#         'feature': feature_names,
#         'importance': importances
#     }).sort_values(by='importance', ascending=False)
# def evaluate_model(model, X, y, metric):
#     """Evaluate model using the specified metric."""
#     y_pred = model.predict(X)
#     return metric(y, y_pred)
