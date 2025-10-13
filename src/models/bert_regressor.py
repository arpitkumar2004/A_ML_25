# Bert regressor model using transformers
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from src.training.metrics import smape
from src.utils.seed_everything import seed_everything
from sklearn.model_selection import KFold
import joblib
import os

class BertRegressor(BaseModel):
    def __init__(self, pretrained_model_name='bert-base-uncased', random_seed=42):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.regressor = LinearRegression()
        self.random_seed = random_seed
        seed_everything(self.random_seed)
        self.is_fitted = False

    def fit(self, X, y):
        self.regressor.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, texts):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        embeddings = self._get_bert_embeddings(texts)
        return self.regressor.predict(embeddings)
    
    def _get_bert_embeddings(self, texts):
        self.bert.eval()
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = self.bert(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(cls_embedding)
        return np.array(embeddings)
    
    def feature_importances(self):
        if hasattr(self.regressor, 'coef_'):
            return self.regressor.coef_
        else:
            raise NotImplementedError("Feature importances not available for this model.")
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'regressor': self.regressor,
            'random_seed': self.random_seed
        }, filepath)
        
    def load(self, filepath):
        data = joblib.load(filepath)
        self.regressor = data['regressor']
        self.random_seed = data.get('random_seed', 42)
        seed_everything(self.random_seed)
        self.is_fitted = True
        return self
    
    def score(self, texts, y):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.predict(texts)
        return smape(y, preds)
    
def cross_validate_model(model_class, texts, y, cfg, metric):
    folds = cfg['training'].get('cv_folds', 3)
    seed = cfg['training'].get('random_seed', 42)
    seed_everything(seed)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(texts)):
        X_tr, X_val = texts[tr_idx], texts[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_class(random_seed=seed)
        model.fit(model._get_bert_embeddings(X_tr), y_tr)
        score = evaluate_model(model, X_val, y_val, metric)
        print(f"Fold {fold} score: {score:.6f}")
        scores.append(score)
    return scores, np.mean(scores), np.std(scores)

def evaluate_model(model, texts, y, metric):
    """Evaluate model using the specified metric."""
    y_pred = model.predict(texts)
    return metric(y, y_pred)

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
    