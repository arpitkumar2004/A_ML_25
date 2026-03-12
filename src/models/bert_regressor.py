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
        _hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, token=_hf_token)
        self.bert = BertModel.from_pretrained(pretrained_model_name, token=_hf_token)
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