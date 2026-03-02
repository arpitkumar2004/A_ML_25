# text+image+tabular fusion neural net

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from src.utils.seed_everything import seed_everything
import joblib
import os
from src.training.metrics import smape
from src.models.base_model import BaseModel, evaluate_model, get_feature_importance
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.sparse import hstack, csr_matrix
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LinearRegression

class FusionDataset(Dataset):
    def __init__(self, texts, images, tabular, targets=None):
        self.texts = texts
        self.images = images
        self.tabular = tabular
        self.targets = targets
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            'text': torch.tensor(self.texts[idx], dtype=torch.float),
            'image': torch.tensor(self.images[idx], dtype=torch.float),
            'tabular': torch.tensor(self.tabular[idx], dtype=torch.float)
        }
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item
    
class FusionNN(BaseEstimator, RegressorMixin):
    def __init__(self, text_dim, image_dim, tabular_dim, hidden_dim=128, random_seed=42, lr=1e-3, batch_size=32, epochs=10):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.tabular_dim = tabular_dim
        self.hidden_dim = hidden_dim
        self.random_seed = random_seed
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        seed_everything(self.random_seed)
        
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.is_fitted = False
        
    def _build_model(self):
        class Net(nn.Module):
            def __init__(self, text_dim, image_dim, tabular_dim, hidden_dim):
                super(Net, self).__init__()
                self.text_fc = nn.Linear(text_dim, hidden_dim)
                self.image_fc = nn.Linear(image_dim, hidden_dim)
                self.tabular_fc = nn.Linear(tabular_dim, hidden_dim)
                self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 1)
                
            def forward(self, text, image, tabular):
                text_out = F.relu(self.text_fc(text))
                image_out = F.relu(self.image_fc(image))
                tabular_out = F.relu(self.tabular_fc(tabular))
                combined = torch.cat((text_out, image_out, tabular_out), dim=1)
                x = F.relu(self.fc1(combined))
                x = self.fc2(x)
                return x.squeeze()
        
        return Net(self.text_dim, self.image_dim, self.tabular_dim, self.hidden_dim)
    
    def fit(self, X_text, X_image, X_tabular, y):
        dataset = FusionDataset(X_text, X_image, X_tabular, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                texts = batch['text']
                images = batch['image']
                tabulars = batch['tabular']
                targets = batch['target']
                outputs = self.model(texts, images, tabulars)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * texts.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
        self.is_fitted = True
        return self
    
    def predict(self, X_text, X_image, X_tabular):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        dataset = FusionDataset(X_text, X_image, X_tabular)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                texts = batch['text']
                images = batch['image']
                tabulars = batch['tabular']
                outputs = self.model(texts, images, tabulars)
                preds.extend(outputs.cpu().numpy())
        return np.array(preds)
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'random_seed': self.random_seed
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.random_seed = checkpoint.get('random_seed', 42)
        seed_everything(self.random_seed)
        self.is_fitted = True
        return self
    
    def score(self, X_text, X_image, X_tabular, y):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.predict(X_text, X_image, X_tabular)
        return smape(y, preds)