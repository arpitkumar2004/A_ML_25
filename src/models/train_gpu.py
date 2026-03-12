import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# GPU libraries
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import lightgbm as lgb

# ---------------------------
# SMAPE function
# ---------------------------
def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator[denominator == 0] = 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

# ---------------------------
# Flatten embeddings
# ---------------------------
def flatten_embeddings(df, feat_cols):
    print("Flattening embedding columns...")
    def flatten_row(row):
        flattened = []
        for col in feat_cols:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                flattened.extend(val)
            else:
                raise ValueError(f"Column {col} contains non-array values")
        return flattened
    X = df.apply(flatten_row, axis=1).to_list()
    X = np.array(X)
    print(f"Flattened features shape: {X.shape}")
    return X

# ---------------------------
# PyTorch MLP model
# ---------------------------
class MLPRegression(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)
        

def train_mlp_gpu(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, device='cuda'):
    input_dim = X_train.shape[1]
    model = MLPRegression(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            val_pred = model(X_val_tensor).detach().cpu().numpy().flatten()
            val_smape = smape(y_val, val_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val SMAPE: {val_smape:.4f}%")

    return model

# ---------------------------
# Train multiple models with GPU
# ---------------------------
def train_multiple_models_gpu(df, feat_cols, target_col, save_models=True):
    print("Starting model training pipeline (GPU-enabled)...")

    X = flatten_embeddings(df, feat_cols)
    y = df[target_col].astype(float).values
    print(f"Target variable shape: {y.shape}")

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Feature scaling for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Ridge Regression
    print("\nTraining Ridge Regression ...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    score = smape(y_test, y_pred_ridge)
    print(f"Ridge SMAPE: {score:.4f}%")
    if save_models: joblib.dump(ridge, "Ridge_price_model.pkl")
    results["Ridge"] = score

    # MLP with PyTorch on GPU
    print("\nTraining PyTorch MLP on GPU ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp_model = train_mlp_gpu(X_train_scaled, y_train, X_test_scaled, y_test, epochs=200, lr=1e-3, device=device)
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    score = smape(y_test, y_pred_mlp)
    print(f"MLP SMAPE: {score:.4f}%")
    if save_models: torch.save(mlp_model.state_dict(), "MLP_GPU_price_model.pth")
    results["MLP_GPU"] = score

    # XGBoost GPU
    print("\nTraining XGBoost GPU ...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        tree_method='gpu_hist', predictor='gpu_predictor', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    score = smape(y_test, y_pred_xgb)
    print(f"XGBoost GPU SMAPE: {score:.4f}%")
    if save_models: joblib.dump(xgb_model, "XGBoost_GPU_price_model.pkl")
    results["XGBoost_GPU"] = score

    # LightGBM GPU
    print("\nTraining LightGBM GPU ...")
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=-1, device='gpu', random_state=42)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    score = smape(y_test, y_pred_lgb)
    print(f"LightGBM GPU SMAPE: {score:.4f}%")
    if save_models: joblib.dump(lgb_model, "LightGBM_GPU_price_model.pkl")
    results["LightGBM_GPU"] = score

    print("\nAll models trained and evaluated.")
    return results

# ---------------------------
# Example usage
# ---------------------------
    # # Example dataset with random embeddings
    # merged_df = pd.DataFrame({
    #     'image_embedding': [np.random.rand(128) for _ in range(100)],
    #     'measure_embedding': [np.random.rand(10) for _ in range(100)],
    #     'item_embedding_vector': [np.random.rand(64) for _ in range(100)],
    #     'price': np.random.rand(100) * 100
    # })

feat_cols = ['image_embedding', 'measure_embedding', 'item_embedding_vector']
target_col = 'price'

results = train_multiple_models_gpu(merged_df, feat_cols, target_col, save_models=True)

print("\nSMAPE results for all models:")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}%")
