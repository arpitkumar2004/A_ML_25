import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

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
# Token projection layer
# ---------------------------
class TokenProjection(nn.Module):
    def __init__(self, input_dim, token_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, token_dim)
    def forward(self, x):
        return self.proj(x)

# ---------------------------
# Transformer-based regressor
# ---------------------------
class TransformerRegressor(nn.Module):
    def __init__(self, token_dim, num_tokens, nhead=4, dim_feedforward=128, num_layers=1):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(token_dim, 1)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling over tokens
        out = self.fc_out(x)
        return out.squeeze(1)

# ---------------------------
# Prepare embeddings (projection + stacking)
# ---------------------------
def prepare_embeddings(df, feat_cols, token_dim=32, device='cuda'):
    projected_tokens = []
    for col in feat_cols:
        X_col = np.stack(df[col].values)  # (num_samples, col_dim)
        input_dim = X_col.shape[1]
        proj_layer = TokenProjection(input_dim, token_dim).to(device)
        with torch.no_grad():
            X_tensor = torch.tensor(X_col, dtype=torch.float32).to(device)
            projected = proj_layer(X_tensor)
        projected_tokens.append(projected)
    X_final = torch.stack(projected_tokens, dim=1)  # (num_samples, num_tokens, token_dim)
    return X_final

# ---------------------------
# Train Transformer
# ---------------------------
def train_transformer(df, feat_cols, target_col, token_dim=32, epochs=50, batch_size=32, lr=1e-3, device='cuda'):
    print("Preparing data...")
    X = prepare_embeddings(df, feat_cols, token_dim, device)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).to(device)

    # Train-test split
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    num_tokens = X_train.shape[1]
    model = TransformerRegressor(token_dim=token_dim, num_tokens=num_tokens).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training TransformerRegressor ...")
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # SMAPE evaluation every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test).cpu().numpy()
                score = smape(y_test.cpu().numpy(), y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train):.4f}, Test SMAPE: {score:.4f}%")
            model.train()

    return model, X_test, y_test

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Example dataset with variable-length embeddings
    # merged_df = pd.DataFrame({
    #     'image_embedding': [np.random.rand(128) for _ in range(200)],
    #     'measure_embedding': [np.random.rand(10) for _ in range(200)],
    #     'item_embedding_vector': [np.random.rand(64) for _ in range(200)],
    #     'price': np.random.rand(200) * 100
    # })

feat_cols = ['image_embedding', 'measure_embedding', 'item_embedding_vector']
target_col = 'price'

model, X_test, y_test = train_transformer(
    merged_df, feat_cols, target_col, token_dim=32, epochs=50, batch_size=32, lr=1e-3, device=device
)

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()
    final_smape = smape(y_test.cpu().numpy(), y_pred)
print(f"\nFinal SMAPE on test set: {final_smape:.4f}%")

# Save model
torch.save(model.state_dict(), "Transformer_price_model.pth")
print("Transformer model saved as Transformer_price_model.pth")
