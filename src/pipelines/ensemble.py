import pandas as pd
import torch
import joblib
import numpy as np

def ensemble_predict(models_dict, df_test, feat_cols, method='mean', weights=None, mlp_device='cuda'):
    """
    models_dict: dictionary of trained models, e.g. {"Ridge": ridge_model, "MLP_GPU": mlp_model, ...}
    df_test: test DataFrame containing embeddings
    feat_cols: list of embedding columns
    method: 'mean' or 'weighted'
    weights: dictionary of weights per model (used if method='weighted')
    mlp_device: device for PyTorch MLP
    """
    # Flatten embeddings for sklearn/XGBoost/LightGBM
    def flatten_embeddings_row(row):
        flattened = []
        for col in feat_cols:
            flattened.extend(row[col])
        return flattened
    X_test_flat = np.array(df_test.apply(flatten_embeddings_row, axis=1).to_list())

    predictions = []

    for name, model in models_dict.items():
        if name == 'MLP_GPU':
            # PyTorch MLP
            X_tensor = torch.tensor(X_test_flat, dtype=torch.float32).to(mlp_device)
            model.eval()
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy().flatten()
        else:
            # Ridge, XGBoost, LightGBM
            y_pred = model.predict(X_test_flat)

        predictions.append(y_pred)

    predictions = np.array(predictions)  # shape: (num_models, num_samples)

    # Combine predictions
    if method == 'mean':
        final_pred = predictions.mean(axis=0)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Weights dictionary must be provided for weighted averaging")
        w = np.array([weights.get(name, 1.0) for name in models_dict.keys()])
        w = w / w.sum()
        final_pred = np.average(predictions, axis=0, weights=w)
    else:
        raise ValueError("method must be 'mean' or 'weighted'")

    # Create output DataFrame
    output_df = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': final_pred
    })

    return output_df

fiolder_path = '/content/drive/MyDrive/amazon ml challenge files/models'

ridge_model = joblib.load(os.path.join(fiolder_path, 'Ridge_price_model.pkl'))
mlp_model = torch.load(os.path.join(fiolder_path, 'price_nn_model.pth'))
xgb_model = joblib.load(os.path.join(fiolder_path, 'XGBoost_price_model.pkl'))
lgb_model = joblib.load(os.path.join(fiolder_path, 'LightGBM_price_model.pkl'))


# Assuming you already have your trained models loaded or in memory
models_dict = {"Ridge": ridge_model, "MLP_GPU": mlp_model, "XGBoost_GPU": xgb_model, "LightGBM_GPU": lgb_model}

# For demonstration, let's assume models_dict is already defined
final_df = ensemble_predict(models_dict, X, feat_col, method='mean', mlp_device='cuda')

# Export to CSV
final_df.to_csv("ensemble_predictions.csv", index=False)
print("Predictions saved to ensemble_predictions.csv")
