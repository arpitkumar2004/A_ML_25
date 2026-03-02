import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
# from catboost import CatBoostRegressor
import umap
from tqdm import tqdm
from src.utils.seed_everything import seed_everything


# --- Helper functions ------------------------------------------------------

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def preprocess_text(df, text_col="description", max_features=1000):
    print("[INFO] Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_text = vectorizer.fit_transform(df[text_col].fillna(""))
    return X_text, vectorizer


def reduce_umap(X, n_components=32, random_state=42):
    print("[INFO] Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=10, min_dist=0.1)
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer


def train_base_models(X, y, models, n_splits=3, random_state=42):
    print("[INFO] Starting base model training...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros((len(y), len(models)))
    model_names = list(models.keys())
    scores = {}

    for i, (name, model) in enumerate(models.items()):
        fold_preds = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_preds[val_idx] = preds
        oof_preds[:, i] = fold_preds
        scores[name] = smape(y, fold_preds)
        print(f"{name} SMAPE: {scores[name]:.3f}")
    return oof_preds, scores


def train_stacker(oof_preds, y, n_splits=3, random_state=42):
    print("[INFO] Training stacker (Ridge)...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_meta = np.zeros(len(y))
    meta_model = Ridge(alpha=1.0)
    for train_idx, val_idx in kf.split(oof_preds, y):
        X_tr, X_val = oof_preds[train_idx], oof_preds[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        meta_model.fit(X_tr, y_tr)
        preds = meta_model.predict(X_val)
        oof_meta[val_idx] = preds
    score = smape(y, oof_meta)
    print(f"Stacker SMAPE: {score:.3f}")
    meta_model.fit(oof_preds, y)
    return meta_model, score


# --- Main run --------------------------------------------------------------

def main():
    seed_everything(42)
    print("[INFO] Loading dataset...")
    # Dummy data (replace with your own CSV)
    data = pd.DataFrame({
        "unique_identifier": range(1, 301),
        "description": ["Gift Basket Village Gourmet Meat and Cheese Set"] * 100 +
                       ["Premium Chocolate Box with Almonds"] * 100 +
                       ["Organic Tea Sampler Pack"] * 100,
        "price": np.random.uniform(15, 80, 300)
    })

    # Text embeddings (TF-IDF)
    X_text, vectorizer = preprocess_text(data, "description", max_features=500)
    X_text = X_text.toarray()

    # Dimensionality reduction (UMAP)
    X_umap, reducer = reduce_umap(X_text, n_components=16)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_umap)
    y = data["price"].values

    # Base models
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0),
        # "CatBoost": CatBoostRegressor(iterations=200, depth=4, learning_rate=0.1, verbose=False, random_seed=42)
    }

    # Train base models
    oof_preds, base_scores = train_base_models(X, y, models, n_splits=3)

    # Train stacker
    meta_model, stack_score = train_stacker(oof_preds, y)

    # Summary
    print("\n===== Model Performance (SMAPE) =====")
    summary = {**base_scores, "Stacked Ensemble": stack_score}
    for k, v in summary.items():
        print(f"{k:<15} : {v:.3f}")

    print("\n[INFO] Done ✅ – End-to-end pipeline executed successfully.")


if __name__ == "__main__":
    main()
