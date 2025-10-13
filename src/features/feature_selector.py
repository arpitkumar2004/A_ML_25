import numpy as np


# placeholder for feature selection logic (L1, mutual_info, shap)
def select_features(X, y, names=None, k=100):
    # trivial: return all features and names
    return list(range(X.shape[1])), names

def merge_embeddings(df, embed_cols=None):
    """Merge all text adn image embedding columns into a new dataset."""
    if embed_cols is None:
        embed_cols = ['text_embeddings', 'image_embeddings']
    return np.concatenate([df[col].values for col in embed_cols], axis=1)
