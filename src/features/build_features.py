import numpy as np
# from .text_embeddings import fit_tfidf, load_tfidf
from .numeric_features import build_numeric_features
from src.data.text_cleaning import clean_text

def build_features_for_train(df, cfg):
    texts = df[cfg['data']['text_col']].astype(str).apply(clean_text).tolist()
    tfidf_cfg = cfg.get('features', {}).get('tfidf', {})
    X_text, vect = fit_tfidf(texts,
                             max_features=tfidf_cfg.get('max_features', 5000),
                             ngram_range=tuple(tfidf_cfg.get('ngram_range', (1,2))))
    X_num = build_numeric_features(df)
    # combine
    from scipy.sparse import hstack, csr_matrix
    X = hstack([X_text, csr_matrix(X_num)])
    return X, vect

def build_features_for_inference(df, cfg, vect):
    texts = df[cfg['data']['text_col']].astype(str).apply(clean_text).tolist()
    X_text = load_tfidf(texts, vect)
    X_num = build_numeric_features(df)
    # combine
    from scipy.sparse import hstack, csr_matrix
    X = hstack([X_text, csr_matrix(X_num)])
    return X
