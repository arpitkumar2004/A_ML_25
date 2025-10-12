import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings("ignore")


def build_text_embeddings(df, text_columns=None):

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s\.\-\/%]", " ", text)
        text = re.sub(r"\boz\b", "ounce", text)
        text = re.sub(r"\bpack of\b", "pack_of", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------- Helper: Build Embeddings -------------------
    def build_embeddings(df, text_col, prefix):
        print(f"🔹 Creating embeddings for {text_col}")

        df[text_col] = df[text_col].fillna("").apply(clean_text)

        # TF-IDF + SVD
        tfidf = TfidfVectorizer(
            max_features=500, ngram_range=(1, 2), stop_words="english"
        )
        X_tfidf = tfidf.fit_transform(df[text_col])
        svd = TruncatedSVD(n_components=min(50, X_tfidf.shape[1] - 1), random_state=42)
        X_tfidf = svd.fit_transform(X_tfidf)

        # Try SentenceTransformer, else fallback to random vectors
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            X_embed = model.encode(df[text_col].tolist(), show_progress_bar=False)
        except Exception:
            print("⚠️ SBERT not found, using random fallback embeddings (300-dim).")
            embed_dim = 300

            def rand_vec(seed):
                rng = np.random.RandomState(seed)
                return rng.randn(embed_dim)

            X_embed = np.vstack(
                [
                    np.mean(
                        [rand_vec(abs(hash(t)) % (2**32)) for t in s.split()]
                        or [np.zeros(embed_dim)],
                        axis=0,
                    )
                    for s in df[text_col]
                ]
            )

        X_embed = normalize(X_embed)
        combined = np.hstack([X_tfidf, X_embed])

        df[f"{prefix}_embedding_vector"] = [vec for vec in combined]

        print(f"✅ {text_col}: embedding dim {combined.shape[1]}")
        return df

    # ------------------- Main Loop -------------------
    if text_columns is None:
        text_columns = [
            ("item_name", "item"),
            ("bullet_points_str", "bullet"),
            ("product_description", "desc"),
        ]

    for col, prefix in text_columns:
        if col in df.columns:
            df = build_embeddings(df, col, prefix)
        else:
            print(
                f"⚠️ Column '{col}' not found — filling {prefix}_embedding_vector with NaN."
            )
            df[f"{prefix}_embedding_vector"] = np.nan

    print("\n✅ Final embedding columns added:")
    print([f"{prefix}_embedding_vector" for _, prefix in text_columns])
    return df
