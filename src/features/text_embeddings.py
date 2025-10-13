import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

def embeded_text_features(df):
    """
    Generate text embeddings and related features for specified text columns in the dataframe.
    Uses TF-IDF, SVD, and SentenceTransformer (if available) for embeddings.
    Falls back to random vectors if SentenceTransformer is not available.
    
    Args:
        df (pd.DataFrame): Input dataframe with text columns.
    Returns:
        pd.DataFrame: Dataframe with new embedding features added.
    """
    # ===========================
    # 1️⃣ Text Cleaning Function
    # ===========================
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s\.\-\/%]", " ", text)
        text = re.sub(r"\boz\b", "ounce", text)
        text = re.sub(r"\bpack of\b", "pack_of", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ===========================
    # 2️⃣ Function to process a text column
    # ===========================
    def process_text_column(df, text_col, prefix):
        print(f"🔹 Processing column: {text_col}")

        # Clean
        df[text_col] = df[text_col].fillna("").apply(clean_text)

        # Text stats
        df[f'{prefix}_num_words'] = df[text_col].apply(lambda s: len(s.split()))
        df[f'{prefix}_num_chars'] = df[text_col].apply(len)
        df[f'{prefix}_num_sentences'] = df[text_col].apply(lambda s: s.count(';') + s.count('.') + 1 if s else 0)

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        X_tfidf_full = tfidf.fit_transform(df[text_col])

        # SVD (dimensionality reduction)
        svd = TruncatedSVD(n_components=min(50, X_tfidf_full.shape[1]-1), random_state=42)
        X_tfidf = svd.fit_transform(X_tfidf_full)

        # Sentence Embeddings (SBERT or fallback)
        try:
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            X_embed = sbert_model.encode(df[text_col].tolist(), show_progress_bar=False)
            embedding_method = "sentence-transformer"
        except Exception as e:
            print("SBERT not available, using random fallback embeddings.")
            embed_dim = 300
            def avg_embed_fallback(text):
                tokens = text.split()
                if not tokens:
                    return np.zeros(embed_dim)
                vecs = [np.random.RandomState(hash(t) % (2**32)).randn(embed_dim) for t in tokens]
                return np.mean(vecs, axis=0)
            X_embed = np.vstack([avg_embed_fallback(t) for t in df[text_col]])
            embedding_method = "random-fallback"

        X_embed = normalize(X_embed)

        # Combine TF-IDF + Embedding (optional)
        X_combined = np.hstack([X_tfidf, X_embed])

        # Add final combined embedding as one column
        df[f'{prefix}_embedding_vector'] = [vec for vec in X_combined]

        print(f"✅ Finished {text_col} | TF-IDF dim: {X_tfidf.shape[1]} | Embedding dim: {X_embed.shape[1]} | Combined dim: {X_combined.shape[1]}")
        return df

    # ===========================
    # 4️⃣ Apply to all text columns
    # ===========================
    text_columns = [
        ('item_name', 'item'),
        ('bullet_points_str', 'bullet'),
        ('product_description', 'desc')
    ]

    for col, prefix in text_columns:
        if col in df.columns:
            df = process_text_column(df, col, prefix)
        else:
            df[f'{prefix}_embedding_vector'] = np.nan  # if column missing


    print("\n🎯 Final dataframe columns:")
    print(df.columns)
    print("\n✅ All text columns converted into embedding vectors and added to df.")
    
    return df

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
