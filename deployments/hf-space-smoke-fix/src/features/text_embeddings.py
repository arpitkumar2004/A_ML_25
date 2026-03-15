# src/features/text_embeddings.py
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from scipy import sparse
from scipy.sparse import csr_matrix
import warnings
from typing import Optional, Tuple, Iterable
import os
import numpy as np
import joblib
from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory
from ..utils.fingerprint import stable_hash
from ..data.text_cleaning import TextCleaner
warnings.filterwarnings("ignore")

logger = LoggerFactory.get("text_embeddings")

# Try to import sentence-transformers; fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class TextEmbedder:
    """
    Create or load text embeddings. Supports two modes:
      - 'sbert' : sentence-transformers model (dense embeddings)
      - 'tfidf' : TF-IDF sparse features
    Usage:
      te = TextEmbedder(method='sbert', model_name='all-MiniLM-L6-v2', cache_path='data/processed/text_emb.npy')
      X = te.fit_transform(texts)
    """
    def __init__(self,
                 method: str = "sbert",
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_path: Optional[str] = "data/processed/text_embeddings.joblib",
                 vectorizer_path: Optional[str] = "data/processed/tfidf_vectorizer.joblib",
                 tfidf_max_features: int = 5000,
                 tfidf_ngram_range: Tuple[int,int] = (1,2)):
        self.method = method.lower()
        self.model_name = model_name
        self.cache_path = cache_path
        self.vectorizer_path = vectorizer_path
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self._model = None
        self._vectorizer = None

    def _load_cache(self, fingerprint: Optional[str] = None):
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached text features from {self.cache_path}")
            payload = joblib.load(self.cache_path)
            if isinstance(payload, dict) and "data" in payload and "fingerprint" in payload:
                if fingerprint is not None and payload.get("fingerprint") != fingerprint:
                    return None
                return payload.get("data")
            if fingerprint is not None:
                # Old cache format had no fingerprint, so avoid stale reuse.
                return None
            return payload
        return None

    def _save_cache(self, obj, fingerprint: Optional[str] = None):
        if not self.cache_path:
            return
        IO.save_pickle({"fingerprint": fingerprint, "data": obj}, self.cache_path)
        logger.info(f"Saved text features to {self.cache_path}")

    def _save_vectorizer(self):
        if self.method != "tfidf" or self._vectorizer is None or not self.vectorizer_path:
            return
        IO.save_pickle(self._vectorizer, self.vectorizer_path)

    def _load_vectorizer(self):
        if self._vectorizer is not None or not self.vectorizer_path:
            return
        if os.path.exists(self.vectorizer_path):
            self._vectorizer = IO.load_pickle(self.vectorizer_path)
            logger.info(f"Loaded TF-IDF vectorizer from {self.vectorizer_path}")

    def _init_sbert(self):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Install to use 'sbert' method.")
        if self._model is None:
            _hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if _hf_token:
                os.environ.setdefault("HF_TOKEN", _hf_token)
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)
            self._model = SentenceTransformer(self.model_name, token=_hf_token or None)
            logger.info(f"Loaded SentenceTransformer: {self.model_name}")

    def _init_tfidf(self, sample_texts: Iterable[str]):
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=self.tfidf_max_features,
                                               ngram_range=self.tfidf_ngram_range,
                                               stop_words='english')
            self._vectorizer.fit(sample_texts)
            logger.info("Fitted TF-IDF vectorizer")

    def _fingerprint(self, texts: Iterable[str], mode: str) -> str:
        sample = [str(t) for t in texts]
        payload = {
            "mode": mode,
            "method": self.method,
            "model_name": self.model_name,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_range": list(self.tfidf_ngram_range),
            "n": len(sample),
            "texts_hash": stable_hash(sample),
        }
        return stable_hash(payload)

    def fit_transform(self, texts: Iterable[str], use_cache: bool = True, fingerprint: Optional[str] = None):
        """
        Fit (if needed) and transform texts into embeddings or tfidf matrix.
        Returns dense np.ndarray (for sbert) or scipy csr_matrix (for tfidf).
        """
        texts = [TextCleaner.basic(t) for t in texts]
        fp = fingerprint or self._fingerprint(texts, mode="fit_transform")
        cached = self._load_cache(fp) if use_cache else None
        if cached is not None:
            return cached

        if self.method == "sbert":
            self._init_sbert()
            emb = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=64)
            self._save_cache(emb, fp)
            return emb
        elif self.method == "tfidf":
            self._init_tfidf(texts)
            X = self._vectorizer.transform(texts)
            self._save_vectorizer()
            self._save_cache(X, fp)
            return X
        else:
            raise ValueError(f"Unknown text embedding method: {self.method}")

    def transform(self, texts: Iterable[str], use_cache: bool = True, fingerprint: Optional[str] = None):
        # Pure transform assuming model/vectorizer already initialized
        texts = [TextCleaner.basic(t) for t in texts]
        fp = fingerprint or self._fingerprint(texts, mode="transform")
        cached = self._load_cache(fp) if use_cache else None
        if cached is not None:
            return cached

        if self.method == "sbert":
            if self._model is None:
                self._init_sbert()
            out = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=64)
            self._save_cache(out, fp)
            return out
        else:
            self._load_vectorizer()
            if self._vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer not fitted. Call fit_transform in training first.")
            out = self._vectorizer.transform(texts)
            self._save_cache(out, fp)
            return out


# Seperater line

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
            _hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2', token=_hf_tok or None)
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
            _hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            model = SentenceTransformer("all-MiniLM-L6-v2", token=_hf_tok or None)
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
