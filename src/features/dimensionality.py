# src/features/dimensionality.py
from typing import Optional, Tuple
import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..utils.fingerprint import stable_hash

logger = LoggerFactory.get("dimensionality")


try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


class DimReducer:
    """
    Wrapper to apply PCA or UMAP dimensionality reduction and cache results.
    Usage:
      dr = DimReducer(method='pca', n_components=50, cache_path='data/processed/pca.joblib')
      Xr = dr.fit_transform(X)
    """
    def __init__(self, method: str = "pca", n_components: int = 50, cache_path: Optional[str] = "data/processed/dimred.joblib", random_state: int = 42):
        self.method = method.lower()
        self.n_components = n_components
        self.cache_path = cache_path
        self.random_state = random_state
        self.model = None

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached dim reducer from {self.cache_path}")
            return joblib.load(self.cache_path)
        return None

    def _save_cache(self, obj):
        if not self.cache_path:
            return
        IO.save_pickle(obj, self.cache_path)
        logger.info(f"Saved dim reducer to {self.cache_path}")

    def _build_fingerprint(self, X: np.ndarray, user_fingerprint: Optional[str] = None) -> str:
        if user_fingerprint is not None:
            return user_fingerprint
        stats = {
            "rows": int(X.shape[0]),
            "cols": int(X.shape[1]),
            "dtype": str(X.dtype),
            "mean": float(np.mean(X)) if X.size else 0.0,
            "std": float(np.std(X)) if X.size else 0.0,
        }
        payload = {
            "method": self.method,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "stats": stats,
        }
        return stable_hash(payload)

    def fit_transform(self, X: np.ndarray, use_cache: bool = True, fingerprint: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Fit transformer and return transformed X and diagnostics.
        If use_cache and cache exists, returns cached transformer result.
        """
        fp = self._build_fingerprint(X, fingerprint)
        if use_cache and os.path.exists(self.cache_path):
            data = self._load_cache()
            if isinstance(data, dict) and data.get("fingerprint") == fp:
                if "X_reduced" in data:
                    return data["X_reduced"], data.get("meta", {})
                if "model" in data:
                    self.model = data["model"]
                    return self.model.transform(X), data.get("meta", {})

        if self.method == "pca":
            self.model = PCA(n_components=self.n_components, random_state=self.random_state)
            Xr = self.model.fit_transform(X)
            meta = {"explained_variance_ratio": self.model.explained_variance_ratio_.tolist()}
        elif self.method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Install umap-learn to use method='umap'.")
            self.model = umap.UMAP(n_components=self.n_components, random_state=self.random_state)
            Xr = self.model.fit_transform(X)
            meta = {}
        else:
            raise ValueError(f"Unknown reducer: {self.method}")

        self._save_cache({"fingerprint": fp, "model": self.model, "X_reduced": Xr, "meta": meta})
        return Xr, meta

    def transform(self, X: np.ndarray):
        if self.model is None:
            if self.cache_path and os.path.exists(self.cache_path):
                data = self._load_cache()
                self.model = data["model"]
            else:
                raise RuntimeError("Reducer not fitted.")
        return self.model.transform(X)
