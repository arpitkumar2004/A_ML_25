from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("feature_selector")


class FeatureSelector:
    """Persisted feature selector with train/inference parity."""

    def __init__(
        self,
        method: str = "mutual_info",
        k: int = 512,
        min_features: int = 32,
        save_path: str = "data/processed/feature_selector.joblib",
        random_state: int = 42,
    ):
        self.method = method.lower()
        self.k = int(k)
        self.min_features = int(min_features)
        self.save_path = save_path
        self.random_state = random_state

        self.selected_indices_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def _effective_k(self, n_features: int) -> int:
        if n_features <= 0:
            return 0
        return max(self.min_features, min(self.k, n_features))

    def _save(self) -> None:
        payload = {
            "method": self.method,
            "k": self.k,
            "min_features": self.min_features,
            "selected_indices": self.selected_indices_,
            "scores": self.scores_,
            "feature_names": self.feature_names_,
        }
        IO.save_pickle(payload, self.save_path)

    def _load(self) -> None:
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(self.save_path)
        payload = IO.load_pickle(self.save_path)
        self.method = payload.get("method", self.method)
        self.k = int(payload.get("k", self.k))
        self.min_features = int(payload.get("min_features", self.min_features))
        self.selected_indices_ = np.asarray(payload.get("selected_indices"), dtype=int)
        self.scores_ = payload.get("scores")
        self.feature_names_ = payload.get("feature_names")

    def fit_transform(
        self,
        X: Any,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if y is None:
            raise ValueError("FeatureSelector.fit_transform requires target y.")

        n_features = X.shape[1]
        k_eff = self._effective_k(n_features)
        if k_eff <= 0:
            self.selected_indices_ = np.array([], dtype=int)
            self.scores_ = np.array([])
            self.feature_names_ = feature_names
            self._save()
            return X, {"selected_count": 0, "total_features": n_features, "method": self.method}

        # f_regression is robust and sparse-friendly; mutual_info can be unstable on sparse high-dim inputs.
        if sparse.issparse(X) or self.method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=k_eff)
            X_selected = selector.fit_transform(X, y)
            scores = selector.scores_
        elif self.method == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=self.random_state)
            idx = np.argsort(np.nan_to_num(scores, nan=-1.0))[-k_eff:]
            idx = np.sort(idx)
            X_selected = X[:, idx]
            self.selected_indices_ = idx
            self.scores_ = np.asarray(scores)
            self.feature_names_ = feature_names
            self._save()
            return X_selected, {
                "selected_count": int(len(idx)),
                "total_features": int(n_features),
                "method": self.method,
            }
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        idx = selector.get_support(indices=True)
        self.selected_indices_ = np.asarray(idx, dtype=int)
        self.scores_ = np.asarray(scores) if scores is not None else None
        self.feature_names_ = feature_names
        self._save()
        return X_selected, {
            "selected_count": int(len(idx)),
            "total_features": int(n_features),
            "method": self.method,
        }

    def transform(self, X: Any) -> Any:
        if self.selected_indices_ is None:
            self._load()
        if self.selected_indices_ is None or len(self.selected_indices_) == 0:
            return X
        if sparse.issparse(X):
            return X[:, self.selected_indices_]
        return np.asarray(X)[:, self.selected_indices_]


def select_features(X, y, names=None, k=100):
    """Backwards-compatible wrapper used by legacy notebooks/scripts."""
    selector = FeatureSelector(method="f_regression", k=k)
    _, meta = selector.fit_transform(X, y, feature_names=names)
    selected = selector.selected_indices_.tolist() if selector.selected_indices_ is not None else []
    logger.info("Selected %s/%s features", meta.get("selected_count"), meta.get("total_features"))
    selected_names = [names[i] for i in selected] if names else names
    return selected, selected_names


def merge_embeddings(df, embed_cols=None):
    """Merge all text and image embedding columns into a new dense dataset."""
    if embed_cols is None:
        embed_cols = ["text_embeddings", "image_embeddings"]
    return np.concatenate([df[col].values for col in embed_cols], axis=1)
