from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
from scipy import sparse
from scipy.stats import skewtest

from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("post_feature_log_transform")


class PostFeatureLogTransformer:
    """Apply a statistically-gated log1p transform on final features.

    Train mode:
      - tests each feature column with skewness test (H0: skew == 0)
      - applies log1p only when column is non-negative and significantly right-skewed
      - persists selected column indices for inference parity

    Inference mode:
      - loads persisted selected indices and applies identical log1p transform
    """

    def __init__(
        self,
        enabled: bool = False,
        alpha: float = 0.01,
        min_skew: float = 1.0,
        min_samples: int = 30,
        max_test_rows: int = 50000,
        save_path: str = "data/processed/post_feature_log_transform.joblib",
        random_state: int = 42,
    ):
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.min_skew = float(min_skew)
        self.min_samples = int(min_samples)
        self.max_test_rows = int(max_test_rows)
        self.save_path = save_path
        self.random_state = int(random_state)

        self.selected_indices_: Optional[np.ndarray] = None
        self.stats_: Optional[Dict[str, Any]] = None

    def _save(self) -> None:
        payload = {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "min_skew": self.min_skew,
            "min_samples": self.min_samples,
            "max_test_rows": self.max_test_rows,
            "selected_indices": self.selected_indices_,
            "stats": self.stats_,
        }
        IO.save_pickle(payload, self.save_path)

    def _load(self) -> None:
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(self.save_path)
        payload = IO.load_pickle(self.save_path)
        self.selected_indices_ = np.asarray(payload.get("selected_indices", []), dtype=int)
        self.stats_ = payload.get("stats", {})

    def _to_dense_for_testing(self, X: Any) -> np.ndarray:
        if sparse.issparse(X):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("PostFeatureLogTransformer expects 2D feature matrix.")
        return X_arr

    def _sample_rows(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        if n_rows <= self.max_test_rows:
            return X
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(n_rows, size=self.max_test_rows, replace=False)
        return X[idx]

    def _choose_columns(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        Xs = self._sample_rows(X)
        n_rows, n_cols = Xs.shape
        if n_rows < max(self.min_samples, 8):
            return np.array([], dtype=int), {
                "selected_count": 0,
                "total_features": int(n_cols),
                "tested_rows": int(n_rows),
                "skipped_reason": "insufficient_rows",
            }

        selected: List[int] = []
        tested_cols = 0
        skipped_negative = 0
        skipped_constant = 0
        failed_tests = 0

        for j in range(n_cols):
            col = Xs[:, j]
            col = col[np.isfinite(col)]
            if col.size < max(self.min_samples, 8):
                continue
            if np.min(col) < 0:
                skipped_negative += 1
                continue
            if np.allclose(col, col[0]):
                skipped_constant += 1
                continue

            tested_cols += 1
            try:
                z_stat, p_value = skewtest(col)
            except Exception:
                failed_tests += 1
                continue

            col_mean = float(np.mean(col))
            col_std = float(np.std(col)) + 1e-12
            col_skew = float(np.mean(((col - col_mean) / col_std) ** 3))
            if (p_value < self.alpha) and (col_skew >= self.min_skew):
                selected.append(j)

        return np.asarray(selected, dtype=int), {
            "selected_count": int(len(selected)),
            "total_features": int(n_cols),
            "tested_rows": int(n_rows),
            "tested_columns": int(tested_cols),
            "skipped_negative_columns": int(skipped_negative),
            "skipped_constant_columns": int(skipped_constant),
            "failed_tests": int(failed_tests),
            "alpha": self.alpha,
            "min_skew": self.min_skew,
            "method": "skewtest+log1p",
        }

    @staticmethod
    def _apply_log1p(X: Any, indices: np.ndarray) -> Any:
        if indices.size == 0:
            return X
        if sparse.issparse(X):
            X = X.tolil(copy=True)
            for idx in indices.tolist():
                col = X[:, idx].toarray().ravel()
                col = np.log1p(np.clip(col, a_min=0.0, a_max=None))
                X[:, idx] = col.reshape(-1, 1)
            return X.tocsr()

        X_out = np.asarray(X).copy()
        X_out[:, indices] = np.log1p(np.clip(X_out[:, indices], a_min=0.0, a_max=None))
        return X_out

    def fit_transform(self, X: Any) -> Tuple[Any, Dict[str, Any]]:
        if not self.enabled:
            return X, {"enabled": False, "applied": False}

        X_dense = self._to_dense_for_testing(X)
        selected, stats = self._choose_columns(X_dense)
        self.selected_indices_ = selected
        self.stats_ = stats
        self._save()

        X_out = self._apply_log1p(X, selected)
        logger.info(
            "Post-feature log transform selected %d/%d columns (alpha=%s, min_skew=%s)",
            stats.get("selected_count", 0),
            stats.get("total_features", 0),
            self.alpha,
            self.min_skew,
        )
        return X_out, {"enabled": True, "applied": True, **stats, "save_path": self.save_path}

    def transform(self, X: Any) -> Tuple[Any, Dict[str, Any]]:
        # If explicitly enabled OR learned artifact exists, apply for train/infer parity.
        should_try = self.enabled or os.path.exists(self.save_path)
        if not should_try:
            return X, {"enabled": False, "applied": False}

        if self.selected_indices_ is None:
            self._load()
        indices = self.selected_indices_ if self.selected_indices_ is not None else np.array([], dtype=int)
        X_out = self._apply_log1p(X, indices)
        return X_out, {
            "enabled": True,
            "applied": True,
            "selected_count": int(indices.size),
            "save_path": self.save_path,
            "mode": "inference",
        }
