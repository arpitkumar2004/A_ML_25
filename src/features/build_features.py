# src/features/build_features.py
from typing import Optional, Tuple, List, Any
import os
import numpy as np
import pandas as pd
from scipy import sparse
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from ..utils.fingerprint import stable_hash, hash_texts
from .text_embeddings import TextEmbedder
from .image_embeddings import ImageEmbedder
from .numeric_features import NumericBuilder
from .feature_selector import FeatureSelector

logger = LoggerFactory.get("build_features")


class FeatureBuilder:
    """
    Orchestrates building the final feature matrix:
      - text embeddings (sbert/tfidf)
      - image embeddings (clip)
      - parsed numeric features (value, unit)
    Produces:
      - X: np.ndarray or scipy.sparse matrix
      - feature_metadata: dict with keys -> slices or names for interpretability
    """
    def __init__(self,
                 text_cfg: dict,
                 image_cfg: dict,
                 numeric_cfg: dict,
                 selector_cfg: Optional[dict] = None,
                 output_cache: Optional[str] = "data/processed/features.joblib"):
        self.text_embedder = TextEmbedder(**text_cfg)
        self.image_embedder = ImageEmbedder(**image_cfg)
        self.numeric_builder = NumericBuilder(scaler_path=numeric_cfg.get("scaler_path", "data/processed/numeric_scaler.joblib"))
        self.selector_cfg = selector_cfg or {}
        self.selector_enabled = bool(self.selector_cfg.get("enabled", False))
        self.feature_selector = FeatureSelector(
            method=self.selector_cfg.get("method", "mutual_info"),
            k=int(self.selector_cfg.get("k", 1024)),
            min_features=int(self.selector_cfg.get("min_features", 64)),
            save_path=self.selector_cfg.get("save_path", "data/processed/feature_selector.joblib"),
            random_state=int(self.selector_cfg.get("random_state", 42)),
        )
        self.output_cache = output_cache

    @staticmethod
    def _derive_numeric_cols(df: pd.DataFrame, text_col: str) -> List[str]:
        preferred = [
            "parsed_value",
            "parsed_ounces",
            "parsed_value_log1p",
            "parsed_total_weight_g",
            "parsed_total_volume_ml",
            "parsed_total_count_units",
            "parsed_quantity_mentions",
            "parsed_has_quantity",
            "parsed_weight_log1p",
            "parsed_volume_log1p",
            "parsed_count_log1p",
            f"{text_col}_clean_len",
        ]
        existing = [c for c in preferred if c in df.columns]
        if existing:
            return existing
        return [c for c in df.select_dtypes(include=["number"]).columns if c.lower() != "price"]

    def _feature_fingerprint(
        self,
        df: pd.DataFrame,
        text_col: str,
        image_col: str,
        numeric_cols: List[str],
        mode: str,
    ) -> str:
        text_vals = df[text_col].fillna("").astype(str).tolist() if text_col in df.columns else []
        image_vals = df[image_col].fillna("").astype(str).tolist() if image_col in df.columns else []
        if numeric_cols:
            numeric_records = (
                df[numeric_cols]
                .fillna(0.0)
                .astype(float)
                .round(6)
                .astype(str)
                .agg("|".join, axis=1)
                .tolist()
            )
            numeric_hash = hash_texts(numeric_records)
        else:
            numeric_hash = stable_hash([])

        payload = {
            "mode": mode,
            "rows": int(len(df)),
            "text_col": text_col,
            "image_col": image_col,
            "numeric_cols": list(numeric_cols),
            "text_hash": hash_texts(text_vals),
            "image_hash": hash_texts(image_vals),
            "numeric_hash": numeric_hash,
            "selector_enabled": self.selector_enabled,
            "selector_cfg": self.selector_cfg,
        }
        return stable_hash(payload)

    def _load_feature_cache(self, fingerprint: str):
        if not self.output_cache or not os.path.exists(self.output_cache):
            return None
        payload = IO.load_pickle(self.output_cache)
        if isinstance(payload, dict) and payload.get("fingerprint") == fingerprint:
            return payload
        return None

    def build(self, df: pd.DataFrame, text_col: str = "Description", image_col: str = "image_path",
              numeric_cols: Optional[List[str]] = None, force_rebuild: bool = False,
              y: Optional[np.ndarray] = None, mode: str = "train") -> Tuple[Any, dict]:
        """
        Build and return final features and meta info.
        If fingerprint-matched cache exists and not force_rebuild, load and return cached.
        """
        mode = (mode or "train").lower()
        if mode not in {"train", "inference"}:
            raise ValueError(f"Unsupported mode='{mode}'. Expected 'train' or 'inference'.")

        if numeric_cols is None:
            numeric_cols = self._derive_numeric_cols(df, text_col=text_col)

        feature_fp = self._feature_fingerprint(df, text_col, image_col, numeric_cols, mode)
        if not force_rebuild:
            cached_payload = self._load_feature_cache(feature_fp)
            if cached_payload is not None:
                logger.info(f"Loading fingerprint-matched feature matrix from {self.output_cache}")
                return cached_payload["X"], cached_payload["meta"]

        # When forcing rebuild, bypass all sub-caches to avoid stale-row mismatches.
        use_sub_cache = not force_rebuild

        # 1) Text embeddings
        texts = df[text_col].fillna("").astype(str).tolist()
        text_fp = stable_hash({"mode": mode, "texts": hash_texts(texts), "method": self.text_embedder.method})
        if mode == "train":
            X_text = self.text_embedder.fit_transform(texts, use_cache=use_sub_cache, fingerprint=text_fp)
        else:
            X_text = self.text_embedder.transform(texts, use_cache=use_sub_cache, fingerprint=text_fp)

        # 2) Image embeddings
        if image_col in df.columns:
            image_paths = df[image_col].fillna("").astype(str).tolist()
            image_fp = stable_hash({"mode": mode, "images": hash_texts(image_paths), "model": self.image_embedder.model_name})
            X_image = self.image_embedder.embed(image_paths, use_cache=use_sub_cache, fingerprint=image_fp)
        else:
            X_image = np.zeros((len(df), 512), dtype=float)  # fallback

        # Guard against stale caches producing mismatched row counts.
        expected_rows = len(df)
        text_rows = X_text.shape[0] if hasattr(X_text, "shape") else expected_rows
        image_rows = X_image.shape[0] if hasattr(X_image, "shape") else expected_rows
        if text_rows != expected_rows or image_rows != expected_rows:
            logger.warning(
                "Detected feature row mismatch (expected=%s, text=%s, image=%s). Recomputing without sub-cache.",
                expected_rows,
                text_rows,
                image_rows,
            )
            if mode == "train":
                X_text = self.text_embedder.fit_transform(texts, use_cache=False, fingerprint=text_fp)
            else:
                X_text = self.text_embedder.transform(texts, use_cache=False, fingerprint=text_fp)
            if image_col in df.columns:
                X_image = self.image_embedder.embed(image_paths, use_cache=False, fingerprint=image_fp)
            else:
                X_image = np.zeros((len(df), 512), dtype=float)

        # 3) Numeric features
        if mode == "train":
            X_num, used_numeric_cols = self.numeric_builder.fit(df, numeric_cols=numeric_cols)
        else:
            X_num, used_numeric_cols = self.numeric_builder.transform(df, numeric_cols=numeric_cols)

        # 4) Combine
        # If X_text is sparse (tfidf), handle separately
        if sparse.issparse(X_text):
            # convert numeric and image to sparse and hstack
            from scipy.sparse import csr_matrix, hstack
            X_img_sp = csr_matrix(X_image)
            X_num_sp = csr_matrix(X_num)
            X = hstack([X_text, X_img_sp, X_num_sp], format="csr")
            logger.info("Built sparse stacked feature matrix")
        else:
            # dense concatenation
            X = np.concatenate([X_text, X_image, X_num], axis=1)
            logger.info("Built dense stacked feature matrix")

        selection_meta = {"enabled": self.selector_enabled, "applied": False}
        if self.selector_enabled:
            if mode == "train":
                if y is None:
                    logger.warning("Feature selection enabled but y is None; skipping selector during training mode.")
                else:
                    X, selector_details = self.feature_selector.fit_transform(X, np.asarray(y), feature_names=None)
                    selection_meta.update({"applied": True, **selector_details})
            else:
                X = self.feature_selector.transform(X)
                selection_meta.update({"applied": True})

        meta = {
            "text_shape": X_text.shape,
            "image_shape": X_image.shape,
            "numeric_shape": X_num.shape,
            "numeric_cols": used_numeric_cols,
            "selection": selection_meta,
            "feature_fingerprint": feature_fp,
            "mode": mode,
        }

        # cache
        if self.output_cache:
            IO.save_pickle({"fingerprint": feature_fp, "X": X, "meta": meta}, self.output_cache)
            logger.info(f"Saved combined feature matrix to {self.output_cache}")
        return X, meta
