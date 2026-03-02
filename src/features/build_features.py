# src/features/build_features.py
from typing import Optional, Tuple, List, Any
import numpy as np
import pandas as pd
from scipy import sparse
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO
from .text_embeddings import TextEmbedder
from .image_embeddings import ImageEmbedder
from .numeric_features import NumericBuilder

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
                 output_cache: Optional[str] = "data/processed/features.joblib"):
        self.text_embedder = TextEmbedder(**text_cfg)
        self.image_embedder = ImageEmbedder(**image_cfg)
        self.numeric_builder = NumericBuilder(scaler_path=numeric_cfg.get("scaler_path", "data/processed/numeric_scaler.joblib"))
        self.output_cache = output_cache

    def build(self, df: pd.DataFrame, text_col: str = "Description", image_col: str = "image_path",
              numeric_cols: Optional[List[str]] = None, force_rebuild: bool = False) -> Tuple[Any, dict]:
        """
        Build and return final features and meta info.
        If cache exists and not force_rebuild, load and return cached.
        """
        if self.output_cache and os.path.exists(self.output_cache) and not force_rebuild:
            logger.info(f"Loading cached feature matrix from {self.output_cache}")
            cached = IO.load_pickle(self.output_cache)
            return cached["X"], cached["meta"]

        # 1) Text embeddings
        texts = df[text_col].fillna("").astype(str).tolist()
        X_text = self.text_embedder.fit_transform(texts, use_cache=True)

        # 2) Image embeddings
        if image_col in df.columns:
            image_paths = df[image_col].fillna("").astype(str).tolist()
            X_image = self.image_embedder.embed(image_paths, use_cache=True)
        else:
            X_image = np.zeros((len(df), 512), dtype=float)  # fallback

        # 3) Numeric features
        # allow explicit numeric cols or auto-detect
        if numeric_cols is None:
            # choose parsed features if present (parsed_value, parsed_ounces etc.)
            possible = ["parsed_value", "parsed_ounces", f"{text_col}_clean_len"]
            numeric_cols = [c for c in possible if c in df.columns]
        X_num, used_numeric_cols = self.numeric_builder.fit(df, numeric_cols=numeric_cols)

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

        meta = {
            "text_shape": X_text.shape,
            "image_shape": X_image.shape,
            "numeric_shape": X_num.shape,
            "numeric_cols": used_numeric_cols
        }

        # cache
        IO.save_pickle({"X": X, "meta": meta}, self.output_cache)
        logger.info(f"Saved combined feature matrix to {self.output_cache}")
        return X, meta
