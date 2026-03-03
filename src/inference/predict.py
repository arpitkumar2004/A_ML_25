# src/inference/predict.py
import os
import re
from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
from scipy import sparse

from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory
from ..data.parse_features import Parser
from ..features.build_features import FeatureBuilder
from ..features.dimensionality import DimReducer
from ..models.stacker import Stacker
from ..models.base_model import BaseModel

logger = LoggerFactory.get("predict")

class PredictPipeline:
    """
    Unified inference pipeline:
      - Accepts raw dataframe (same schema as training)
      - Builds features (using cached text/image embeddings if present)
      - Applies dimensionality reduction (PCA/UMAP) if a reducer is available
      - Loads base models and stacker and produces final predictions
    """

    def __init__(self,
                 text_cfg: Dict = None,
                 image_cfg: Dict = None,
                 numeric_cfg: Dict = None,
                 feature_cache: str = "data/processed/features.joblib",
                 dim_cache: str = "data/processed/dimred.joblib",
                 models_dir: str = "experiments/models",
                 oof_meta_path: str = "experiments/oof/model_names.joblib",
                 stacker_path: str = "experiments/models/stacker.joblib"):
        self.text_cfg = text_cfg or {"method": "sbert", "cache_path":"data/processed/text_embeddings.joblib"}
        self.image_cfg = image_cfg or {"cache_path":"data/processed/image_embeddings.joblib"}
        self.numeric_cfg = numeric_cfg or {"scaler_path":"data/processed/numeric_scaler.joblib"}
        self.feature_cache = feature_cache
        self.dim_cache = dim_cache
        self.models_dir = models_dir
        self.oof_meta_path = oof_meta_path
        self.stacker_path = stacker_path

        # lazy attributes
        self._feature_builder = FeatureBuilder(self.text_cfg, self.image_cfg, self.numeric_cfg, output_cache=self.feature_cache)
        self._dim_reducer = None
        self._base_models = None
        self._model_names = None
        self._stacker = None

    def _load_dim_reducer(self):
        if self._dim_reducer is None:
            if os.path.exists(self.dim_cache):
                self._dim_reducer = DimReducer(cache_path=self.dim_cache)
                # try to load model from disk (DimReducer loads when transform called)
                logger.info(f"Dim reducer configured to use cache: {self.dim_cache}")
            else:
                logger.info("No dim reducer cache found; reducer will not be applied.")
                self._dim_reducer = None
        return self._dim_reducer

    def _discover_base_models(self):
        # Discover saved fold models in models_dir and build a mapping name -> list(paths)
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".joblib") or f.endswith(".pkl")]
        # Supported formats:
        #  1) fold_{i}_{ModelName}.joblib
        #  2) {ModelName}_fold{i}.pkl
        models_by_type = {}
        for f in model_files:
            p = os.path.join(self.models_dir, f)

            match_style_a = re.match(r"^fold_\d+_(?P<name>.+)\.(joblib|pkl)$", f)
            match_style_b = re.match(r"^(?P<name>.+)_fold\d+\.(joblib|pkl)$", f)

            if match_style_a:
                model_type = match_style_a.group("name")
            elif match_style_b:
                model_type = match_style_b.group("name")
            else:
                continue

            models_by_type.setdefault(model_type, []).append(p)
        logger.info(f"Found base model artifacts: { {k: len(v) for k,v in models_by_type.items()} }")
        self._base_models = models_by_type
        # If model names metadata exists, use that
        if os.path.exists(self.oof_meta_path):
            try:
                self._model_names = IO.load_pickle(self.oof_meta_path)
            except Exception:
                self._model_names = None

    def _load_stacker(self):
        if os.path.exists(self.stacker_path):
            self._stacker = Stacker(method="ridge", save_path=self.stacker_path)
            self._stacker.load(self.stacker_path)
            logger.info(f"Loaded stacker from {self.stacker_path}")
        else:
            logger.info("No stacker model found at path; ensemble averaging will be used.")
            self._stacker = None

    def _predict_with_saved_models(self, X):
        """
        For each model type (e.g., LGBModel), load each fold model and average predictions.
        Returns a DataFrame of shape (n_samples, n_model_types)
        """
        if self._base_models is None:
            self._discover_base_models()
        preds = {}
        for model_type, paths in self._base_models.items():
            # load each fold and average
            preds_per_fold = []
            for p in paths:
                try:
                    m = IO.load_pickle(p)
                    X_model = self._align_features_for_model(X, m)
                    preds_per_fold.append(m.predict(X_model))
                except Exception as e:
                    logger.warning(f"Loading/predicting with model {p} failed: {e}")
            if preds_per_fold:
                avg_pred = np.mean(np.vstack(preds_per_fold), axis=0)
                preds[model_type] = avg_pred
        if not preds:
            raise RuntimeError("No base model predictions available.")
        df_preds = pd.DataFrame(preds)
        return df_preds

    def _align_features_for_model(self, X, model):
        """Best-effort feature-width alignment for legacy serialized models.

        Some old artifacts were trained with a fixed feature width and fail when
        current runtime features differ. For operational serving checks, we align
        by truncating or zero-padding columns to the model's expected width.
        """
        expected = getattr(model, "n_features_", None)
        if expected is None and hasattr(model, "model"):
            expected = getattr(model.model, "n_features_", None)

        if expected is None:
            return X

        current = X.shape[1]
        if current == expected:
            return X

        logger.warning(f"Aligning feature width for legacy model: current={current}, expected={expected}")

        if sparse.issparse(X):
            X_csr = X.tocsr()
            if current > expected:
                return X_csr[:, :expected]
            pad = sparse.csr_matrix((X_csr.shape[0], expected - current), dtype=X_csr.dtype)
            return sparse.hstack([X_csr, pad], format="csr")

        X_np = np.asarray(X)
        if current > expected:
            return X_np[:, :expected]
        pad = np.zeros((X_np.shape[0], expected - current), dtype=X_np.dtype)
        return np.hstack([X_np, pad])

    def predict(self, df: pd.DataFrame, text_col: str = "Description", image_col: str = "image_path", force_rebuild_features: bool = False):
        """
        Input: df with same schema as training (unique_identifier, Description, Price optional, image_path optional)
        Returns: numpy array of final predictions (same order as df)
        """
        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe passed to PredictPipeline.predict")

        if text_col in df.columns:
            df = Parser.add_parsed_features(df, text_col=text_col)

        # Build features (reuses cached embeddings if available)
        X_raw, meta = self._feature_builder.build(df, text_col=text_col, image_col=image_col, force_rebuild=force_rebuild_features)

        # convert sparse -> dense if reducer requires it
        X_dense = None
        if hasattr(X_raw, "toarray") or hasattr(X_raw, "todense"):
            try:
                X_dense = np.asarray(X_raw.todense() if hasattr(X_raw, "todense") else X_raw.toarray())
            except Exception:
                X_dense = None
        else:
            X_dense = np.asarray(X_raw)

        # Apply dim reducer if available
        reducer = self._load_dim_reducer()
        if reducer is not None and X_dense is not None:
            try:
                X_final = reducer.transform(X_dense)
            except Exception as e:
                logger.warning(f"Dim reducer transform failed: {e}; using dense features instead.")
                X_final = X_dense
        else:
            X_final = X_dense if X_dense is not None else X_raw

        # Load base models and compute predictions
        self._discover_base_models()
        df_base_preds = self._predict_with_saved_models(X_final)
        logger.info(f"Computed base model predictions with columns: {list(df_base_preds.columns)}")

        # Load stacker if available
        self._load_stacker()
        if self._stacker is not None:
            final_preds = self._stacker.predict(df_base_preds.values)
        else:
            # average base models
            final_preds = df_base_preds.mean(axis=1).values

        return final_preds
