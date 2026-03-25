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
from ..utils.column_aliases import normalize_to_train_schema, resolve_column_name
from ..utils.model_bundle import resolve_bundle_path, bundle_runtime_contract, load_bundle_manifest

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
                 selector_cfg: Dict = None,
                 post_log_cfg: Dict = None,
                 feature_cache: Optional[str] = None,
                 dim_cache: Optional[str] = None,
                 models_dir: Optional[str] = None,
                 oof_meta_path: Optional[str] = None,
                 stacker_path: Optional[str] = None,
                 bundle_path: Optional[str] = None,
                 run_id: Optional[str] = None,
                 registry_dir: str = "experiments/registry"):
        self.registry_dir = registry_dir
        self.run_id = run_id
        self.bundle_path = resolve_bundle_path(bundle_path=bundle_path, run_id=run_id, registry_dir=registry_dir, require_exists=False) if (bundle_path or run_id) else None

        bundle_cfg: Dict[str, Any] = {}
        bundle_contract: Dict[str, str] = {}
        if self.bundle_path:
            manifest = load_bundle_manifest(self.bundle_path)
            bundle_cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
            bundle_contract = bundle_runtime_contract(self.bundle_path)

        resolved_text_cfg = dict(bundle_cfg.get("text_cfg", {}) if isinstance(bundle_cfg.get("text_cfg", {}), dict) else {})
        if text_cfg:
            resolved_text_cfg.update(text_cfg)
        resolved_text_cfg.setdefault("method", "sbert")
        if self.bundle_path and "cache_path" not in resolved_text_cfg:
            resolved_text_cfg["cache_path"] = None
        else:
            resolved_text_cfg.setdefault("cache_path", "data/processed/text_embeddings.joblib")
        if bundle_contract.get("text_vectorizer_path"):
            resolved_text_cfg["vectorizer_path"] = bundle_contract["text_vectorizer_path"]
        else:
            resolved_text_cfg.setdefault("vectorizer_path", "data/processed/tfidf_vectorizer.joblib")

        resolved_image_cfg = dict(bundle_cfg.get("image_cfg", {}) if isinstance(bundle_cfg.get("image_cfg", {}), dict) else {})
        if image_cfg:
            resolved_image_cfg.update(image_cfg)
        if self.bundle_path and "cache_path" not in resolved_image_cfg:
            resolved_image_cfg["cache_path"] = None
        else:
            resolved_image_cfg.setdefault("cache_path", "data/processed/image_embeddings.joblib")

        resolved_numeric_cfg = dict(bundle_cfg.get("numeric_cfg", {}) if isinstance(bundle_cfg.get("numeric_cfg", {}), dict) else {})
        if numeric_cfg:
            resolved_numeric_cfg.update(numeric_cfg)
        resolved_numeric_cfg["scaler_path"] = bundle_contract.get("numeric_scaler_path") or resolved_numeric_cfg.get("scaler_path", "data/processed/numeric_scaler.joblib")

        resolved_selector_cfg = dict(bundle_cfg.get("selector_cfg", {}) if isinstance(bundle_cfg.get("selector_cfg", {}), dict) else {})
        if selector_cfg:
            resolved_selector_cfg.update(selector_cfg)
        if bundle_contract.get("selector_path"):
            resolved_selector_cfg["save_path"] = bundle_contract["selector_path"]

        resolved_post_log_cfg = dict(bundle_cfg.get("post_log_cfg", {}) if isinstance(bundle_cfg.get("post_log_cfg", {}), dict) else {})
        if post_log_cfg:
            resolved_post_log_cfg.update(post_log_cfg)
        if bundle_contract.get("post_log_transform_path"):
            resolved_post_log_cfg["save_path"] = bundle_contract["post_log_transform_path"]

        self.text_cfg = resolved_text_cfg
        self.image_cfg = resolved_image_cfg
        self.numeric_cfg = resolved_numeric_cfg
        self.selector_cfg = resolved_selector_cfg
        self.post_log_cfg = resolved_post_log_cfg
        self.feature_cache = feature_cache if self.bundle_path else (feature_cache or "data/processed/features.joblib")
        self.dim_cache = bundle_contract.get("dim_cache") or dim_cache or "data/processed/dimred.joblib"
        self.models_dir = bundle_contract.get("models_dir") or models_dir or "experiments/models"
        self.oof_meta_path = bundle_contract.get("oof_meta_path") or oof_meta_path or "experiments/oof/model_names.joblib"
        self.stacker_path = bundle_contract.get("stacker_path") or stacker_path or "experiments/models/stacker.joblib"

        # lazy attributes
        self._feature_builder = FeatureBuilder(
            self.text_cfg,
            self.image_cfg,
            self.numeric_cfg,
            selector_cfg=self.selector_cfg,
            post_log_cfg=self.post_log_cfg,
            output_cache=self.feature_cache,
        )
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
        if not os.path.isdir(self.models_dir):
            raise FileNotFoundError(f"models_dir not found: {self.models_dir}")
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

    def predict(
        self,
        df: pd.DataFrame,
        text_col: str = "catalog_content",
        image_col: str = "image_link",
        force_rebuild_features: bool = False,
        return_diagnostics: bool = False,
    ):
        """
        Input: df with same schema as training (unique_identifier, Description, Price optional, image_path optional)
        Returns: numpy array of final predictions (same order as df)
        """
        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe passed to PredictPipeline.predict")

        incoming_columns = list(df.columns)
        df, rename_map = normalize_to_train_schema(df)
        text_col = resolve_column_name(df.columns, text_col)
        image_col = resolve_column_name(df.columns, image_col)

        if text_col in df.columns:
            df = Parser.add_parsed_features(df, text_col=text_col)

        # Build features (reuses cached embeddings if available)
        X_raw, meta = self._feature_builder.build(
            df,
            text_col=text_col,
            image_col=image_col,
            force_rebuild=force_rebuild_features,
            mode="inference",
        )

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

        if not return_diagnostics:
            return final_preds

        parsed_signals: Dict[str, Any] = {}
        if len(df):
            first_row = df.iloc[0]
            parsed_signals = {
                "parsed_value": float(first_row.get("parsed_value", 0.0) or 0.0),
                "parsed_unit": str(first_row.get("parsed_unit", "") or ""),
                "parsed_ounces": float(first_row.get("parsed_ounces", 0.0) or 0.0),
                "quantity_mentions": int(first_row.get("parsed_quantity_mentions", 0.0) or 0.0),
                "total_weight_g": float(first_row.get("parsed_total_weight_g", 0.0) or 0.0),
                "total_volume_ml": float(first_row.get("parsed_total_volume_ml", 0.0) or 0.0),
                "total_count_units": float(first_row.get("parsed_total_count_units", 0.0) or 0.0),
            }

        diagnostics = {
            "schema_alignment": {
                "rename_map": rename_map,
                "original_columns": incoming_columns,
                "normalized_columns": list(df.columns),
                "resolved_text_col": text_col,
                "resolved_image_col": image_col,
            },
            "feature_matrix": {
                "raw_shape": list(X_raw.shape) if hasattr(X_raw, "shape") else None,
                "final_shape": list(X_final.shape) if hasattr(X_final, "shape") else None,
                "text_shape": list(meta.get("text_shape", [])),
                "image_shape": list(meta.get("image_shape", [])),
                "numeric_shape": list(meta.get("numeric_shape", [])),
                "selection": meta.get("selection"),
                "post_log_transform": meta.get("post_log_transform"),
            },
            "feature_extraction": {
                "text": {
                    "method": meta.get("text_method"),
                    "dimensions": int(meta["text_shape"][1]) if len(meta.get("text_shape", ())) > 1 else 0,
                    "blank_rows": int(meta.get("text_blank_rows", 0)),
                },
                "image": {
                    "backend": meta.get("image_backend"),
                    "model_name": meta.get("image_model_name"),
                    "dimensions": int(meta["image_shape"][1]) if len(meta.get("image_shape", ())) > 1 else 0,
                    "zero_rows": int(meta.get("image_zero_rows", 0)),
                },
                "numeric": {
                    "columns": list(meta.get("numeric_cols", [])),
                    "dimensions": int(meta["numeric_shape"][1]) if len(meta.get("numeric_shape", ())) > 1 else 0,
                },
            },
            "parsed_signals": parsed_signals,
            "ensemble": {
                "base_model_count": int(len(df_base_preds.columns)),
                "base_model_names": list(df_base_preds.columns),
                "base_model_outputs": {
                    key: float(value) for key, value in df_base_preds.iloc[0].to_dict().items()
                } if len(df_base_preds) else {},
                "stacker_enabled": bool(self._stacker is not None),
                "stacker_path": self.stacker_path if self._stacker is not None else None,
                "final_prediction": float(final_preds[0]) if len(final_preds) else None,
            },
        }
        return final_preds, diagnostics
