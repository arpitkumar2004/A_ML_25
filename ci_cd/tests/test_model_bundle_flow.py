import os

import pandas as pd

from src.inference.predict import PredictPipeline
from src.pipelines.train_pipeline import run_train_pipeline
from src.utils.io import IO
from src.utils.registry_loader import RegistryLoader


def _tiny_train_df() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "sample_id": idx + 1,
                "catalog_content": f"Organic tea pack {idx + 1} count {2 + idx}",
                "image_link": "",
                "price": float(10 + idx),
            }
        )
    return pd.DataFrame(rows)


def test_train_pipeline_emits_immutable_bundle_and_registry_links(tmp_path):
    train_csv = tmp_path / "train.csv"
    _tiny_train_df().to_csv(train_csv, index=False)

    working_dir = tmp_path / "working"
    experiments_dir = tmp_path / "experiments"
    registry_dir = tmp_path / "registry"

    cfg = {
        "data_path": str(train_csv),
        "sample_frac": 1.0,
        "random_state": 42,
        "seed": 42,
        "text_col": "catalog_content",
        "image_col": "image_link",
        "target_col": "price",
        "id_col": "sample_id",
        "text_cfg": {
            "method": "tfidf",
            "cache_path": str(working_dir / "text_embeddings.joblib"),
            "vectorizer_path": str(working_dir / "tfidf_vectorizer.joblib"),
            "tfidf_max_features": 32,
            "tfidf_ngram_range": (1, 2),
        },
        "image_cfg": {
            "cache_path": str(working_dir / "image_embeddings.joblib"),
        },
        "numeric_cfg": {
            "scaler_path": str(working_dir / "numeric_scaler.joblib"),
        },
        "selector_cfg": {
            "enabled": True,
            "method": "f_regression",
            "k": 8,
            "min_features": 4,
            "save_path": str(working_dir / "feature_selector.joblib"),
            "random_state": 42,
        },
        "post_log_cfg": {
            "enabled": False,
            "save_path": str(working_dir / "post_feature_log_transform.joblib"),
        },
        "feature_cache": str(working_dir / "features.joblib"),
        "dim_cache": str(working_dir / "dimred.joblib"),
        "dim_method": "pca",
        "dim_components": 4,
        "n_splits": 2,
        "run_stacker": False,
        "experiments_dir": str(experiments_dir),
        "registry_dir": str(registry_dir),
    }

    summary = run_train_pipeline(cfg, model_name="Linear")

    assert os.path.isdir(summary["bundle_path"])
    assert os.path.exists(summary["manifest_path"])

    manifest = IO.load_json(summary["manifest_path"])
    bundle_meta = manifest["outputs"]["bundle"]
    assert bundle_meta["bundle_path"] == summary["bundle_path"]
    assert os.path.exists(bundle_meta["manifest_path"])
    assert os.path.isdir(bundle_meta["models_dir"])
    assert os.path.exists(bundle_meta["text_vectorizer_path"])
    assert os.path.exists(manifest["outputs"]["numeric_scaler_path"])

    registry_entry = RegistryLoader(registry_dir=str(registry_dir)).get_run_by_id(summary["run_id"])
    assert registry_entry is not None
    assert registry_entry["bundle_path"] == summary["bundle_path"]


def test_predict_pipeline_can_load_from_bundle(tmp_path):
    train_csv = tmp_path / "train.csv"
    df_train = _tiny_train_df()
    df_train.to_csv(train_csv, index=False)

    cfg = {
        "data_path": str(train_csv),
        "sample_frac": 1.0,
        "random_state": 42,
        "seed": 42,
        "text_col": "catalog_content",
        "image_col": "image_link",
        "target_col": "price",
        "id_col": "sample_id",
        "text_cfg": {
            "method": "tfidf",
            "cache_path": str(tmp_path / "text_embeddings.joblib"),
            "vectorizer_path": str(tmp_path / "tfidf_vectorizer.joblib"),
            "tfidf_max_features": 32,
            "tfidf_ngram_range": (1, 2),
        },
        "image_cfg": {
            "cache_path": str(tmp_path / "image_embeddings.joblib"),
        },
        "numeric_cfg": {
            "scaler_path": str(tmp_path / "numeric_scaler.joblib"),
        },
        "selector_cfg": {
            "enabled": True,
            "method": "f_regression",
            "k": 8,
            "min_features": 4,
            "save_path": str(tmp_path / "feature_selector.joblib"),
            "random_state": 42,
        },
        "post_log_cfg": {
            "enabled": False,
            "save_path": str(tmp_path / "post_feature_log_transform.joblib"),
        },
        "feature_cache": str(tmp_path / "features.joblib"),
        "dim_cache": str(tmp_path / "dimred.joblib"),
        "dim_method": "pca",
        "dim_components": 4,
        "n_splits": 2,
        "run_stacker": False,
        "experiments_dir": str(tmp_path / "experiments"),
        "registry_dir": str(tmp_path / "registry"),
    }

    summary = run_train_pipeline(cfg, model_name="Linear")
    pipeline = PredictPipeline(bundle_path=summary["bundle_path"], registry_dir=cfg["registry_dir"])
    preds = pipeline.predict(df_train.head(3).copy(), text_col="catalog_content", image_col="image_link", force_rebuild_features=True)

    assert len(preds) == 3

