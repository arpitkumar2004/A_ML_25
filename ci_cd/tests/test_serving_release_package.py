import os

import pandas as pd

from scripts.create_deployment_package import create_deployment_package
from src.pipelines.train_pipeline import run_train_pipeline
from src.registry.model_registry import promote_run
from src.utils.io import IO


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


def test_create_deployment_package_snapshots_bundle_and_contract(tmp_path):
    train_csv = tmp_path / "train.csv"
    _tiny_train_df().to_csv(train_csv, index=False)

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
        "image_cfg": {"cache_path": str(tmp_path / "image_embeddings.joblib")},
        "numeric_cfg": {"scaler_path": str(tmp_path / "numeric_scaler.joblib")},
        "selector_cfg": {
            "enabled": True,
            "method": "f_regression",
            "k": 8,
            "min_features": 4,
            "save_path": str(tmp_path / "feature_selector.joblib"),
            "random_state": 42,
        },
        "post_log_cfg": {"enabled": False, "save_path": str(tmp_path / "post_feature_log_transform.joblib")},
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
    promote_run(summary["run_id"], "production", registry_dir=cfg["registry_dir"])

    package = create_deployment_package(
        run_id=summary["run_id"],
        output_dir=str(tmp_path / "deployments" / "current"),
        registry_dir=cfg["registry_dir"],
        service_image=f"aml25-serving:{summary['run_id']}",
    )

    assert os.path.isdir(package["bundle_output_dir"])
    assert os.path.exists(os.path.join(package["bundle_output_dir"], "manifest.json"))
    assert os.path.exists(package["deployment_manifest_path"])
    assert os.path.exists(package["service_contract_path"])
    assert os.path.exists(package["service_env_path"])

    service_contract = IO.load_json(package["service_contract_path"])
    deployment_manifest = IO.load_json(package["deployment_manifest_path"])
    service_env = open(package["service_env_path"], "r", encoding="utf-8").read()

    assert service_contract["run_id"] == summary["run_id"]
    assert service_contract["bundle_mount_path"] == "/opt/model-bundle"
    assert "/readyz" in service_contract["health_endpoints"]
    assert deployment_manifest["service_image"] == f"aml25-serving:{summary['run_id']}"
    assert "MODEL_BUNDLE_PATH=/opt/model-bundle" in service_env
    assert f"MODEL_RUN_ID={summary['run_id']}" in service_env
