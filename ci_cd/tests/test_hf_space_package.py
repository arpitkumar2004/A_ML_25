import os

import pandas as pd

from scripts.create_hf_space_package import create_hf_space_package
from scripts.publish_to_hf_space import publish_to_hf_space
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


def _prepare_run(tmp_path):
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
    return cfg, summary


def test_create_hf_space_package_builds_docker_space_repo(tmp_path):
    cfg, summary = _prepare_run(tmp_path)
    output_dir = tmp_path / "hf-space-package"

    package = create_hf_space_package(
        run_id=summary["run_id"],
        output_dir=str(output_dir),
        registry_dir=cfg["registry_dir"],
        service_image=f"hf-space:{summary['run_id']}",
        space_repo_id="arpitkumariitkgp/aml25",
    )

    assert os.path.exists(package["dockerfile_path"])
    assert os.path.exists(package["readme_path"])
    assert os.path.isdir(package["bundle_dir"])
    assert os.path.isdir(package["metadata_dir"])
    assert os.path.exists(output_dir / "start-serving.sh")
    assert os.path.exists(output_dir / ".hfignore")
    assert os.path.exists(output_dir / "space_package_summary.json")

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    dockerfile = (output_dir / "Dockerfile").read_text(encoding="utf-8")
    summary_json = IO.load_json(str(output_dir / "space_package_summary.json"))

    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert summary["run_id"] in readme
    assert "EXPOSE 7860" in dockerfile
    assert summary_json["space_repo_id"] == "arpitkumariitkgp/aml25"


def test_publish_to_hf_space_uses_hf_api_upload_folder(monkeypatch, tmp_path):
    package_dir = tmp_path / "space"
    package_dir.mkdir()
    (package_dir / "README.md").write_text("test", encoding="utf-8")
    (package_dir / "Dockerfile").write_text("FROM python:3.10-slim", encoding="utf-8")

    captured = {}

    class DummyCommitInfo:
        commit_url = "https://huggingface.co/spaces/arpitkumariitkgp/aml25/commit/123"
        oid = "123"

    class DummyApi:
        def __init__(self, token):
            captured["token"] = token

        def create_repo(self, **kwargs):
            captured["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            captured["upload_folder"] = kwargs
            return DummyCommitInfo()

    monkeypatch.setattr("scripts.publish_to_hf_space.HfApi", DummyApi)

    result = publish_to_hf_space(
        package_dir=str(package_dir),
        space_repo_id="arpitkumariitkgp/aml25",
        token="secret-token",
        commit_message="Deploy test package",
        create_if_missing=True,
    )

    assert captured["token"] == "secret-token"
    assert captured["create_repo"]["repo_id"] == "arpitkumariitkgp/aml25"
    assert captured["create_repo"]["repo_type"] == "space"
    assert captured["upload_folder"]["repo_id"] == "arpitkumariitkgp/aml25"
    assert captured["upload_folder"]["repo_type"] == "space"
    assert captured["upload_folder"]["delete_patterns"] == "*"
    assert result["oid"] == "123"
