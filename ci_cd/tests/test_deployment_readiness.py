import os

import pandas as pd

from scripts.health_check import run_health_checks
from scripts.pre_deployment_checks import run_pre_deployment_checks
from scripts.test_model_inference import run_inference_smoke_test
from scripts.validate_production_model import validate_production_model
from src.pipelines.train_pipeline import run_train_pipeline
from src.registry.model_registry import promote_run
from src.utils.deployment_state import (
    resolve_live_deployment_state,
    write_deployment_manifest,
    write_production_tracker,
)
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


def _train_cfg(tmp_path) -> dict:
    working_dir = tmp_path / "working"
    experiments_dir = tmp_path / "experiments"
    registry_dir = tmp_path / "registry"
    train_csv = tmp_path / "train.csv"
    _tiny_train_df().to_csv(train_csv, index=False)
    return {
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


def _prepare_deployed_run(tmp_path):
    cfg = _train_cfg(tmp_path)
    summary = run_train_pipeline(cfg, model_name="Linear")
    promote_run(summary["run_id"], "production", registry_dir=cfg["registry_dir"])
    return cfg, summary


def test_deployment_manifest_and_tracker_follow_promoted_bundle(tmp_path):
    cfg, summary = _prepare_deployed_run(tmp_path)
    deployment_manifest_path = tmp_path / "deployment_manifest.json"
    tracker_path = tmp_path / "production_tracker.json"

    deployment_manifest = write_deployment_manifest(
        run_id=summary["run_id"],
        strategy="canary",
        manifest_path=str(deployment_manifest_path),
        registry_dir=cfg["registry_dir"],
        deployed_by="pytest",
        canary_percent=10,
        status="canary",
    )
    tracker = write_production_tracker(
        run_id=summary["run_id"],
        strategy="canary",
        manifest_path=str(tracker_path),
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        metrics={"latency_p95": 0.5, "error_rate": 0.0, "smape": 10.0},
        status="active",
    )

    assert deployment_manifest["run_id"] == summary["run_id"]
    assert deployment_manifest["bundle_path"] == summary["bundle_path"]
    assert deployment_manifest["bundle_validation"]["valid"] is True
    assert deployment_manifest["bundle_config_sha256"]
    assert tracker["deployment_manifest_path"] == str(deployment_manifest_path)
    assert tracker["registry_consistency"]["matches_active_production"] is True

    state = resolve_live_deployment_state(
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        tracker_path=str(tracker_path),
    )
    assert state["valid"] is True


def test_predeployment_inference_health_and_validation_use_live_bundle(tmp_path):
    cfg, summary = _prepare_deployed_run(tmp_path)
    deployment_manifest_path = tmp_path / "deployment_manifest.json"
    tracker_path = tmp_path / "production_tracker.json"
    health_output = tmp_path / "health.json"
    readiness_output = tmp_path / "predeploy.json"
    inference_output = tmp_path / "inference.json"
    dataset_path = tmp_path / "train.csv"

    write_deployment_manifest(
        run_id=summary["run_id"],
        strategy="blue_green",
        manifest_path=str(deployment_manifest_path),
        registry_dir=cfg["registry_dir"],
        deployed_by="pytest",
        status="active",
    )

    predeploy = run_pre_deployment_checks(
        run_id=summary["run_id"],
        output=str(readiness_output),
        registry_dir=cfg["registry_dir"],
        check_tests=True,
        check_coverage=True,
        check_security=True,
    )
    assert predeploy["all_passed"] is True

    inference = run_inference_smoke_test(
        run_id=summary["run_id"],
        test_dataset=str(dataset_path),
        registry_dir=cfg["registry_dir"],
        output=str(inference_output),
    )
    assert inference["passed"] is True
    assert inference["metrics"]["error_rate"] == 0.0

    write_production_tracker(
        run_id=summary["run_id"],
        strategy="blue_green",
        manifest_path=str(tracker_path),
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        metrics=inference["metrics"],
        status="active",
    )

    health = run_health_checks(
        do_check_mlflow=False,
        include_production_model=True,
        include_registry=True,
        include_inference=True,
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        tracker_path=str(tracker_path),
        sample_dataset=str(dataset_path),
    )
    IO.save_json(health, str(health_output), indent=2)
    assert health["all_passed"] is True

    validation = validate_production_model(
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        tracker_path=str(tracker_path),
    )
    assert validation["validation_passed"] is True
    assert validation["run_id"] == summary["run_id"]
    assert os.path.exists(inference_output)
