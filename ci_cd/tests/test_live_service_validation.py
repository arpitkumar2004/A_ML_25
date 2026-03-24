import importlib

from scripts.health_check import run_health_checks
from scripts.monitor_canary_metrics import monitor_canary_metrics
from scripts.validate_production_model import validate_production_model
from src.pipelines.train_pipeline import run_train_pipeline
from src.registry.model_registry import promote_run
from src.utils.deployment_state import write_deployment_manifest, write_production_tracker


def _tiny_train_df():
    import pandas as pd

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


def test_serving_service_info_exposes_active_bundle(monkeypatch, tmp_path):
    cfg, summary = _prepare_run(tmp_path)

    monkeypatch.setenv("REGISTRY_DIR", cfg["registry_dir"])
    monkeypatch.setenv("MODEL_BUNDLE_PATH", summary["bundle_path"])
    monkeypatch.setenv("MODEL_RUN_ID", summary["run_id"])
    monkeypatch.setenv("CANARY_PERCENT", "0")
    monkeypatch.delenv("CANARY_BUNDLE_PATH", raising=False)
    monkeypatch.delenv("CANARY_RUN_ID", raising=False)

    serving_module = importlib.import_module("src.serving.app")
    serving_module.service = serving_module.ModelService()
    serving_module.service.initialize()

    ready = serving_module.readyz()
    info = serving_module.service_info()
    metrics = serving_module.metrics_json()
    health = serving_module.healthz()

    assert ready["ready"] is True
    assert info["run_id"] == summary["run_id"]
    assert info["bundle_path"] == summary["bundle_path"]
    assert metrics["service"]["run_id"] == summary["run_id"]
    assert health["service"]["bundle_path"] == summary["bundle_path"]


def test_health_and_validation_prefer_live_service_probe(monkeypatch, tmp_path):
    cfg, summary = _prepare_run(tmp_path)
    deployment_manifest_path = tmp_path / "deployment_manifest.json"
    tracker_path = tmp_path / "production_tracker.json"

    write_deployment_manifest(
        run_id=summary["run_id"],
        strategy="blue_green",
        manifest_path=str(deployment_manifest_path),
        registry_dir=cfg["registry_dir"],
        deployed_by="pytest",
        status="active",
    )
    write_production_tracker(
        run_id=summary["run_id"],
        strategy="blue_green",
        manifest_path=str(tracker_path),
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        metrics={"latency_p95": 0.25, "error_rate": 0.0, "smape": 10.0},
        status="active",
    )

    live_probe = {
        "reachable": True,
        "valid": True,
        "errors": [],
        "service_info": {
            "run_id": summary["run_id"],
            "bundle_path": summary["bundle_path"],
        },
        "metrics_payload": {
            "request_count": 100,
            "error_count": 0,
            "latency_ms": {"p95": 250.0},
            "service": {"run_id": summary["run_id"], "bundle_path": summary["bundle_path"]},
        },
        "metrics": {"latency_p95": 0.25, "error_rate": 0.0},
    }

    monkeypatch.setattr("scripts.health_check.try_probe_live_service", lambda *args, **kwargs: live_probe)
    monkeypatch.setattr(
        "scripts.health_check.try_probe_live_prediction",
        lambda *args, **kwargs: {
            **live_probe,
            "prediction_response": {"predictions": [{"sample_id": 1, "predicted_price": 10.0}]},
        },
    )
    monkeypatch.setattr("scripts.validate_production_model.try_probe_live_service", lambda *args, **kwargs: live_probe)
    monkeypatch.setattr("scripts.monitor_canary_metrics.try_probe_live_service", lambda *args, **kwargs: live_probe)

    health = run_health_checks(
        do_check_mlflow=False,
        include_production_model=True,
        include_registry=True,
        include_inference=True,
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        tracker_path=str(tracker_path),
        service_base_url="http://live-service",
    )
    validation = validate_production_model(
        registry_dir=cfg["registry_dir"],
        deployment_manifest_path=str(deployment_manifest_path),
        tracker_path=str(tracker_path),
        service_base_url="http://live-service",
    )
    canary = monitor_canary_metrics(
        service_base_url="http://live-service",
        expected_run_id=summary["run_id"],
    )

    assert health["all_passed"] is True
    assert validation["validation_passed"] is True
    assert validation["metrics_source"] == "http://live-service/metrics/json"
    assert canary["passed"] is True
