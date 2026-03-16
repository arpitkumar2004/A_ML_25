import json

from scripts.validate_promotion import validate_promotion


def test_validate_promotion_falls_back_to_local_bundle_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    bundle_dir = tmp_path / "experiments" / "runs" / "train_20260315T093004Z" / "bundle"
    reports_dir = bundle_dir / "reports"
    models_dir = bundle_dir / "models"
    artifacts_dir = bundle_dir / "artifacts"
    registry_dir = tmp_path / "experiments" / "registry"
    slos_dir = tmp_path / "configs" / "validation"

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    slos_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "numeric_scaler.joblib").write_text("stub", encoding="utf-8")
    (reports_dir / "model_comparison.csv").write_text(
        "model,rmse,mae,r2,smape\nLinear,0.8,0.6,0.18,25.4487\n",
        encoding="utf-8",
    )

    manifest = {
        "run_id": "train_20260315T093004Z",
        "config": {
            "sample_frac": 0.01,
        },
        "outputs": {
            "model_report": "experiments/runs/train_20260315T093004Z/bundle/reports/model_comparison.csv",
            "bundle": {
                "bundle_path": "experiments/runs/train_20260315T093004Z/bundle",
                "models_dir": "experiments/runs/train_20260315T093004Z/bundle/models",
                "numeric_scaler_path": "experiments/runs/train_20260315T093004Z/bundle/artifacts/numeric_scaler.joblib",
            },
        },
        "timings_seconds": {
            "total": 15.27,
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    index_payload = {
        "runs": [
            {
                "run_id": "train_20260315T093004Z",
                "manifest_path": "experiments/runs/train_20260315T093004Z/bundle/manifest.json",
                "stage": "train",
                "status": "staging",
                "tracking": {
                    "mlflow": {
                        "enabled": False,
                        "mlflow_run_id": None,
                    }
                },
                "bundle_path": "experiments/runs/train_20260315T093004Z/bundle",
            }
        ],
        "active_production_run_id": None,
    }
    (registry_dir / "index.json").write_text(json.dumps(index_payload), encoding="utf-8")
    (slos_dir / "slos.yaml").write_text(
        "validation:\n"
        "  production:\n"
        "    min_accuracy: 0.70\n"
        "    max_smape: 30.0\n"
        "    max_latency_p95: 2.0\n"
        "    max_error_rate: 0.02\n",
        encoding="utf-8",
    )

    result = validate_promotion(run_id="train_20260315T093004Z", target_stage="production")

    assert result["passed"] is True
    assert result["metrics_source"] == "local_bundle"
    assert result["metrics"]["smape"] == 25.4487
    assert result["resolved_mlflow_run_id"] == "train_20260315T093004Z"
