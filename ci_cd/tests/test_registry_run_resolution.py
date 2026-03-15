import json

from src.utils.registry_loader import RegistryLoader


def test_resolve_bundle_backed_run_id_prefers_canonical_train_run(tmp_path):
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {
                "run_id": "train_20260315T123659Z",
                "stage": "train",
                "status": "staging",
                "bundle_path": "experiments/runs/train_20260315T123659Z/bundle",
                "tracking": {
                    "mlflow": {
                        "mlflow_run_id": "legacy_mlflow_id_123",
                    }
                },
            },
            {
                "run_id": "legacy_mlflow_id_123",
                "stage": "promotion",
                "status": "staging",
                "bundle_path": None,
            },
        ],
        "active_production_run_id": None,
    }
    (registry_dir / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    loader = RegistryLoader(registry_dir=str(registry_dir))

    assert loader.resolve_bundle_backed_run_id("train_20260315T123659Z") == "train_20260315T123659Z"
    assert loader.resolve_bundle_backed_run_id("legacy_mlflow_id_123") == "train_20260315T123659Z"
