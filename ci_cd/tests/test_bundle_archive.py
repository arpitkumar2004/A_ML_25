import json
from pathlib import Path

from src.utils.bundle_archive import create_bundle_archive, restore_bundle_archive


def _write_minimal_bundle(bundle_dir: Path, run_id: str) -> None:
    reports_dir = bundle_dir / "reports"
    models_dir = bundle_dir / "models"
    artifacts_dir = bundle_dir / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "numeric_scaler.joblib").write_text("stub", encoding="utf-8")
    (models_dir / "fold_0.joblib").write_text("stub", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "outputs": {
            "bundle": {
                "bundle_path": f"experiments/runs/{run_id}/bundle",
                "models_dir": f"experiments/runs/{run_id}/bundle/models",
                "numeric_scaler_path": f"experiments/runs/{run_id}/bundle/artifacts/numeric_scaler.joblib",
            }
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_create_and_restore_bundle_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "train_20260316T120000Z"
    bundle_dir = tmp_path / "experiments" / "runs" / run_id / "bundle"
    registry_dir = tmp_path / "experiments" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_bundle(bundle_dir=bundle_dir, run_id=run_id)

    index_payload = {
        "runs": [
            {
                "run_id": run_id,
                "bundle_path": f"experiments/runs/{run_id}/bundle",
                "manifest_path": f"experiments/runs/{run_id}/bundle/manifest.json",
            }
        ],
        "active_production_run_id": None,
    }
    (registry_dir / "index.json").write_text(json.dumps(index_payload), encoding="utf-8")

    archive_meta = create_bundle_archive(run_id=run_id, output_dir="dist/bundles")
    assert archive_meta["archive_name"] == f"{run_id}-bundle.zip"
    assert Path(archive_meta["archive_path"]).exists()

    restored_root = tmp_path / "restored"
    restore_result = restore_bundle_archive(
        run_id=run_id,
        archive_path=archive_meta["archive_path"],
        output_root=str(restored_root),
    )

    restored_bundle = restored_root / "experiments" / "runs" / run_id / "bundle"
    assert restored_bundle.exists()
    assert restore_result["bundle_path"] == str(restored_bundle)
    assert restore_result["validation"]["valid"] is True
