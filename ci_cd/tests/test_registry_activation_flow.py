from src.registry.model_registry import activate_run, register_run, promote_run
from src.utils.io import IO


def test_production_promotion_can_be_approved_without_activation(tmp_path):
    registry_dir = tmp_path / "registry"
    manifest_a = tmp_path / "run_a_manifest.json"
    manifest_b = tmp_path / "run_b_manifest.json"
    IO.save_json({"run_id": "run_a"}, str(manifest_a), indent=2)
    IO.save_json({"run_id": "run_b"}, str(manifest_b), indent=2)

    register_run(
        run_id="run_a",
        manifest_path=str(manifest_a),
        stage="train",
        registry_dir=str(registry_dir),
        bundle_path="experiments/runs/run_a/bundle",
    )
    register_run(
        run_id="run_b",
        manifest_path=str(manifest_b),
        stage="train",
        registry_dir=str(registry_dir),
        bundle_path="experiments/runs/run_b/bundle",
    )

    promote_run("run_a", "production", registry_dir=str(registry_dir))
    promote_run("run_b", "production", registry_dir=str(registry_dir), activate_production=False)

    payload = IO.load_json(str(registry_dir / "index.json"))
    run_a = next(run for run in payload["runs"] if run["run_id"] == "run_a")
    run_b = next(run for run in payload["runs"] if run["run_id"] == "run_b")

    assert payload["active_production_run_id"] == "run_a"
    assert run_a["status"] == "production"
    assert run_b["status"] == "production"


def test_activate_run_switches_active_production_and_archives_previous(tmp_path):
    registry_dir = tmp_path / "registry"
    manifest_a = tmp_path / "run_a_manifest.json"
    manifest_b = tmp_path / "run_b_manifest.json"
    IO.save_json({"run_id": "run_a"}, str(manifest_a), indent=2)
    IO.save_json({"run_id": "run_b"}, str(manifest_b), indent=2)

    register_run(
        run_id="run_a",
        manifest_path=str(manifest_a),
        stage="train",
        registry_dir=str(registry_dir),
        bundle_path="experiments/runs/run_a/bundle",
    )
    register_run(
        run_id="run_b",
        manifest_path=str(manifest_b),
        stage="train",
        registry_dir=str(registry_dir),
        bundle_path="experiments/runs/run_b/bundle",
    )

    promote_run("run_a", "production", registry_dir=str(registry_dir))
    promote_run("run_b", "production", registry_dir=str(registry_dir), activate_production=False)
    activate_run("run_b", registry_dir=str(registry_dir))

    payload = IO.load_json(str(registry_dir / "index.json"))
    run_a = next(run for run in payload["runs"] if run["run_id"] == "run_a")
    run_b = next(run for run in payload["runs"] if run["run_id"] == "run_b")

    assert payload["active_production_run_id"] == "run_b"
    assert run_a["status"] == "archived"
    assert run_b["status"] == "production"
