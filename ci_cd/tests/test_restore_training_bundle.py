from pathlib import Path

from scripts.restore_training_bundle import restore_training_bundle


def test_restore_training_bundle_prefers_release_then_falls_back_to_artifact(tmp_path, monkeypatch):
    release_calls = []
    artifact_calls = []

    def _fail_release(**kwargs):
        release_calls.append(kwargs)
        raise FileNotFoundError("release asset missing")

    def _artifact_restore(**kwargs):
        artifact_calls.append(kwargs)
        bundle_path = Path(kwargs["output_root"]) / "experiments" / "runs" / kwargs["run_id"] / "bundle"
        bundle_path.mkdir(parents=True, exist_ok=True)
        return {
            "restored": True,
            "source": "artifact",
            "run_id": kwargs["run_id"],
            "bundle_path": str(bundle_path),
        }

    monkeypatch.setattr("scripts.restore_training_bundle.restore_bundle_from_release_asset", _fail_release)
    monkeypatch.setattr("scripts.restore_training_bundle.restore_training_bundle_from_artifact", _artifact_restore)

    result = restore_training_bundle(
        run_id="train_20260316T130000Z",
        repo="owner/repo",
        token="token",
        output_root=str(tmp_path),
    )

    assert result["restored"] is True
    assert result["source"] == "artifact"
    assert len(release_calls) == 1
    assert len(artifact_calls) == 1
    assert result["attempts"][0]["source"] == "github_release"
