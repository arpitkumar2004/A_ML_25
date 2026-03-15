import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _github_json(url: str, token: str) -> Dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "aml25-bundle-restore",
        },
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_artifact_zip(url: str, token: str, dest_zip: Path) -> None:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "aml25-bundle-restore",
        },
    )
    with urlopen(request) as response, dest_zip.open("wb") as handle:
        handle.write(response.read())


def _artifact_candidates(repo: str, token: str, artifact_name: str) -> List[Dict[str, Any]]:
    payload = _github_json(f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100", token)
    artifacts = payload.get("artifacts", [])
    candidates = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == artifact_name and not artifact.get("expired", False)
    ]
    candidates.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return candidates


def restore_training_bundle_from_artifact(
    run_id: str,
    repo: str,
    token: str,
    artifact_name: str = "training-artifacts",
    output_root: str = ".",
) -> Dict[str, Any]:
    expected_bundle = Path(output_root) / "experiments" / "runs" / run_id / "bundle"
    if expected_bundle.exists():
        result = {
            "restored": True,
            "source": "workspace",
            "run_id": run_id,
            "bundle_path": str(expected_bundle),
        }
        print(json.dumps(result, indent=2))
        return result

    candidates = _artifact_candidates(repo=repo, token=token, artifact_name=artifact_name)
    if not candidates:
        raise FileNotFoundError(f"No non-expired GitHub artifact named '{artifact_name}' was found for {repo}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for artifact in candidates:
            zip_path = tmp_root / f"{artifact['id']}.zip"
            extract_dir = tmp_root / f"artifact_{artifact['id']}"
            _download_artifact_zip(str(artifact["archive_download_url"]), token=token, dest_zip=zip_path)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            candidate_bundle = extract_dir / "experiments" / "runs" / run_id / "bundle"
            if candidate_bundle.exists():
                destination = expected_bundle
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(candidate_bundle, destination)
                result = {
                    "restored": True,
                    "source": "artifact",
                    "run_id": run_id,
                    "bundle_path": str(destination),
                    "artifact_id": artifact.get("id"),
                    "artifact_name": artifact.get("name"),
                    "artifact_created_at": artifact.get("created_at"),
                }
                print(json.dumps(result, indent=2))
                return result

    known_runs = ", ".join([str(a.get("id")) for a in candidates[:5]])
    raise FileNotFoundError(
        f"Bundle for run '{run_id}' was not found in recent '{artifact_name}' artifacts for {repo}. "
        f"Checked artifact ids: {known_runs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a bundle-backed run from GitHub Actions training artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--artifact-name", default="training-artifacts")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.repo:
        raise ValueError("Missing repo. Use --repo or set GITHUB_REPOSITORY.")
    if not args.token:
        raise ValueError("Missing token. Use --token or set GITHUB_TOKEN.")

    result = restore_training_bundle_from_artifact(
        run_id=args.run_id,
        repo=args.repo,
        token=args.token,
        artifact_name=args.artifact_name,
        output_root=args.output_root,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
