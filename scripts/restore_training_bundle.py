import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.restore_bundle_from_release_asset import restore_bundle_from_release_asset
from scripts.restore_training_bundle_from_artifact import restore_training_bundle_from_artifact


def restore_training_bundle(
    run_id: str,
    repo: str,
    token: str,
    artifact_name: str = "training-artifacts",
    output_root: str = ".",
    tag_prefix: str = "bundle-archive",
) -> dict:
    workspace_bundle = Path(output_root) / "experiments" / "runs" / run_id / "bundle"
    if workspace_bundle.exists():
        result = {
            "restored": True,
            "source": "workspace",
            "run_id": run_id,
            "bundle_path": str(workspace_bundle),
        }
        print(json.dumps(result, indent=2))
        return result

    attempts: list[dict] = []

    try:
        return restore_bundle_from_release_asset(
            run_id=run_id,
            repo=repo,
            token=token,
            output_root=output_root,
            tag_prefix=tag_prefix,
        )
    except Exception as exc:
        attempts.append({"source": "github_release", "error": str(exc)})

    try:
        result = restore_training_bundle_from_artifact(
            run_id=run_id,
            repo=repo,
            token=token,
            artifact_name=artifact_name,
            output_root=output_root,
        )
        result["attempts"] = attempts
        return result
    except Exception as exc:
        attempts.append({"source": "github_artifact", "error": str(exc)})

    raise FileNotFoundError(
        f"Unable to restore bundle for {run_id}. Attempts: {json.dumps(attempts, indent=2)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a training bundle from durable storage or fallback artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--artifact-name", default="training-artifacts")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--tag-prefix", default=os.getenv("BUNDLE_RELEASE_TAG_PREFIX", "bundle-archive"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.repo:
        raise ValueError("Missing repo. Use --repo or set GITHUB_REPOSITORY.")
    if not args.token:
        raise ValueError("Missing token. Use --token or set GITHUB_TOKEN.")

    result = restore_training_bundle(
        run_id=args.run_id,
        repo=args.repo,
        token=args.token,
        artifact_name=args.artifact_name,
        output_root=args.output_root,
        tag_prefix=args.tag_prefix,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
