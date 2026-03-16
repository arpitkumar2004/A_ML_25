import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.bundle_archive import bundle_release_tag, create_bundle_archive
from src.utils.github_release_store import ensure_release, upload_release_asset


def persist_bundle_release_asset(
    run_id: str,
    repo: str,
    token: str,
    registry_dir: str = "experiments/registry",
    output_dir: str = "dist/bundle-archives",
    tag_prefix: str = "bundle-archive",
) -> dict:
    archive_meta = create_bundle_archive(run_id=run_id, registry_dir=registry_dir, output_dir=output_dir)
    tag = bundle_release_tag(run_id=run_id, tag_prefix=tag_prefix)
    release = ensure_release(
        repo=repo,
        token=token,
        tag=tag,
        name=f"Bundle Archive {run_id}",
        body=f"Durable bundle archive for {run_id}",
    )

    with open(archive_meta["archive_path"], "rb") as handle:
        asset = upload_release_asset(
            release=release,
            repo=repo,
            token=token,
            asset_name=archive_meta["archive_name"],
            asset_bytes=handle.read(),
        )

    result = {
        **archive_meta,
        "storage": "github_release",
        "release_id": release.get("id"),
        "release_tag": tag,
        "release_url": release.get("html_url"),
        "asset_id": asset.get("id"),
        "asset_url": asset.get("url"),
        "browser_download_url": asset.get("browser_download_url"),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and persist a bundle archive as a GitHub release asset")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--output-dir", default="dist/bundle-archives")
    parser.add_argument("--tag-prefix", default=os.getenv("BUNDLE_RELEASE_TAG_PREFIX", "bundle-archive"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.repo:
        raise ValueError("Missing repo. Use --repo or set GITHUB_REPOSITORY.")
    if not args.token:
        raise ValueError("Missing token. Use --token or set GITHUB_TOKEN.")

    result = persist_bundle_release_asset(
        run_id=args.run_id,
        repo=args.repo,
        token=args.token,
        registry_dir=args.registry_dir,
        output_dir=args.output_dir,
        tag_prefix=args.tag_prefix,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
