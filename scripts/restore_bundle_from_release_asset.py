import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.bundle_archive import bundle_archive_name, bundle_release_tag, restore_bundle_archive
from src.utils.github_release_store import download_release_asset, get_release_by_tag


def restore_bundle_from_release_asset(
    run_id: str,
    repo: str,
    token: str,
    output_root: str = ".",
    tag_prefix: str = "bundle-archive",
) -> dict:
    tag = bundle_release_tag(run_id=run_id, tag_prefix=tag_prefix)
    release = get_release_by_tag(repo=repo, token=token, tag=tag)
    if release is None:
        raise FileNotFoundError(f"No GitHub release found for tag '{tag}' in {repo}")

    target_name = bundle_archive_name(run_id)
    asset = next((item for item in release.get("assets", []) if item.get("name") == target_name), None)
    if asset is None:
        raise FileNotFoundError(f"Release '{tag}' does not contain asset '{target_name}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / target_name
        archive_bytes = download_release_asset(str(asset["url"]), token=token)
        archive_path.write_bytes(archive_bytes)
        restored = restore_bundle_archive(run_id=run_id, archive_path=str(archive_path), output_root=output_root)

    result = {
        **restored,
        "storage": "github_release",
        "release_id": release.get("id"),
        "release_tag": tag,
        "release_url": release.get("html_url"),
        "asset_id": asset.get("id"),
        "asset_name": asset.get("name"),
        "browser_download_url": asset.get("browser_download_url"),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a bundle from a GitHub release asset")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--tag-prefix", default=os.getenv("BUNDLE_RELEASE_TAG_PREFIX", "bundle-archive"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.repo:
        raise ValueError("Missing repo. Use --repo or set GITHUB_REPOSITORY.")
    if not args.token:
        raise ValueError("Missing token. Use --token or set GITHUB_TOKEN.")

    result = restore_bundle_from_release_asset(
        run_id=args.run_id,
        repo=args.repo,
        token=args.token,
        output_root=args.output_root,
        tag_prefix=args.tag_prefix,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
