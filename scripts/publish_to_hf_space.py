import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import HfApi


def publish_to_hf_space(
    package_dir: str,
    space_repo_id: str,
    token: str,
    commit_message: str = "",
    create_if_missing: bool = False,
) -> Dict[str, Any]:
    package_path = Path(package_dir)
    if not package_path.exists():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")
    if not (package_path / "README.md").exists():
        raise FileNotFoundError(f"README.md missing from package directory: {package_dir}")
    if not (package_path / "Dockerfile").exists():
        raise FileNotFoundError(f"Dockerfile missing from package directory: {package_dir}")

    api = HfApi(token=token)
    if create_if_missing:
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token,
        )

    commit_message = commit_message or f"Deploy A_ML_25 Space package to {space_repo_id}"
    commit_info = api.upload_folder(
        folder_path=str(package_path),
        repo_id=space_repo_id,
        repo_type="space",
        token=token,
        commit_message=commit_message,
        delete_patterns="*",
    )

    result = {
        "space_repo_id": space_repo_id,
        "package_dir": str(package_path),
        "commit_message": commit_message,
        "commit_url": getattr(commit_info, "commit_url", None),
        "oid": getattr(commit_info, "oid", None),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a generated Docker Space package to Hugging Face Spaces")
    parser.add_argument("--package-dir", required=True, help="Directory created by create_hf_space_package.py")
    parser.add_argument("--space-repo-id", required=True, help="Hugging Face Space repo ID, e.g. user/space-name")
    parser.add_argument("--token", default=os.getenv("HF_SPACE_TOKEN", ""), help="Hugging Face write token")
    parser.add_argument("--commit-message", default="")
    parser.add_argument("--create-if-missing", action="store_true")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("No Hugging Face token provided. Use --token or set HF_SPACE_TOKEN.")

    publish_to_hf_space(
        package_dir=args.package_dir,
        space_repo_id=args.space_repo_id,
        token=args.token,
        commit_message=args.commit_message,
        create_if_missing=args.create_if_missing,
    )


if __name__ == "__main__":
    main()
