from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from typing import Any, Dict

from .model_bundle import resolve_bundle_path, validate_bundle


def bundle_archive_name(run_id: str) -> str:
    return f"{run_id}-bundle.zip"


def bundle_release_tag(run_id: str, tag_prefix: str = "bundle-archive") -> str:
    return f"{tag_prefix}-{run_id}"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle_archive(
    run_id: str,
    registry_dir: str = "experiments/registry",
    output_dir: str = "dist/bundle-archives",
) -> Dict[str, Any]:
    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)
    bundle_validation = validate_bundle(bundle_path)
    if not bundle_validation["valid"]:
        raise ValueError(
            f"Cannot archive invalid bundle for {run_id}: {json.dumps(bundle_validation['problems'])}"
        )

    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, bundle_archive_name(run_id))
    prefix = os.path.join("experiments", "runs", run_id, "bundle")

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_path):
            for name in files:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, bundle_path)
                arcname = os.path.join(prefix, rel_path).replace(os.sep, "/")
                zf.write(abs_path, arcname=arcname)

    return {
        "run_id": run_id,
        "bundle_path": bundle_path,
        "archive_path": archive_path,
        "archive_name": os.path.basename(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": os.path.getsize(archive_path),
    }


def restore_bundle_archive(
    run_id: str,
    archive_path: str,
    output_root: str = ".",
    overwrite: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Bundle archive not found: {archive_path}")

    output_root = os.path.abspath(output_root)
    expected_bundle = os.path.join(output_root, "experiments", "runs", run_id, "bundle")

    if overwrite and os.path.isdir(expected_bundle):
        shutil.rmtree(expected_bundle)

    os.makedirs(output_root, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_root)

    validation = validate_bundle(expected_bundle)
    if not validation["valid"]:
        raise ValueError(f"Restored bundle is invalid for {run_id}: {json.dumps(validation['problems'])}")

    return {
        "run_id": run_id,
        "archive_path": archive_path,
        "bundle_path": expected_bundle,
        "archive_sha256": sha256_file(archive_path),
        "validation": validation,
    }
