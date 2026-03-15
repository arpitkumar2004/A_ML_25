from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

from .io import IO


def bundle_root(run_id: str, experiments_dir: str = "experiments") -> str:
    return os.path.join(experiments_dir, "runs", run_id, "bundle")


def bundle_layout(run_id: str, experiments_dir: str = "experiments") -> Dict[str, str]:
    root = bundle_root(run_id=run_id, experiments_dir=experiments_dir)
    artifacts_dir = os.path.join(root, "artifacts")
    models_dir = os.path.join(root, "models")
    oof_dir = os.path.join(root, "oof")
    reports_dir = os.path.join(root, "reports")
    schema_dir = os.path.join(root, "schema")
    return {
        "bundle_path": root,
        "manifest_path": os.path.join(root, "manifest.json"),
        "config_path": os.path.join(root, "config.json"),
        "artifacts_dir": artifacts_dir,
        "models_dir": models_dir,
        "oof_dir": oof_dir,
        "reports_dir": reports_dir,
        "schema_dir": schema_dir,
        "dim_cache": os.path.join(artifacts_dir, "dimred.joblib"),
        "feature_cache": os.path.join(artifacts_dir, "features.joblib"),
        "selector_path": os.path.join(artifacts_dir, "feature_selector.joblib"),
        "numeric_scaler_path": os.path.join(artifacts_dir, "numeric_scaler.joblib"),
        "text_vectorizer_path": os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"),
        "text_embeddings_cache": os.path.join(artifacts_dir, "text_embeddings.joblib"),
        "image_embeddings_cache": os.path.join(artifacts_dir, "image_embeddings.joblib"),
        "post_log_transform_path": os.path.join(artifacts_dir, "post_feature_log_transform.joblib"),
        "model_report_csv": os.path.join(reports_dir, "model_comparison.csv"),
        "model_report_joblib": os.path.join(reports_dir, "model_comparison.joblib"),
        "dim_meta_path": os.path.join(reports_dir, "dim_meta.joblib"),
        "stacker_summary_path": os.path.join(reports_dir, "stacker_summary.joblib"),
        "oof_path": os.path.join(oof_dir, "oof_matrix.joblib"),
        "oof_meta_path": os.path.join(oof_dir, "model_names.joblib"),
        "stacker_path": os.path.join(models_dir, "stacker.joblib"),
    }


def ensure_bundle_layout(run_id: str, experiments_dir: str = "experiments") -> Dict[str, str]:
    layout = bundle_layout(run_id=run_id, experiments_dir=experiments_dir)
    for key in ("bundle_path", "artifacts_dir", "models_dir", "oof_dir", "reports_dir", "schema_dir"):
        os.makedirs(layout[key], exist_ok=True)
    return layout


def copy_file_if_exists(src: Optional[str], dst: str) -> bool:
    if not src or not os.path.exists(src) or not os.path.isfile(src):
        return False
    IO.ensure_dir(dst)
    if os.path.abspath(src) == os.path.abspath(dst):
        return True
    shutil.copy2(src, dst)
    return True


def copy_tree_contents(src_dir: Optional[str], dst_dir: str) -> int:
    if not src_dir or not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied += 1
        elif os.path.isfile(src):
            copy_file_if_exists(src, dst)
            copied += 1
    return copied


def load_bundle_manifest(bundle_path: str) -> Dict[str, Any]:
    manifest_path = os.path.join(bundle_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    return IO.load_json(manifest_path)


def _normalize_bundle_meta_path(bundle_path: str, candidate: Optional[str], fallback: str) -> str:
    if not candidate:
        return fallback

    normalized = str(candidate).replace("\\", "/").strip()
    suffix = ""
    if normalized.endswith("/bundle"):
        suffix = ""
    elif "/bundle/" in normalized:
        suffix = normalized.rsplit("/bundle/", 1)[1]

    if suffix:
        return os.path.normpath(os.path.join(bundle_path, *[part for part in suffix.split("/") if part]))

    if os.path.isabs(normalized):
        return os.path.normpath(normalized)

    return os.path.normpath(os.path.join(bundle_path, *[part for part in normalized.split("/") if part]))


def bundle_runtime_contract(bundle_path: str) -> Dict[str, str]:
    manifest = load_bundle_manifest(bundle_path)
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    bundle_meta = outputs.get("bundle", {}) if isinstance(outputs.get("bundle", {}), dict) else {}
    fallback_layout = bundle_layout(run_id="placeholder")
    fallback_suffixes = {
        "models_dir": os.path.relpath(fallback_layout["models_dir"], fallback_layout["bundle_path"]),
        "oof_meta_path": os.path.relpath(fallback_layout["oof_meta_path"], fallback_layout["bundle_path"]),
        "stacker_path": os.path.relpath(fallback_layout["stacker_path"], fallback_layout["bundle_path"]),
        "dim_cache": os.path.relpath(fallback_layout["dim_cache"], fallback_layout["bundle_path"]),
        "feature_cache": os.path.relpath(fallback_layout["feature_cache"], fallback_layout["bundle_path"]),
        "selector_path": os.path.relpath(fallback_layout["selector_path"], fallback_layout["bundle_path"]),
        "numeric_scaler_path": os.path.relpath(fallback_layout["numeric_scaler_path"], fallback_layout["bundle_path"]),
        "text_vectorizer_path": os.path.relpath(fallback_layout["text_vectorizer_path"], fallback_layout["bundle_path"]),
        "post_log_transform_path": os.path.relpath(fallback_layout["post_log_transform_path"], fallback_layout["bundle_path"]),
    }
    return {
        "bundle_path": bundle_path,
        "manifest_path": os.path.join(bundle_path, "manifest.json"),
        "models_dir": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("models_dir"),
            os.path.join(bundle_path, fallback_suffixes["models_dir"]),
        ),
        "oof_meta_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("oof_meta_path"),
            os.path.join(bundle_path, fallback_suffixes["oof_meta_path"]),
        ),
        "stacker_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("stacker_path"),
            os.path.join(bundle_path, fallback_suffixes["stacker_path"]),
        ),
        "dim_cache": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("dim_cache") or outputs.get("dim_cache"),
            os.path.join(bundle_path, fallback_suffixes["dim_cache"]),
        ),
        "feature_cache": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("feature_cache") or outputs.get("feature_cache"),
            os.path.join(bundle_path, fallback_suffixes["feature_cache"]),
        ),
        "selector_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("selector_path") or outputs.get("selector_path"),
            os.path.join(bundle_path, fallback_suffixes["selector_path"]),
        ),
        "numeric_scaler_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("numeric_scaler_path") or outputs.get("numeric_scaler_path"),
            os.path.join(bundle_path, fallback_suffixes["numeric_scaler_path"]),
        ),
        "text_vectorizer_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("text_vectorizer_path"),
            os.path.join(bundle_path, fallback_suffixes["text_vectorizer_path"]),
        ),
        "post_log_transform_path": _normalize_bundle_meta_path(
            bundle_path,
            bundle_meta.get("post_log_transform_path"),
            os.path.join(bundle_path, fallback_suffixes["post_log_transform_path"]),
        ),
    }


def resolve_bundle_path(
    bundle_path: Optional[str] = None,
    run_id: Optional[str] = None,
    registry_dir: str = "experiments/registry",
    require_exists: bool = True,
) -> Optional[str]:
    if bundle_path:
        if require_exists and not os.path.isdir(bundle_path):
            raise FileNotFoundError(f"Bundle path not found: {bundle_path}")
        return bundle_path

    if not run_id:
        return None

    from .registry_loader import RegistryLoader

    loader = RegistryLoader(registry_dir=registry_dir)
    entry = loader.get_run_by_id(run_id)
    if entry is None:
        raise ValueError(f"Run {run_id} not found in registry: {registry_dir}")

    resolved = entry.get("bundle_path")
    if not resolved:
        manifest_path = entry.get("manifest_path")
        if manifest_path and os.path.exists(manifest_path):
            try:
                manifest = IO.load_json(manifest_path)
                outputs = manifest.get("outputs", {})
                bundle_meta = outputs.get("bundle", {}) if isinstance(outputs, dict) else {}
                resolved = bundle_meta.get("bundle_path")
            except Exception:
                resolved = None

    if not resolved:
        raise ValueError(f"Run {run_id} does not have a bundle_path in registry or manifest")

    if require_exists and not os.path.isdir(resolved):
        raise FileNotFoundError(f"Resolved bundle path does not exist: {resolved}")
    return str(resolved)


def validate_bundle(bundle_path: str) -> Dict[str, Any]:
    contract = bundle_runtime_contract(bundle_path)
    required = {
        "manifest_path": contract["manifest_path"],
        "models_dir": contract["models_dir"],
        "numeric_scaler_path": contract["numeric_scaler_path"],
    }
    problems = []
    for key, path in required.items():
        if key.endswith("_dir"):
            if not os.path.isdir(path):
                problems.append(f"missing:{key}:{path}")
        else:
            if not os.path.exists(path):
                problems.append(f"missing:{key}:{path}")
    return {
        "bundle_path": bundle_path,
        "valid": len(problems) == 0,
        "problems": problems,
        "contract": contract,
    }
