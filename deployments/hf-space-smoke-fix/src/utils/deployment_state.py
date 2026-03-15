from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .io import IO
from .model_bundle import load_bundle_manifest, resolve_bundle_path, validate_bundle
from .registry_loader import RegistryLoader


DEFAULT_DEPLOYMENT_MANIFEST_PATH = "experiments/registry/deployment_manifest.json"
DEFAULT_PRODUCTION_TRACKER_PATH = "experiments/registry/production_tracker.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    payload = IO.load_json(path)
    return payload if isinstance(payload, dict) else None


def _find_previous_production_candidate(loader: RegistryLoader, exclude_run_id: str) -> Optional[str]:
    candidates = [
        run
        for run in loader.list_runs()
        if run.get("run_id") != exclude_run_id and run.get("status") in {"production", "archived"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda run: str(run.get("updated_utc") or run.get("created_utc") or ""), reverse=True)
    return str(candidates[0].get("run_id")) if candidates[0].get("run_id") else None


def extract_slo_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}

    source = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    metrics: Dict[str, float] = {}

    for key in ("accuracy", "smape", "latency_p95", "error_rate", "success_rate", "throughput_qps"):
        value = source.get(key)
        if value is not None:
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue

    latency_ms = source.get("latency_ms")
    if "latency_p95" not in metrics and isinstance(latency_ms, dict) and latency_ms.get("p95") is not None:
        try:
            metrics["latency_p95"] = float(latency_ms["p95"]) / 1000.0
        except (TypeError, ValueError):
            pass

    request_count = source.get("request_count")
    error_count = source.get("error_count")
    if "error_rate" not in metrics and request_count is not None and error_count is not None:
        try:
            request_total = max(float(request_count), 1.0)
            metrics["error_rate"] = float(error_count) / request_total
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return metrics


def build_deployment_manifest(
    run_id: str,
    strategy: str,
    registry_dir: str = "experiments/registry",
    environment: str = "production",
    deployed_by: str = "automation",
    deployment_url: str = "",
    canary_percent: Optional[float] = None,
    status: str = "active",
    service_image: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    loader = RegistryLoader(registry_dir=registry_dir)
    entry = loader.get_run_by_id(run_id)
    if entry is None:
        raise ValueError(f"Run {run_id} not found in registry: {registry_dir}")

    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)
    bundle_validation = validate_bundle(bundle_path)
    if not bundle_validation["valid"]:
        raise ValueError(f"Bundle validation failed for {run_id}: {bundle_validation['problems']}")

    bundle_manifest = load_bundle_manifest(bundle_path)
    config_path = os.path.join(bundle_path, "config.json")
    active_production_run_id = loader.get_active_production_run_id()
    previous_run_id = _find_previous_production_candidate(loader, exclude_run_id=run_id)
    if active_production_run_id and active_production_run_id != run_id:
        previous_run_id = active_production_run_id

    return {
        "schema_version": 1,
        "run_id": run_id,
        "bundle_path": bundle_path,
        "bundle_manifest_path": os.path.join(bundle_path, "manifest.json"),
        "bundle_config_path": config_path,
        "bundle_config_sha256": _sha256_file(config_path),
        "registry_dir": registry_dir,
        "registry_manifest_path": entry.get("manifest_path"),
        "registry_status": entry.get("status"),
        "registry_stage": entry.get("stage"),
        "strategy": strategy,
        "environment": environment,
        "status": status,
        "deployed_at_utc": _utc_now(),
        "deployed_by": deployed_by,
        "deployment_url": deployment_url,
        "canary_percent": float(canary_percent) if canary_percent is not None else None,
        "service_image": service_image or os.getenv("SERVING_IMAGE", ""),
        "previous_production_run_id": previous_run_id,
        "tracking": entry.get("tracking", {}),
        "bundle_validation": bundle_validation,
        "bundle_manifest_summary": {
            "keys": sorted(bundle_manifest.keys()),
            "outputs_present": sorted((bundle_manifest.get("outputs") or {}).keys()),
        },
        "metadata": metadata or {},
    }


def write_deployment_manifest(
    run_id: str,
    strategy: str,
    manifest_path: str = DEFAULT_DEPLOYMENT_MANIFEST_PATH,
    registry_dir: str = "experiments/registry",
    environment: str = "production",
    deployed_by: str = "automation",
    deployment_url: str = "",
    canary_percent: Optional[float] = None,
    status: str = "active",
    service_image: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = build_deployment_manifest(
        run_id=run_id,
        strategy=strategy,
        registry_dir=registry_dir,
        environment=environment,
        deployed_by=deployed_by,
        deployment_url=deployment_url,
        canary_percent=canary_percent,
        status=status,
        service_image=service_image,
        metadata=metadata,
    )
    IO.save_json(manifest, manifest_path, indent=2)
    return manifest


def write_production_tracker(
    run_id: str,
    strategy: str,
    manifest_path: str = DEFAULT_PRODUCTION_TRACKER_PATH,
    registry_dir: str = "experiments/registry",
    deployment_manifest_path: str = DEFAULT_DEPLOYMENT_MANIFEST_PATH,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "active",
) -> Dict[str, Any]:
    loader = RegistryLoader(registry_dir=registry_dir)
    entry = loader.get_run_by_id(run_id)
    if entry is None:
        raise ValueError(f"Run {run_id} not found in registry: {registry_dir}")

    deployment_manifest = _load_json_if_exists(deployment_manifest_path)
    if deployment_manifest is None or deployment_manifest.get("run_id") != run_id:
        deployment_manifest = write_deployment_manifest(
            run_id=run_id,
            strategy=strategy,
            manifest_path=deployment_manifest_path,
            registry_dir=registry_dir,
            status=status,
        )

    existing = _load_json_if_exists(manifest_path) or {}
    tracker_metrics = metrics if isinstance(metrics, dict) else existing.get("metrics", {})
    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)

    tracker = {
        "schema_version": 1,
        "run_id": run_id,
        "bundle_path": bundle_path,
        "strategy": strategy,
        "status": status,
        "last_updated_utc": _utc_now(),
        "deployment_manifest_path": deployment_manifest_path,
        "deployment": {
            "deployed_at_utc": deployment_manifest.get("deployed_at_utc"),
            "environment": deployment_manifest.get("environment"),
            "deployed_by": deployment_manifest.get("deployed_by"),
            "deployment_url": deployment_manifest.get("deployment_url"),
            "service_image": deployment_manifest.get("service_image"),
            "bundle_config_sha256": deployment_manifest.get("bundle_config_sha256"),
            "previous_production_run_id": deployment_manifest.get("previous_production_run_id"),
        },
        "registry_consistency": {
            "active_production_run_id": loader.get_active_production_run_id(),
            "entry_status": entry.get("status"),
            "entry_stage": entry.get("stage"),
            "matches_active_production": loader.get_active_production_run_id() == run_id,
        },
        "metrics": tracker_metrics,
    }
    IO.save_json(tracker, manifest_path, indent=2)
    return tracker


def resolve_live_deployment_state(
    registry_dir: str = "experiments/registry",
    deployment_manifest_path: str = DEFAULT_DEPLOYMENT_MANIFEST_PATH,
    tracker_path: str = DEFAULT_PRODUCTION_TRACKER_PATH,
) -> Dict[str, Any]:
    loader = RegistryLoader(registry_dir=registry_dir)
    errors = []

    active_production_run_id = loader.get_active_production_run_id()
    active_entry = loader.get_active_production_entry()
    deployment_manifest = _load_json_if_exists(deployment_manifest_path)
    tracker = _load_json_if_exists(tracker_path)

    if not active_production_run_id:
        errors.append("missing_active_production_run_id")

    resolved_bundle_path = None
    if active_entry and active_entry.get("bundle_path"):
        resolved_bundle_path = str(active_entry.get("bundle_path"))
    elif deployment_manifest and deployment_manifest.get("bundle_path"):
        resolved_bundle_path = str(deployment_manifest.get("bundle_path"))
    elif tracker and tracker.get("bundle_path"):
        resolved_bundle_path = str(tracker.get("bundle_path"))

    if deployment_manifest and active_production_run_id and deployment_manifest.get("run_id") != active_production_run_id:
        errors.append(
            f"deployment_manifest_run_mismatch:{deployment_manifest.get('run_id')}!={active_production_run_id}"
        )
    if tracker and active_production_run_id and tracker.get("run_id") != active_production_run_id:
        errors.append(f"production_tracker_run_mismatch:{tracker.get('run_id')}!={active_production_run_id}")
    if deployment_manifest and tracker and deployment_manifest.get("run_id") != tracker.get("run_id"):
        errors.append(
            f"deployment_tracker_run_mismatch:{deployment_manifest.get('run_id')}!={tracker.get('run_id')}"
        )

    bundle_validation: Optional[Dict[str, Any]] = None
    if resolved_bundle_path:
        try:
            bundle_validation = validate_bundle(resolved_bundle_path)
            if not bundle_validation["valid"]:
                errors.extend(bundle_validation["problems"])
        except Exception as exc:
            errors.append(f"bundle_validation_error:{exc}")
    else:
        errors.append("missing_bundle_path")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "registry_dir": registry_dir,
        "active_production_run_id": active_production_run_id,
        "active_entry": active_entry,
        "bundle_path": resolved_bundle_path,
        "bundle_validation": bundle_validation,
        "deployment_manifest": deployment_manifest,
        "production_tracker": tracker,
    }
