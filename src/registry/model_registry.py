from typing import Any, Dict, List, Optional
import os
from datetime import datetime, timezone

from ..utils.io import IO


def _index_path(registry_dir: str) -> str:
    return os.path.join(registry_dir, "index.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index(registry_dir: str) -> Dict[str, Any]:
    path = _index_path(registry_dir)
    if os.path.exists(path):
        return IO.load_json(path)
    return {"runs": [], "active_production_run_id": None}


def _save_index(registry_dir: str, index: Dict[str, Any]) -> str:
    path = _index_path(registry_dir)
    IO.save_json(index, path, indent=2)
    return path


def register_run(
    run_id: str,
    manifest_path: str,
    stage: str,
    registry_dir: str = "experiments/registry",
    status: Optional[str] = None,
    tracking: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(registry_dir, exist_ok=True)
    idx = _load_index(registry_dir)
    runs: List[Dict[str, Any]] = idx.get("runs", [])

    default_status = status or ("staging" if stage == "train" else "recorded")
    found = False
    for r in runs:
        if r.get("run_id") == run_id:
            r.update(
                {
                    "manifest_path": manifest_path,
                    "stage": stage,
                    "status": r.get("status", default_status),
                    "tracking": tracking or r.get("tracking", {}),
                    "updated_utc": _utc_now(),
                }
            )
            found = True
            break

    if not found:
        runs.append(
            {
                "run_id": run_id,
                "manifest_path": manifest_path,
                "stage": stage,
                "status": default_status,
                "tracking": tracking or {},
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
            }
        )

    idx["runs"] = runs
    return _save_index(registry_dir, idx)


def list_runs(registry_dir: str = "experiments/registry") -> List[Dict[str, Any]]:
    idx = _load_index(registry_dir)
    return idx.get("runs", [])


def get_active_production(registry_dir: str = "experiments/registry") -> Optional[str]:
    idx = _load_index(registry_dir)
    return idx.get("active_production_run_id")


def promote_run(run_id: str, target_stage: str = "production", registry_dir: str = "experiments/registry") -> str:
    if target_stage not in {"staging", "canary", "production"}:
        raise ValueError("target_stage must be one of: staging, canary, production")

    idx = _load_index(registry_dir)
    runs: List[Dict[str, Any]] = idx.get("runs", [])
    if not any(r.get("run_id") == run_id for r in runs):
        raise ValueError(f"run_id '{run_id}' not found in registry")

    for r in runs:
        if r.get("run_id") == run_id:
            r["status"] = target_stage
            r["updated_utc"] = _utc_now()
        elif target_stage == "production" and r.get("status") == "production":
            r["status"] = "archived"
            r["updated_utc"] = _utc_now()

    if target_stage == "production":
        idx["active_production_run_id"] = run_id

    idx["runs"] = runs
    return _save_index(registry_dir, idx)


def rollback_to_run(run_id: str, registry_dir: str = "experiments/registry") -> str:
    idx = _load_index(registry_dir)
    runs: List[Dict[str, Any]] = idx.get("runs", [])
    if not any(r.get("run_id") == run_id for r in runs):
        raise ValueError(f"run_id '{run_id}' not found in registry")

    for r in runs:
        if r.get("status") == "production" and r.get("run_id") != run_id:
            r["status"] = "archived"
            r["updated_utc"] = _utc_now()
        if r.get("run_id") == run_id:
            r["status"] = "production"
            r["updated_utc"] = _utc_now()

    idx["active_production_run_id"] = run_id
    idx["runs"] = runs
    return _save_index(registry_dir, idx)
