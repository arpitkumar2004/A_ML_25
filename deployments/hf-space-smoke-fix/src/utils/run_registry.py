from typing import Any, Dict, Optional
import os
import json
import hashlib
from datetime import datetime, timezone

from .io import IO


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def _stable_config_hash(cfg: Dict[str, Any]) -> str:
    """Return deterministic hash for config lineage tracking in manifests."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_run_manifest(
    run_id: str,
    stage: str,
    cfg: Dict[str, Any],
    outputs: Dict[str, Any],
    timings: Optional[Dict[str, float]] = None,
    registry_dir: str = "experiments/registry",
    out_path: Optional[str] = None,
) -> str:
    os.makedirs(registry_dir, exist_ok=True)
    seed = cfg.get("seed", cfg.get("random_state", cfg.get("trainer", {}).get("random_state"))) if isinstance(cfg, dict) else None
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "config_sha256": _stable_config_hash(cfg),
            "pythonhashseed": os.getenv("PYTHONHASHSEED", ""),
            "seed": seed,
        },
        "config": cfg,
        "outputs": outputs,
        "timings_seconds": timings or {},
    }
    manifest_path = out_path or os.path.join(registry_dir, f"{run_id}_{stage}_manifest.json")
    IO.save_json(manifest, manifest_path, indent=2)
    return manifest_path


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    IO.ensure_dir(path)
    line = record.copy()
    line["ts_utc"] = datetime.now(timezone.utc).isoformat()
    import json

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=True) + "\n")
