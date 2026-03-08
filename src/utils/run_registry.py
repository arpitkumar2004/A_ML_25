from typing import Any, Dict, Optional
import os
from datetime import datetime, timezone

from .io import IO


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def write_run_manifest(
    run_id: str,
    stage: str,
    cfg: Dict[str, Any],
    outputs: Dict[str, Any],
    timings: Optional[Dict[str, float]] = None,
    registry_dir: str = "experiments/registry",
) -> str:
    os.makedirs(registry_dir, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "outputs": outputs,
        "timings_seconds": timings or {},
    }
    out_path = os.path.join(registry_dir, f"{run_id}_{stage}_manifest.json")
    IO.save_json(manifest, out_path, indent=2)
    return out_path


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    IO.ensure_dir(path)
    line = record.copy()
    line["ts_utc"] = datetime.now(timezone.utc).isoformat()
    import json

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=True) + "\n")
