from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class RegistryLoader:
    def __init__(self, registry_dir: str = "experiments/registry"):
        self.registry_dir = registry_dir
        self.index_path = os.path.join(self.registry_dir, "index.json")

    def load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Registry index not found: {self.index_path}")
        with open(self.index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Registry index must be a JSON object")
        return payload

    def list_runs(self) -> List[Dict[str, Any]]:
        payload = self.load_index()
        runs = payload.get("runs", [])
        return runs if isinstance(runs, list) else []

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        for run in self.list_runs():
            if run.get("run_id") == run_id:
                return run
        return None

    def get_active_production_run_id(self) -> Optional[str]:
        payload = self.load_index()
        run_id = payload.get("active_production_run_id")
        return run_id if run_id else None

    def get_active_production_entry(self) -> Optional[Dict[str, Any]]:
        run_id = self.get_active_production_run_id()
        if not run_id:
            return None
        return self.get_run_by_id(run_id)

    def get_bundle_path_for_run(self, run_id: str) -> Optional[str]:
        entry = self.get_run_by_id(run_id)
        if entry is None:
            return None
        bundle_path = entry.get("bundle_path")
        return str(bundle_path) if bundle_path else None

    def find_run_by_mlflow_run_id(self, mlflow_run_id: str) -> Optional[Dict[str, Any]]:
        for run in self.list_runs():
            tracking = run.get("tracking", {})
            if not isinstance(tracking, dict):
                continue
            mlflow_meta = tracking.get("mlflow", {})
            if not isinstance(mlflow_meta, dict):
                continue
            if str(mlflow_meta.get("mlflow_run_id") or "") == str(mlflow_run_id):
                return run
        return None

    def list_bundle_backed_runs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        runs = [run for run in self.list_runs() if run.get("bundle_path")]
        runs.sort(key=lambda run: str(run.get("updated_utc") or run.get("created_utc") or ""), reverse=True)
        if limit is not None:
            return runs[:limit]
        return runs

    def resolve_bundle_backed_run_id(self, candidate_run_id: str) -> Optional[str]:
        entry = self.get_run_by_id(candidate_run_id)
        if entry is not None and entry.get("bundle_path"):
            return str(entry.get("run_id"))

        mlflow_match = self.find_run_by_mlflow_run_id(candidate_run_id)
        if mlflow_match is not None and mlflow_match.get("bundle_path"):
            return str(mlflow_match.get("run_id"))

        return None

    def get_active_production_bundle_path(self) -> Optional[str]:
        entry = self.get_active_production_entry()
        if entry is None:
            return None
        bundle_path = entry.get("bundle_path")
        return str(bundle_path) if bundle_path else None
