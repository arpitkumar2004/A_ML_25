"""
Health checks for deployment state, registry integrity, and bundle-backed inference.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import PredictPipeline
from src.utils.deployment_state import resolve_live_deployment_state
from src.utils.registry_loader import RegistryLoader
from src.utils.service_probe import try_probe_live_prediction, try_probe_live_service


def check_mlflow(uri: str) -> Dict[str, Any]:
    result = {
        "name": "MLflow Connectivity",
        "passed": False,
        "message": "",
    }
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        experiments = mlflow.search_experiments(max_results=1)
        result["passed"] = True
        result["message"] = f"Connected. {len(experiments)} experiment(s) accessible."
    except Exception as exc:
        result["message"] = f"Connection failed: {exc}"
    return result


def check_production_model_state(
    registry_dir: str,
    deployment_manifest_path: str,
    tracker_path: str,
    service_base_url: str = "",
) -> Dict[str, Any]:
    result = {
        "name": "Production Model State",
        "passed": False,
        "message": "",
    }
    try:
        state = resolve_live_deployment_state(
            registry_dir=registry_dir,
            deployment_manifest_path=deployment_manifest_path,
            tracker_path=tracker_path,
        )

        if service_base_url:
            live = try_probe_live_service(
                service_base_url,
                expected_run_id=state["active_production_run_id"],
            )
            if live["reachable"]:
                result["passed"] = live["valid"]
                result["message"] = (
                    f"Live service reports run {live['service_info'].get('run_id')}"
                    if live["valid"]
                    else f"Live service invalid: {live['errors']}"
                )
                return result

        result["passed"] = state["valid"]
        result["message"] = (
            f"Active production run {state['active_production_run_id']} is bundle-backed"
            if state["valid"]
            else f"Invalid live deployment state: {state['errors']}"
        )
    except Exception as exc:
        result["message"] = f"Error checking production model state: {exc}"
    return result


def check_registry(registry_dir: str) -> Dict[str, Any]:
    result = {
        "name": "Model Registry",
        "passed": False,
        "message": "",
    }
    try:
        loader = RegistryLoader(registry_dir=registry_dir)
        runs = loader.list_runs()
        production_run = loader.get_active_production_run_id()
        production_entry = loader.get_active_production_entry()

        result["passed"] = bool(runs) and bool(production_run) and production_entry is not None and bool(
            production_entry.get("bundle_path")
        )
        result["message"] = (
            f"Registry OK: {len(runs)} runs, active production={production_run}"
            if result["passed"]
            else f"Registry incomplete: runs={len(runs)}, active_production={production_run}"
        )
    except FileNotFoundError:
        result["message"] = "Registry index not found"
    except Exception as exc:
        result["message"] = f"Registry validation failed: {exc}"
    return result


def _load_sample_frame(sample_dataset: str = "") -> pd.DataFrame:
    if sample_dataset:
        frame = pd.read_csv(sample_dataset)
        if frame.empty:
            raise ValueError("Health check dataset is empty")
        return frame.head(1).copy()

    return pd.DataFrame(
        [
            {
                "sample_id": 1,
                "catalog_content": "health check sample product",
                "image_link": "",
            }
        ]
    )


def check_inference(
    registry_dir: str,
    deployment_manifest_path: str,
    tracker_path: str,
    sample_dataset: str = "",
    service_base_url: str = "",
) -> Dict[str, Any]:
    result = {
        "name": "Inference Smoke Test",
        "passed": False,
        "message": "",
    }
    try:
        state = resolve_live_deployment_state(
            registry_dir=registry_dir,
            deployment_manifest_path=deployment_manifest_path,
            tracker_path=tracker_path,
        )
        if not state["valid"] or not state.get("bundle_path"):
            result["message"] = f"Deployment state invalid: {state['errors']}"
            return result

        if service_base_url:
            live = try_probe_live_prediction(
                base_url=service_base_url,
                expected_run_id=state["active_production_run_id"],
            )
            result["passed"] = live["valid"]
            result["message"] = (
                f"Live inference path OK for run {state['active_production_run_id']}"
                if live["valid"]
                else f"Live service inference path invalid: {live['errors']}"
            )
            return result

        sample = _load_sample_frame(sample_dataset)
        text_col = "catalog_content" if "catalog_content" in sample.columns else "Description"
        image_col = "image_link" if "image_link" in sample.columns else "image_url"
        if image_col not in sample.columns:
            sample[image_col] = ""

        pipeline = PredictPipeline(bundle_path=state["bundle_path"], registry_dir=registry_dir)
        preds = pipeline.predict(sample, text_col=text_col, image_col=image_col, force_rebuild_features=True)

        result["passed"] = len(preds) == len(sample)
        result["message"] = (
            f"Inference OK for run {state['active_production_run_id']}"
            if result["passed"]
            else f"Prediction length mismatch: {len(preds)} vs {len(sample)}"
        )
    except Exception as exc:
        result["message"] = f"Inference test failed: {exc}"
    return result


def run_health_checks(
    do_check_mlflow: bool = True,
    include_production_model: bool = True,
    include_registry: bool = True,
    include_inference: bool = True,
    registry_dir: str = "experiments/registry",
    deployment_manifest_path: str = "experiments/registry/deployment_manifest.json",
    tracker_path: str = "experiments/registry/production_tracker.json",
    sample_dataset: str = "",
    service_base_url: str = "",
) -> Dict[str, Any]:
    checks = []
    passed_count = 0

    if do_check_mlflow:
        result = check_mlflow(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        checks.append(result)
        if result["passed"]:
            passed_count += 1

    if include_production_model:
        result = check_production_model_state(
            registry_dir=registry_dir,
            deployment_manifest_path=deployment_manifest_path,
            tracker_path=tracker_path,
            service_base_url=service_base_url,
        )
        checks.append(result)
        if result["passed"]:
            passed_count += 1

    if include_registry:
        result = check_registry(registry_dir=registry_dir)
        checks.append(result)
        if result["passed"]:
            passed_count += 1

    if include_inference:
        result = check_inference(
            registry_dir=registry_dir,
            deployment_manifest_path=deployment_manifest_path,
            tracker_path=tracker_path,
            sample_dataset=sample_dataset,
            service_base_url=service_base_url,
        )
        checks.append(result)
        if result["passed"]:
            passed_count += 1

    all_passed = passed_count == len(checks)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy" if all_passed else "degraded",
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "all_passed": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Health check system")
    parser.add_argument("--check-mlflow", action="store_true", default=False)
    parser.add_argument("--check-production-model", action="store_true", default=False)
    parser.add_argument("--check-registry", action="store_true", default=False)
    parser.add_argument("--check-inference", action="store_true", default=False)
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--deployment-manifest-path", default="experiments/registry/deployment_manifest.json")
    parser.add_argument("--tracker-path", default="experiments/registry/production_tracker.json")
    parser.add_argument("--sample-dataset", default="")
    parser.add_argument("--service-base-url", default=os.getenv("SERVICE_BASE_URL", ""))
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    result = run_health_checks(
        do_check_mlflow=args.check_mlflow,
        include_production_model=True if not any(
            [args.check_mlflow, args.check_production_model, args.check_registry, args.check_inference]
        ) else args.check_production_model,
        include_registry=True if not any(
            [args.check_mlflow, args.check_production_model, args.check_registry, args.check_inference]
        ) else args.check_registry,
        include_inference=True if not any(
            [args.check_mlflow, args.check_production_model, args.check_registry, args.check_inference]
        ) else args.check_inference,
        registry_dir=args.registry_dir,
        deployment_manifest_path=args.deployment_manifest_path,
        tracker_path=args.tracker_path,
        sample_dataset=args.sample_dataset,
        service_base_url=args.service_base_url,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
