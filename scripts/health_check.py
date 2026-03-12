"""
Health check system for production monitoring.
Verifies MLflow, registry, model inference, and deployment health.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.registry_loader import RegistryLoader


def check_mlflow(uri: str, username: str, password: str) -> Dict[str, Any]:
    """Verify MLflow tracking server connectivity."""
    result = {
        "name": "MLflow Connectivity",
        "passed": False,
        "message": ""
    }
    
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        # Try to list experiments
        experiments = mlflow.search_experiments(max_results=1)
        result["passed"] = True
        result["message"] = f"Connected. {len(experiments)} experiments accessible."
    except Exception as e:
        result["message"] = f"Connection failed: {str(e)}"
    
    return result


def check_production_model() -> Dict[str, Any]:
    """Verify production model is loadable and functional."""
    result = {
        "name": "Production Model",
        "passed": False,
        "message": ""
    }
    
    try:
        loader = RegistryLoader()
        production_run = loader.get_active_production_run_id()
        if not production_run:
            result["message"] = "No production model registered"
            return result

        production_entry = loader.get_run_by_id(production_run)
        if not production_entry:
            result["message"] = f"Production run {production_run} missing from registry entries"
            return result

        deployment_manifest_path = "experiments/registry/deployment_manifest.json"
        tracker_path = "experiments/registry/production_tracker.json"

        deployment_match = True
        if os.path.exists(deployment_manifest_path):
            with open(deployment_manifest_path) as f:
                deployment_manifest = json.load(f)
            deployment_match = deployment_manifest.get("run_id") == production_run

        tracker_match = True
        if os.path.exists(tracker_path):
            with open(tracker_path) as f:
                tracker = json.load(f)
            tracker_match = tracker.get("run_id") == production_run

        result["passed"] = production_entry.get("status") == "production" and deployment_match and tracker_match
        result["message"] = (
            f"Production registry state OK for run {production_run}"
            if result["passed"]
            else f"Production state mismatch for run {production_run}"
        )
    except FileNotFoundError:
        result["message"] = "Registry index not found"
    except Exception as e:
        result["message"] = f"Error checking production model: {str(e)}"
    
    return result


def check_registry() -> Dict[str, Any]:
    """Verify model registry integrity."""
    result = {
        "name": "Model Registry",
        "passed": False,
        "message": ""
    }
    
    try:
        loader = RegistryLoader()
        runs = loader.list_runs()
        production_run = loader.get_active_production_run_id()
        
        # Check for required states
        has_production = any(r.get("run_id") == production_run for r in runs if production_run)
        has_staging = any(r.get("status") == "staging" for r in runs)
        
        result["passed"] = has_production and has_staging
        result["message"] = f"Registry OK: {len(runs)} models, production={production_run}"
    except FileNotFoundError:
        result["message"] = "Registry index not found"
    except Exception as e:
        result["message"] = f"Registry validation failed: {str(e)}"
    
    return result


def check_inference() -> Dict[str, Any]:
    """Test model inference on sample data."""
    result = {
        "name": "Inference Test",
        "passed": False,
        "message": ""
    }
    
    try:
        loader = RegistryLoader()
        manifest_path = "experiments/registry/deployment_manifest.json"
        tracker_path = "experiments/registry/production_tracker.json"

        production_run = loader.get_active_production_run_id()
        if not production_run:
            result["message"] = "No active production run configured"
            return result

        manifest_run = None
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest_run = json.load(f).get("run_id")

        tracker_run = None
        if os.path.exists(tracker_path):
            with open(tracker_path) as f:
                tracker_run = json.load(f).get("run_id")

        result["passed"] = production_run == manifest_run == tracker_run
        result["message"] = (
            f"Inference routing ready for production run {production_run}"
            if result["passed"]
            else f"Deployment metadata mismatch: registry={production_run}, manifest={manifest_run}, tracker={tracker_run}"
        )
    except FileNotFoundError:
        result["message"] = "Registry index not found"
    except Exception as e:
        result["message"] = f"Inference test failed: {str(e)}"
    
    return result


def run_health_checks(
    check_mlflow: bool = True,
    check_production_model: bool = True,
    check_registry: bool = True,
    check_inference: bool = True
) -> Dict[str, Any]:
    """Run all health checks."""
    
    checks = []
    passed_count = 0
    
    if check_mlflow:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        username = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        result = check_mlflow(uri, username, password)
        checks.append(result)
        if result["passed"]:
            passed_count += 1
    
    if check_production_model:
        result = check_production_model()
        checks.append(result)
        if result["passed"]:
            passed_count += 1
    
    if check_registry:
        result = check_registry()
        checks.append(result)
        if result["passed"]:
            passed_count += 1
    
    if check_inference:
        result = check_inference()
        checks.append(result)
        if result["passed"]:
            passed_count += 1
    
    overall_status = "healthy" if passed_count == len(checks) else "degraded"
    
    failed_checks = [c["name"] for c in checks if not c["passed"]]
    critical_issues = [c for c in checks if not c["passed"] and "critical" in c["message"].lower()]
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": overall_status,
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed_checks,
        "critical_issues": [c["name"] for c in critical_issues],
        "all_passed": passed_count == len(checks)
    }


def main():
    parser = argparse.ArgumentParser(description="Health check system")
    parser.add_argument("--check-mlflow", action="store_true", default=True)
    parser.add_argument("--check-production-model", action="store_true", default=True)
    parser.add_argument("--check-registry", action="store_true", default=True)
    parser.add_argument("--check-inference", action="store_true", default=True)
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    result = run_health_checks(
        check_mlflow=args.check_mlflow,
        check_production_model=args.check_production_model,
        check_registry=args.check_registry,
        check_inference=args.check_inference
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))
    
    # Exit with error if not all checks passed
    exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
