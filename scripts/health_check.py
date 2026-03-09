"""
Health check system for production monitoring.
Verifies MLflow, registry, model inference, and deployment health.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import argparse


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
        registry_path = "experiments/registry/index.json"
        if not os.path.exists(registry_path):
            result["message"] = "Registry index not found"
            return result
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        production_run = registry.get("active_production_run_id")
        if not production_run:
            result["message"] = "No production model registered"
            return result
        
        # Verify model exists
        model_dir = f"experiments/models/{production_run}"
        if os.path.exists(model_dir):
            result["passed"] = True
            result["message"] = f"Production model loaded: {production_run}"
        else:
            result["message"] = f"Production model artifacts missing: {model_dir}"
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
        registry_path = "experiments/registry/index.json"
        if not os.path.exists(registry_path):
            result["message"] = "Registry index not found"
            return result
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        runs = registry.get("runs", [])
        production_run = registry.get("active_production_run_id")
        
        # Check for required states
        has_production = any(r.get("run_id") == production_run for r in runs if production_run)
        has_staging = any(r.get("status") == "staging" for r in runs)
        
        result["passed"] = has_production and has_staging
        result["message"] = f"Registry OK: {len(runs)} models, production={production_run}"
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
        # Import inference module
        from src.inference.predict import predict_batch
        
        # Create minimal test data
        test_data = {
            "text": ["sample product"],
            "images": [None],
            "numeric_features": [[1.0, 2.0, 3.0]]
        }
        
        # Run inference
        predictions = predict_batch(test_data)
        
        if predictions is not None and len(predictions) > 0:
            result["passed"] = True
            result["message"] = f"Inference successful. Sample prediction: {predictions[0]:.2f}"
        else:
            result["message"] = "Inference returned empty result"
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
