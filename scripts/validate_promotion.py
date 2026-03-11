"""
Model validation and promotion checking.
Verifies model is production-ready before promotion.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import argparse


def validate_promotion(
    run_id: str,
    target_stage: str,
    min_accuracy: float = 0.70,
    max_latency_p95: float = 2.0
) -> Dict[str, Any]:
    """
    Validate model metrics are acceptable for promotion.
    
    Args:
        run_id: MLflow run ID
        target_stage: Target stage (staging, canary, production)
        min_accuracy: Minimum acceptable accuracy
        max_latency_p95: Maximum acceptable P95 latency (seconds)
    """
    
    result = {
        "run_id": run_id,
        "target_stage": target_stage,
        "passed": False,
        "metrics": {},
        "failures": [],
        "warnings": []
    }
    
    try:
        import mlflow
        
        # Load metrics from MLflow
        run = mlflow.get_run(run_id)
        if not run:
            result["failures"].append(f"Run {run_id} not found in MLflow")
            return result
        
        metrics = run.data.metrics or {}
        params = run.data.params or {}
        
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            if accuracy < min_accuracy:
                result["failures"].append(
                    f"Accuracy {accuracy:.3f} below threshold {min_accuracy}"
                )
            else:
                result["metrics"]["accuracy"] = accuracy
        else:
            result["warnings"].append("Accuracy metric not logged; promotion will rely on SMAPE-based validation")

        best_smape = metrics.get("best.smape", metrics.get("smape"))
        if best_smape is not None:
            result["metrics"]["smape"] = best_smape
            if target_stage == "production" and best_smape > 25.0:
                result["failures"].append(
                    f"Best SMAPE {best_smape:.3f} exceeds production threshold 25.0"
                )
        else:
            result["warnings"].append("SMAPE metric not logged in MLflow")
        
        # Check latency
        latency_p95 = metrics.get("latency_p95")
        if latency_p95 is not None:
            if latency_p95 > max_latency_p95:
                result["failures"].append(
                    f"P95 latency {latency_p95:.3f}s exceeds threshold {max_latency_p95}s"
                )
            else:
                result["metrics"]["latency_p95"] = latency_p95
        else:
            result["warnings"].append("Latency metric not logged; skipping latency gate")
        
        # Check error rate
        error_rate = metrics.get("error_rate")
        if error_rate is not None:
            if error_rate > 0.05:
                result["failures"].append(
                    f"Error rate {error_rate:.3f} exceeds 5%"
                )
            else:
                result["metrics"]["error_rate"] = error_rate
        else:
            result["warnings"].append("Error rate metric not logged; skipping error-rate gate")
        
        # Get training info
        result["metrics"]["training_duration"] = metrics.get("training_duration_seconds", None)
        result["metrics"]["dataset_size"] = params.get("dataset_size", None)
        
        # Promotion eligibility
        if target_stage == "production":
            result["passed"] = len(result["failures"]) == 0
        else:
            result["passed"] = len(result["failures"]) == 0
        
    except Exception as e:
        result["failures"].append(f"Validation error: {str(e)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate model for promotion")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--target-stage", required=True, help="Target stage")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument("--max-latency-p95", type=float, default=2.0)
    
    args = parser.parse_args()
    
    result = validate_promotion(
        run_id=args.run_id,
        target_stage=args.target_stage,
        min_accuracy=args.min_accuracy,
        max_latency_p95=args.max_latency_p95
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))
    exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
