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
        
        # Check accuracy
        accuracy = metrics.get("accuracy", 0)
        if accuracy < min_accuracy:
            result["failures"].append(
                f"Accuracy {accuracy:.3f} below threshold {min_accuracy}"
            )
        else:
            result["metrics"]["accuracy"] = accuracy
        
        # Check SMAPE if available
        smape = metrics.get("smape", None)
        if smape:
            result["metrics"]["smape"] = smape
            if target_stage == "production" and smape > 0.30:
                result["warnings"].append(
                    f"SMAPE {smape:.3f} is high for production"
                )
        
        # Check latency
        latency_p95 = metrics.get("latency_p95", 0)
        if latency_p95 > max_latency_p95:
            result["failures"].append(
                f"P95 latency {latency_p95:.3f}s exceeds threshold {max_latency_p95}s"
            )
        else:
            result["metrics"]["latency_p95"] = latency_p95
        
        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.05:
            result["failures"].append(
                f"Error rate {error_rate:.3f} exceeds 5%"
            )
        else:
            result["metrics"]["error_rate"] = error_rate
        
        # Get training info
        result["metrics"]["training_duration"] = metrics.get("training_duration_seconds", None)
        result["metrics"]["dataset_size"] = params.get("dataset_size", None)
        
        # Promotion eligibility
        if target_stage == "production":
            if len(result["failures"]) > 0:
                result["passed"] = False
            else:
                result["passed"] = len(result["failures"]) == 0
        else:
            # Staging/canary have looser requirements
            result["passed"] = error_rate <= 0.10
        
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
