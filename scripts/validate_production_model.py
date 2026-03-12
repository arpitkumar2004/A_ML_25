"""
Validate production model performance metrics.
"""

import json
import os
from typing import Dict, Any
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.slo_validator import SLOValidator


def validate_production_model(
    min_accuracy: float | None = None,
    max_latency_p95: float | None = None,
    max_error_rate: float | None = None
) -> Dict[str, Any]:
    """
    Validate that production model meets SLOs.
    
    Args:
        min_accuracy: Minimum accuracy threshold
        max_latency_p95: Maximum P95 latency (seconds)
        max_error_rate: Maximum error rate (0-1)
    
    Returns:
        Validation results
    """
    
    result = {
        "validation_passed": False,
        "checks": {},
        "failures": []
    }
    
    try:
        # Load production tracker
        tracker_path = "experiments/registry/production_tracker.json"
        if not os.path.exists(tracker_path):
            result["failures"].append("Production tracker not found")
            return result
        
        with open(tracker_path) as f:
            tracker = json.load(f)
        
        # Check production metrics (would come from monitoring system)
        metrics = tracker.get("metrics", {})

        validator = SLOValidator()
        validation = validator.validate_metrics(metrics=metrics, stage="production")

        # Backward-compatible CLI overrides
        if min_accuracy is not None:
            accuracy = metrics.get("accuracy", 0.0)
            passed = accuracy >= min_accuracy
            validation["checks"]["accuracy"] = {
                "actual": accuracy,
                "threshold": min_accuracy,
                "passed": passed,
            }
            if not passed:
                validation["failures"].append(f"Accuracy {accuracy:.3f} < {min_accuracy}")
        if max_latency_p95 is not None:
            latency = metrics.get("latency_p95", 0.0)
            passed = latency <= max_latency_p95
            validation["checks"]["latency_p95"] = {
                "actual": latency,
                "threshold": max_latency_p95,
                "passed": passed,
            }
            if not passed:
                validation["failures"].append(f"Latency {latency:.3f}s > {max_latency_p95}s")
        if max_error_rate is not None:
            error_rate = metrics.get("error_rate", 0.0)
            passed = error_rate <= max_error_rate
            validation["checks"]["error_rate"] = {
                "actual": error_rate,
                "threshold": max_error_rate,
                "passed": passed,
            }
            if not passed:
                validation["failures"].append(f"Error rate {error_rate:.3f} > {max_error_rate}")

        result["checks"] = validation["checks"]
        result["failures"] = validation["failures"]
        result["validation_passed"] = len(result["failures"]) == 0
        
    except Exception as e:
        result["failures"].append(f"Validation error: {str(e)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate production model")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-latency-p95", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=None)
    
    args = parser.parse_args()
    
    result = validate_production_model(
        min_accuracy=args.min_accuracy,
        max_latency_p95=args.max_latency_p95,
        max_error_rate=args.max_error_rate
    )
    
    print(json.dumps(result, indent=2))
    exit(0 if result["validation_passed"] else 1)


if __name__ == "__main__":
    main()
