"""
Monitor canary deployment metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.slo_validator import SLOValidator


def monitor_canary_metrics(
    duration: int = 60,
    sample_size: int = 100,
    alert_threshold_latency: float | None = None,
    alert_threshold_error_rate: float | None = None
) -> Dict[str, Any]:
    """
    Monitor canary deployment for anomalies.
    
    Args:
        duration: Monitoring duration in seconds
        sample_size: Number of requests to monitor
        alert_threshold_latency: Max acceptable P95 latency
        alert_threshold_error_rate: Max acceptable error rate
    
    Returns:
        Monitoring results
    """
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "duration_seconds": duration,
        "sample_size": sample_size,
        "metrics": {},
        "passed": True,
        "alerts": []
    }
    
    try:
        thresholds = SLOValidator().get_stage_thresholds("canary")
        latency_threshold = thresholds["max_latency_p95"] if alert_threshold_latency is None else alert_threshold_latency
        error_rate_threshold = thresholds["max_error_rate"] if alert_threshold_error_rate is None else alert_threshold_error_rate

        # Simulate monitoring (in production: use actual request logs)
        # In real system: query Prometheus, DataDog, CloudWatch, etc.
        
        result["metrics"] = {
            "p50_latency": 0.50,
            "p95_latency": 0.85,
            "p99_latency": 1.20,
            "error_rate": 0.001,
            "success_rate": 0.999,
            "throughput_qps": 150,
            "memory_usage_mb": 512,
            "cpu_usage_percent": 45
        }
        
        # Check thresholds
        if result["metrics"]["p95_latency"] > latency_threshold:
            result["alerts"].append(
                f"Latency high: {result['metrics']['p95_latency']}s > {latency_threshold}s"
            )
            result["passed"] = False
        
        if result["metrics"]["error_rate"] > error_rate_threshold:
            result["alerts"].append(
                f"Error rate high: {result['metrics']['error_rate']:.3f} > {error_rate_threshold}"
            )
            result["passed"] = False
        
        result["status"] = "healthy" if result["passed"] else "degraded"
        
    except Exception as e:
        result["alerts"].append(f"Monitoring error: {str(e)}")
        result["passed"] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Monitor canary deployment")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--alert-threshold-latency", type=float, default=None)
    parser.add_argument("--alert-threshold-error-rate", type=float, default=None)
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    result = monitor_canary_metrics(
        duration=args.duration,
        sample_size=args.sample_size,
        alert_threshold_latency=args.alert_threshold_latency,
        alert_threshold_error_rate=args.alert_threshold_error_rate
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))
    exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
