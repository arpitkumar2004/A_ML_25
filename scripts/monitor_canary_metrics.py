"""
Monitor canary deployment metrics using a real metrics snapshot artifact.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import extract_slo_metrics
from src.utils.io import IO
from src.utils.service_probe import try_probe_live_service
from src.validation.slo_validator import SLOValidator


def monitor_canary_metrics(
    duration: int = 60,
    sample_size: int = 100,
    alert_threshold_latency: float | None = None,
    alert_threshold_error_rate: float | None = None,
    metrics_input: str = "",
    service_base_url: str = "",
    expected_run_id: str = "",
) -> Dict[str, Any]:
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "duration_seconds": duration,
        "sample_size": sample_size,
        "metrics_source": metrics_input or None,
        "metrics": {},
        "passed": False,
        "alerts": [],
        "status": "unknown",
    }

    try:
        payload = None
        if service_base_url:
            live = try_probe_live_service(
                base_url=service_base_url,
                expected_run_id=expected_run_id or None,
            )
            if not live["reachable"] or not live["valid"]:
                raise ValueError(f"Live canary probe failed: {live['errors']}")
            payload = live["metrics_payload"]
            result["metrics_source"] = f"{service_base_url.rstrip('/')}/metrics/json"
        elif metrics_input:
            payload = IO.load_json(metrics_input)
        else:
            raise ValueError("No metrics_input or service_base_url provided for canary monitoring")

        metrics = extract_slo_metrics(payload)
        if not metrics:
            raise ValueError("No usable metrics found in canary metrics source")

        validator = SLOValidator()
        thresholds = validator.get_stage_thresholds("canary")
        latency_threshold = thresholds["max_latency_p95"] if alert_threshold_latency is None else alert_threshold_latency
        error_rate_threshold = thresholds["max_error_rate"] if alert_threshold_error_rate is None else alert_threshold_error_rate

        result["metrics"] = metrics
        latency = metrics.get("latency_p95")
        error_rate = metrics.get("error_rate")

        if latency is None:
            result["alerts"].append("Missing latency_p95 metric")
        elif latency > latency_threshold:
            result["alerts"].append(f"Latency high: {latency}s > {latency_threshold}s")

        if error_rate is None:
            result["alerts"].append("Missing error_rate metric")
        elif error_rate > error_rate_threshold:
            result["alerts"].append(f"Error rate high: {error_rate:.3f} > {error_rate_threshold}")

        result["passed"] = len(result["alerts"]) == 0
        result["status"] = "healthy" if result["passed"] else "degraded"
    except Exception as exc:
        result["alerts"].append(f"Monitoring error: {exc}")
        result["passed"] = False
        result["status"] = "degraded"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor canary deployment")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--alert-threshold-latency", type=float, default=None)
    parser.add_argument("--alert-threshold-error-rate", type=float, default=None)
    parser.add_argument("--metrics-input", default="")
    parser.add_argument("--service-base-url", default=os.getenv("CANARY_SERVICE_BASE_URL", ""))
    parser.add_argument("--expected-run-id", default="")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    result = monitor_canary_metrics(
        duration=args.duration,
        sample_size=args.sample_size,
        alert_threshold_latency=args.alert_threshold_latency,
        alert_threshold_error_rate=args.alert_threshold_error_rate,
        metrics_input=args.metrics_input,
        service_base_url=args.service_base_url,
        expected_run_id=args.expected_run_id,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
