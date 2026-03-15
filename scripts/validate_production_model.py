"""
Validate production model metrics against production SLOs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import extract_slo_metrics, resolve_live_deployment_state
from src.utils.io import IO
from src.utils.service_probe import try_probe_live_service
from src.validation.slo_validator import SLOValidator


def validate_production_model(
    min_accuracy: float | None = None,
    max_latency_p95: float | None = None,
    max_error_rate: float | None = None,
    registry_dir: str = "experiments/registry",
    deployment_manifest_path: str = "experiments/registry/deployment_manifest.json",
    tracker_path: str = "experiments/registry/production_tracker.json",
    metrics_input: str = "",
    service_base_url: str = "",
) -> Dict[str, Any]:
    result = {
        "validation_passed": False,
        "run_id": None,
        "bundle_path": None,
        "metrics_source": None,
        "checks": {},
        "failures": [],
    }

    try:
        state = resolve_live_deployment_state(
            registry_dir=registry_dir,
            deployment_manifest_path=deployment_manifest_path,
            tracker_path=tracker_path,
        )
        if not state["valid"]:
            result["failures"].append(f"Live deployment state invalid: {state['errors']}")
            return result

        result["run_id"] = state["active_production_run_id"]
        result["bundle_path"] = state["bundle_path"]

        metrics_payload: Dict[str, Any] | None = None
        if metrics_input:
            metrics_payload = IO.load_json(metrics_input)
            result["metrics_source"] = metrics_input
        elif service_base_url:
            live = try_probe_live_service(
                base_url=service_base_url,
                expected_run_id=state["active_production_run_id"],
            )
            if not live["reachable"] or not live["valid"]:
                result["failures"].append(f"Live service metrics unavailable: {live['errors']}")
                return result
            metrics_payload = live["metrics_payload"]
            result["metrics_source"] = f"{service_base_url.rstrip('/')}/metrics/json"
        elif state.get("production_tracker"):
            metrics_payload = state["production_tracker"]
            result["metrics_source"] = tracker_path

        metrics = extract_slo_metrics(metrics_payload)
        if not metrics:
            result["failures"].append("No production metrics available for validation")
            return result

        validator = SLOValidator()
        validation = validator.validate_metrics(metrics=metrics, stage="production")

        if min_accuracy is not None:
            accuracy = metrics.get("accuracy")
            if accuracy is None:
                result["failures"].append("Accuracy override requested but no accuracy metric is available")
            elif accuracy < min_accuracy:
                result["failures"].append(f"Accuracy {accuracy:.3f} < {min_accuracy}")
            validation["checks"]["accuracy_override"] = {
                "actual": accuracy,
                "threshold": min_accuracy,
                "passed": accuracy is not None and accuracy >= min_accuracy,
            }

        if max_latency_p95 is not None:
            latency = metrics.get("latency_p95")
            if latency is None:
                result["failures"].append("Latency override requested but no latency_p95 metric is available")
            elif latency > max_latency_p95:
                result["failures"].append(f"Latency {latency:.3f}s > {max_latency_p95}s")
            validation["checks"]["latency_override"] = {
                "actual": latency,
                "threshold": max_latency_p95,
                "passed": latency is not None and latency <= max_latency_p95,
            }

        if max_error_rate is not None:
            error_rate = metrics.get("error_rate")
            if error_rate is None:
                result["failures"].append("Error-rate override requested but no error_rate metric is available")
            elif error_rate > max_error_rate:
                result["failures"].append(f"Error rate {error_rate:.3f} > {max_error_rate}")
            validation["checks"]["error_rate_override"] = {
                "actual": error_rate,
                "threshold": max_error_rate,
                "passed": error_rate is not None and error_rate <= max_error_rate,
            }

        result["checks"] = validation["checks"]
        result["failures"].extend(validation["failures"])
        result["validation_passed"] = len(result["failures"]) == 0
    except Exception as exc:
        result["failures"].append(f"Validation error: {exc}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate production model")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-latency-p95", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=None)
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--deployment-manifest-path", default="experiments/registry/deployment_manifest.json")
    parser.add_argument("--tracker-path", default="experiments/registry/production_tracker.json")
    parser.add_argument("--metrics-input", default="")
    parser.add_argument("--service-base-url", default=os.getenv("SERVICE_BASE_URL", ""))
    args = parser.parse_args()

    result = validate_production_model(
        min_accuracy=args.min_accuracy,
        max_latency_p95=args.max_latency_p95,
        max_error_rate=args.max_error_rate,
        registry_dir=args.registry_dir,
        deployment_manifest_path=args.deployment_manifest_path,
        tracker_path=args.tracker_path,
        metrics_input=args.metrics_input,
        service_base_url=args.service_base_url,
    )

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["validation_passed"] else 1)


if __name__ == "__main__":
    main()
