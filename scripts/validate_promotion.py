"""
Model validation and promotion checking.
Verifies model is production-ready before promotion.
"""

import csv
import json
import os
from typing import Dict, Any, Optional
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.slo_validator import SLOValidator
from src.utils.registry_loader import RegistryLoader
from src.utils.model_bundle import load_bundle_manifest, resolve_bundle_path


def validate_promotion(
    run_id: str,
    target_stage: str,
    min_accuracy: float | None = None,
    max_latency_p95: float | None = None
) -> Dict[str, Any]:
    """
    Validate model metrics are acceptable for promotion.

    Args:
        run_id: Canonical local run ID
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
        "warnings": [],
        "resolved_mlflow_run_id": None,
        "metrics_source": None,
    }

    try:
        entry = RegistryLoader().get_run_by_id(run_id)
    except Exception:
        entry = None

    try:
        result["metrics"] = _collect_metrics(run_id=run_id, registry_entry=entry, result=result)

        stage_name = "production" if target_stage == "production" else "canary" if target_stage == "canary" else "staging"
        validator = SLOValidator()
        slo_result = validator.validate_metrics(result["metrics"], stage=stage_name)
        result["failures"].extend(slo_result["failures"])

        accuracy = result["metrics"].get("accuracy")
        latency_p95 = result["metrics"].get("latency_p95")

        if min_accuracy is not None and accuracy is not None and accuracy < min_accuracy:
            result["failures"].append(f"Accuracy {accuracy:.3f} below threshold {min_accuracy}")
        if max_latency_p95 is not None and latency_p95 is not None and latency_p95 > max_latency_p95:
            result["failures"].append(f"P95 latency {latency_p95:.3f}s exceeds threshold {max_latency_p95}s")

        result["passed"] = len(result["failures"]) == 0
    except Exception as e:
        result["failures"].append(f"Validation error: {str(e)}")
    
    return result


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_best_local_report_metrics(report_path: str) -> Dict[str, float]:
    if not os.path.exists(report_path):
        return {}

    best_row: Optional[Dict[str, str]] = None
    best_smape: Optional[float] = None
    with open(report_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_smape = _coerce_float(row.get("smape"))
            if row_smape is None:
                continue
            if best_smape is None or row_smape < best_smape:
                best_smape = row_smape
                best_row = row

    if not best_row:
        return {}

    metrics: Dict[str, float] = {}
    smape = _coerce_float(best_row.get("smape"))
    if smape is not None:
        metrics["smape"] = smape
    accuracy = _coerce_float(best_row.get("accuracy"))
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    return metrics


def _collect_local_bundle_metrics(run_id: str, registry_entry: Optional[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, Any]:
    bundle_path = resolve_bundle_path(run_id=run_id, require_exists=True)
    manifest = load_bundle_manifest(bundle_path)
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    timings = manifest.get("timings_seconds", {}) if isinstance(manifest, dict) else {}
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}

    report_path = ""
    model_report = outputs.get("model_report")
    if model_report:
        report_path = str(model_report).replace("\\", os.sep)
    if not report_path:
        report_path = os.path.join(bundle_path, "reports", "model_comparison.csv")
    if not os.path.exists(report_path):
        report_path = os.path.join(bundle_path, "reports", "model_comparison.csv")

    metrics = _load_best_local_report_metrics(report_path=report_path)
    training_duration = _coerce_float(timings.get("total"))
    if training_duration is not None:
        metrics["training_duration"] = training_duration
    dataset_size = config.get("sample_frac")
    if dataset_size is not None:
        metrics["dataset_size"] = dataset_size

    if "accuracy" not in metrics:
        result["warnings"].append("Accuracy metric not available in local bundle report; promotion will rely on SMAPE-based validation")
    if "smape" not in metrics:
        result["warnings"].append("SMAPE metric not available in local bundle report")
    if "latency_p95" not in metrics:
        result["warnings"].append("Latency metric not available in local training bundle; skipping latency gate")
    if "error_rate" not in metrics:
        result["warnings"].append("Error rate metric not available in local training bundle; skipping error-rate gate")

    return metrics


def _collect_metrics(run_id: str, registry_entry: Optional[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, Any]:
    mlflow_run_id = run_id
    tracking = registry_entry.get("tracking", {}) if isinstance(registry_entry, dict) else {}
    mlflow_meta = tracking.get("mlflow", {}) if isinstance(tracking, dict) else {}
    if mlflow_meta:
        mlflow_run_id = str(mlflow_meta.get("mlflow_run_id") or run_id)
    result["resolved_mlflow_run_id"] = mlflow_run_id

    mlflow_enabled = bool(mlflow_meta.get("enabled")) and bool(mlflow_meta.get("mlflow_run_id"))
    if mlflow_enabled:
        try:
            import mlflow

            run = mlflow.get_run(mlflow_run_id)
            metrics = run.data.metrics or {}
            params = run.data.params or {}
            collected: Dict[str, Any] = {}

            accuracy = metrics.get("accuracy")
            if accuracy is not None:
                collected["accuracy"] = accuracy
            else:
                result["warnings"].append("Accuracy metric not logged in MLflow; promotion will rely on SMAPE-based validation")

            best_smape = metrics.get("best.smape", metrics.get("smape"))
            if best_smape is not None:
                collected["smape"] = best_smape
            else:
                result["warnings"].append("SMAPE metric not logged in MLflow")

            latency_p95 = metrics.get("latency_p95")
            if latency_p95 is not None:
                collected["latency_p95"] = latency_p95
            else:
                result["warnings"].append("Latency metric not logged in MLflow; skipping latency gate")

            error_rate = metrics.get("error_rate")
            if error_rate is not None:
                if error_rate > 0.05:
                    result["failures"].append(f"Error rate {error_rate:.3f} exceeds 5%")
                else:
                    collected["error_rate"] = error_rate
            else:
                result["warnings"].append("Error rate metric not logged in MLflow; skipping error-rate gate")

            training_duration = metrics.get("training_duration_seconds")
            if training_duration is not None:
                collected["training_duration"] = training_duration
            dataset_size = params.get("dataset_size")
            if dataset_size is not None:
                collected["dataset_size"] = dataset_size

            result["metrics_source"] = "mlflow"
            return collected
        except Exception as exc:
            result["warnings"].append(
                f"MLflow metrics unavailable for run {mlflow_run_id}; falling back to local bundle metrics ({exc})"
            )

    result["metrics_source"] = "local_bundle"
    return _collect_local_bundle_metrics(run_id=run_id, registry_entry=registry_entry, result=result)


def main():
    parser = argparse.ArgumentParser(description="Validate model for promotion")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--target-stage", required=True, help="Target stage")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--max-latency-p95", type=float, default=None)
    
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
