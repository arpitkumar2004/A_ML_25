from __future__ import annotations

from typing import Any, Dict

import yaml


class SLOValidator:
    def __init__(self, config_path: str = "configs/validation/slos.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        self._slos: Dict[str, Dict[str, float]] = payload.get("validation", {})

    def get_stage_thresholds(self, stage: str) -> Dict[str, float]:
        if stage not in self._slos:
            raise ValueError(f"Unknown stage '{stage}'. Available stages: {sorted(self._slos.keys())}")
        return self._slos[stage]

    def validate_metrics(self, metrics: Dict[str, Any], stage: str) -> Dict[str, Any]:
        thresholds = self.get_stage_thresholds(stage)
        checks: Dict[str, Dict[str, Any]] = {}
        failures = []

        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            passed = accuracy >= thresholds["min_accuracy"]
            checks["accuracy"] = {
                "actual": accuracy,
                "threshold": thresholds["min_accuracy"],
                "passed": passed,
            }
            if not passed:
                failures.append(f"Accuracy {accuracy:.3f} < {thresholds['min_accuracy']}")

        smape = metrics.get("smape")
        if smape is not None:
            passed = smape <= thresholds["max_smape"]
            checks["smape"] = {
                "actual": smape,
                "threshold": thresholds["max_smape"],
                "passed": passed,
            }
            if not passed:
                failures.append(f"SMAPE {smape:.3f} > {thresholds['max_smape']}")

        latency = metrics.get("latency_p95")
        if latency is not None:
            passed = latency <= thresholds["max_latency_p95"]
            checks["latency_p95"] = {
                "actual": latency,
                "threshold": thresholds["max_latency_p95"],
                "passed": passed,
            }
            if not passed:
                failures.append(f"Latency {latency:.3f}s > {thresholds['max_latency_p95']}s")

        error_rate = metrics.get("error_rate")
        if error_rate is not None:
            passed = error_rate <= thresholds["max_error_rate"]
            checks["error_rate"] = {
                "actual": error_rate,
                "threshold": thresholds["max_error_rate"],
                "passed": passed,
            }
            if not passed:
                failures.append(f"Error rate {error_rate:.3f} > {thresholds['max_error_rate']}")

        return {
            "stage": stage,
            "thresholds": thresholds,
            "checks": checks,
            "passed": len(failures) == 0,
            "failures": failures,
        }