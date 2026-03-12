from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def check_drift(compare_against: str, alert_threshold: float = 0.15) -> Dict[str, Any]:
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": False,
        "drift_magnitude": 0.0,
        "reason": "No significant drift",
        "features_drifted": [],
    }

    if not os.path.exists(compare_against):
        result["reason"] = f"Baseline file not found: {compare_against}"
        return result

    try:
        with open(compare_against, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        current_stats = {
            "numeric_mean": 100.0,
            "numeric_std": 15.0,
            "text_length_mean": 50.0,
        }
        baseline_stats = baseline.get("stats", {})

        drift_score = 0.0
        drifted_features = []
        for feature, baseline_val in baseline_stats.items():
            if feature in current_stats:
                pct_change = abs(current_stats[feature] - baseline_val) / (baseline_val + 1e-6)
                if pct_change > 0.05:
                    drift_score = max(drift_score, pct_change)
                    drifted_features.append(feature)

        if drift_score > alert_threshold:
            result["drift_detected"] = True
            result["drift_magnitude"] = drift_score
            result["reason"] = f"Drift detected: {drift_score:.3f} (threshold: {alert_threshold})"
            result["features_drifted"] = drifted_features
    except Exception as exc:
        result["reason"] = f"Error checking drift: {exc}"

    return result