from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .deployment_state import extract_slo_metrics


def _join_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    normalized_path = path.lstrip("/")
    return urllib.parse.urljoin(normalized_base, normalized_path)


def fetch_json(base_url: str, path: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    url = _join_url(base_url, path)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return data


def probe_live_service(
    base_url: str,
    expected_run_id: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> Dict[str, Any]:
    ready = fetch_json(base_url, "/readyz", timeout_seconds=timeout_seconds)
    service_info = fetch_json(base_url, "/service/info", timeout_seconds=timeout_seconds)
    metrics_payload = fetch_json(base_url, "/metrics/json", timeout_seconds=timeout_seconds)
    metrics = extract_slo_metrics(metrics_payload)

    errors = []
    if not ready.get("ready"):
        errors.append(f"service_not_ready:{ready.get('reason')}")
    if expected_run_id and service_info.get("run_id") != expected_run_id:
        errors.append(f"service_run_id_mismatch:{service_info.get('run_id')}!={expected_run_id}")

    return {
        "reachable": True,
        "base_url": base_url,
        "ready": ready,
        "service_info": service_info,
        "metrics_payload": metrics_payload,
        "metrics": metrics,
        "valid": len(errors) == 0,
        "errors": errors,
    }


def try_probe_live_service(
    base_url: str,
    expected_run_id: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> Dict[str, Any]:
    try:
        return probe_live_service(base_url=base_url, expected_run_id=expected_run_id, timeout_seconds=timeout_seconds)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {
            "reachable": False,
            "base_url": base_url,
            "valid": False,
            "errors": [f"service_probe_failed:{exc}"],
            "metrics": {},
        }
