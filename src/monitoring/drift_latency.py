from typing import Dict, List, Any, Optional
import json
import os
import numpy as np
import pandas as pd

from ..utils.io import IO


def _psi_1d(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-9) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0

    edges = np.quantile(ref, q=np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), eps, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_numeric_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    psi_bins: int = 10,
) -> pd.DataFrame:
    if numeric_columns is None:
        ref_num = reference_df.select_dtypes(include=["number"])  # type: ignore[arg-type]
        cur_num = current_df.select_dtypes(include=["number"])  # type: ignore[arg-type]
        numeric_columns = [c for c in ref_num.columns if c in cur_num.columns]

    rows = []
    for col in numeric_columns:
        ref = reference_df[col].dropna().to_numpy(dtype=float)
        cur = current_df[col].dropna().to_numpy(dtype=float)
        if ref.size == 0 or cur.size == 0:
            continue
        rows.append(
            {
                "feature": col,
                "psi": _psi_1d(ref, cur, bins=psi_bins),
                "ref_mean": float(np.mean(ref)),
                "cur_mean": float(np.mean(cur)),
                "ref_std": float(np.std(ref)),
                "cur_std": float(np.std(cur)),
                "missing_ref": float(reference_df[col].isna().mean()),
                "missing_cur": float(current_df[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame(columns=["feature", "psi"])


def load_latency_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["run_id", "rows", "total_seconds", "predict_seconds", "seconds_per_row", "ts_utc"])
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def summarize_latency(latency_df: pd.DataFrame) -> Dict[str, float]:
    if latency_df.empty:
        return {
            "runs": 0,
            "p50_total_seconds": 0.0,
            "p95_total_seconds": 0.0,
            "p99_total_seconds": 0.0,
            "p95_seconds_per_row": 0.0,
        }

    total = latency_df["total_seconds"].astype(float).to_numpy()
    per_row = latency_df["seconds_per_row"].astype(float).to_numpy()
    return {
        "runs": int(len(latency_df)),
        "p50_total_seconds": float(np.percentile(total, 50)),
        "p95_total_seconds": float(np.percentile(total, 95)),
        "p99_total_seconds": float(np.percentile(total, 99)),
        "p95_seconds_per_row": float(np.percentile(per_row, 95)),
    }


def evaluate_alerts(drift_df: pd.DataFrame, latency_summary: Dict[str, float], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts = []

    psi_warn = float(rules.get("psi_warn", 0.2))
    psi_crit = float(rules.get("psi_critical", 0.3))
    latency_p95_limit = float(rules.get("latency_p95_total_seconds", 5.0))
    per_row_p95_limit = float(rules.get("latency_p95_seconds_per_row", 0.05))

    if not drift_df.empty:
        for _, row in drift_df.iterrows():
            psi = float(row.get("psi", 0.0))
            if psi >= psi_crit:
                alerts.append({"severity": "critical", "type": "drift", "feature": row["feature"], "psi": psi})
            elif psi >= psi_warn:
                alerts.append({"severity": "warning", "type": "drift", "feature": row["feature"], "psi": psi})

    p95_total = float(latency_summary.get("p95_total_seconds", 0.0))
    if p95_total > latency_p95_limit:
        alerts.append({
            "severity": "critical",
            "type": "latency",
            "metric": "p95_total_seconds",
            "value": p95_total,
            "threshold": latency_p95_limit,
        })

    p95_per_row = float(latency_summary.get("p95_seconds_per_row", 0.0))
    if p95_per_row > per_row_p95_limit:
        alerts.append({
            "severity": "warning",
            "type": "latency",
            "metric": "p95_seconds_per_row",
            "value": p95_per_row,
            "threshold": per_row_p95_limit,
        })

    return alerts


def build_monitoring_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    latency_log_path: str,
    rules: Dict[str, Any],
    out_json_path: str,
    out_html_path: Optional[str] = None,
) -> Dict[str, Any]:
    drift_df = compute_numeric_drift(reference_df, current_df)
    latency_df = load_latency_events(latency_log_path)
    latency_summary = summarize_latency(latency_df)
    alerts = evaluate_alerts(drift_df, latency_summary, rules)

    report = {
        "drift_top": drift_df.head(20).to_dict(orient="records"),
        "latency_summary": latency_summary,
        "alerts": alerts,
        "counts": {
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "latency_events": int(len(latency_df)),
        },
    }
    IO.save_json(report, out_json_path, indent=2)

    if out_html_path:
        html = [
            "<html><head><title>Monitoring Dashboard</title></head><body>",
            "<h1>Monitoring Dashboard</h1>",
            "<h2>Latency Summary</h2>",
            "<pre>" + json.dumps(latency_summary, indent=2) + "</pre>",
            "<h2>Alerts</h2>",
            "<pre>" + json.dumps(alerts, indent=2) + "</pre>",
            "<h2>Top Drift Features</h2>",
            drift_df.head(20).to_html(index=False) if not drift_df.empty else "<p>No drift rows.</p>",
            "</body></html>",
        ]
        IO.ensure_dir(out_html_path)
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

    return report
