import argparse
import sys
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.drift_latency import build_monitoring_report, append_drift_snapshot, write_alert_payload


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("monitoring", cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build drift+latency monitoring dashboard and evaluate alert rules")
    parser.add_argument("--config", type=str, default="configs/monitoring/alerts.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    reference_df = pd.read_csv(cfg["reference_path"])
    current_df = pd.read_csv(cfg["current_path"])

    report = build_monitoring_report(
        reference_df=reference_df,
        current_df=current_df,
        latency_log_path=cfg.get("latency_log_path", "experiments/monitoring/latency_events.jsonl"),
        rules=cfg.get("rules", {}),
        out_json_path=cfg.get("out_json_path", "experiments/monitoring/dashboard.json"),
        out_html_path=cfg.get("out_html_path", "experiments/monitoring/dashboard.html"),
    )

    snapshot_path = cfg.get("snapshot_history_path", "experiments/monitoring/drift_snapshots.jsonl")
    alert_payload_path = cfg.get("alert_payload_path", "experiments/monitoring/alert_payload.json")
    append_drift_snapshot(snapshot_path, report)
    write_alert_payload(alert_payload_path, report)

    print("DASHBOARD_OK", {
        "alerts": len(report.get("alerts", [])),
        "json": cfg.get("out_json_path"),
        "html": cfg.get("out_html_path"),
        "snapshot_history": snapshot_path,
        "alert_payload": alert_payload_path,
    })


if __name__ == "__main__":
    main()
