import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if monitoring dashboard contains critical alerts")
    parser.add_argument("--dashboard", default="experiments/monitoring/dashboard.json")
    args = parser.parse_args()

    if not os.path.exists(args.dashboard):
        print(f"MONITORING_CHECK_SKIPPED dashboard_missing={args.dashboard}")
        return

    with open(args.dashboard, "r", encoding="utf-8") as f:
        report = json.load(f)

    alerts = report.get("alerts", [])
    critical = [a for a in alerts if str(a.get("severity", "")).lower() == "critical"]

    if critical:
        print("CRITICAL_ALERTS_FOUND", json.dumps(critical, indent=2))
        sys.exit(1)

    print(f"MONITORING_ALERT_CHECK_OK alerts={len(alerts)} critical=0")


if __name__ == "__main__":
    main()
