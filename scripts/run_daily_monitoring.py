import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily monitoring automation job")
    parser.add_argument("--config", default="configs/monitoring/alerts.yaml")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    cmd_build = [args.python, "scripts/build_monitoring_dashboard.py", "--config", args.config]
    cmd_check = [args.python, "scripts/check_monitoring_alerts.py", "--dashboard", "experiments/monitoring/dashboard.json"]

    print("RUN", " ".join(cmd_build))
    subprocess.run(cmd_build, check=True)

    print("RUN", " ".join(cmd_check))
    subprocess.run(cmd_check, check=True)

    print("DAILY_MONITORING_OK")


if __name__ == "__main__":
    main()
