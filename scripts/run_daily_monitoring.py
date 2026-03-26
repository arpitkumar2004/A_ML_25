import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily monitoring automation job")
    parser.add_argument("--config", default="configs/monitoring/alerts.yaml")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    cmd_build = [args.python, str(ROOT / "scripts" / "build_monitoring_dashboard.py"), "--config", args.config]
    cmd_check = [
        args.python,
        str(ROOT / "scripts" / "check_monitoring_alerts.py"),
        "--dashboard",
        "experiments/monitoring/dashboard.json",
    ]

    print("RUN", " ".join(cmd_build))
    subprocess.run(cmd_build, check=True, cwd=ROOT)

    print("RUN", " ".join(cmd_check))
    subprocess.run(cmd_check, check=True, cwd=ROOT)

    print("DAILY_MONITORING_OK")


if __name__ == "__main__":
    main()
