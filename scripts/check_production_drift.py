import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.drift_detection import check_drift


def main() -> None:
    parser = argparse.ArgumentParser(description="Check production data drift against baseline stats")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--alert-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = check_drift(args.baseline, args.alert_threshold)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0)


if __name__ == "__main__":
    main()