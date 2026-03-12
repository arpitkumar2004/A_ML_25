"""
Check for data/concept drift in production.
"""

import json
import os
from typing import Dict, Any
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.drift_detection import check_drift


def check_data_drift(
    compare_against: str,
    alert_threshold: float = 0.15
) -> Dict[str, Any]:
    return check_drift(compare_against=compare_against, alert_threshold=alert_threshold)


def main():
    parser = argparse.ArgumentParser(description="Check for data drift")
    parser.add_argument("--compare-against", required=True, help="Baseline stats file")
    parser.add_argument("--alert-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    result = check_data_drift(
        compare_against=args.compare_against,
        alert_threshold=args.alert_threshold
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
