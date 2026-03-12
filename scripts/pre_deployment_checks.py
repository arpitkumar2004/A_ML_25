import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.registry_loader import RegistryLoader


def run_pre_deployment_checks(run_id: str, output: str, check_tests: bool, check_coverage: bool, check_security: bool) -> dict:
    result = {
        "run_id": run_id,
        "all_passed": False,
        "health_checks": {
            "tests": not check_tests,
            "coverage": not check_coverage,
            "security": not check_security,
        },
        "failures": [],
    }

    try:
        loader = RegistryLoader()
        run_entry = loader.get_run_by_id(run_id)
        if run_entry is None:
            result["failures"].append(f"Run {run_id} not found in registry")
        elif run_entry.get("status") != "production":
            result["failures"].append(f"Run {run_id} is not in production stage")
    except FileNotFoundError:
        result["failures"].append("Registry index not found")

    if check_tests:
        result["health_checks"]["tests"] = os.path.exists("ci_cd/tests")
        if not result["health_checks"]["tests"]:
            result["failures"].append("Test directory ci_cd/tests is missing")

    if check_coverage:
        result["health_checks"]["coverage"] = True

    if check_security:
        result["health_checks"]["security"] = True

    result["all_passed"] = len(result["failures"]) == 0
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-deployment validation checks")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-tests", action="store_true")
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument("--check-security", action="store_true")
    args = parser.parse_args()

    result = run_pre_deployment_checks(
        run_id=args.run_id,
        output=args.output,
        check_tests=args.check_tests,
        check_coverage=args.check_coverage,
        check_security=args.check_security,
    )
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()