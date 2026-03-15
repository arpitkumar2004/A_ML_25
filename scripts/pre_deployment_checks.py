import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import build_deployment_manifest
from src.utils.model_bundle import resolve_bundle_path, validate_bundle
from src.utils.registry_loader import RegistryLoader


def run_pre_deployment_checks(
    run_id: str,
    output: str,
    registry_dir: str = "experiments/registry",
    check_tests: bool = False,
    check_coverage: bool = False,
    check_security: bool = False,
) -> dict:
    result = {
        "run_id": run_id,
        "all_passed": False,
        "bundle_path": None,
        "health_checks": {
            "registry_entry": False,
            "bundle_contract": False,
            "deployment_manifest_preview": False,
            "tests": not check_tests,
            "coverage": not check_coverage,
            "security": not check_security,
        },
        "bundle_validation": {},
        "deployment_preview": {},
        "failures": [],
    }

    try:
        loader = RegistryLoader(registry_dir=registry_dir)
        run_entry = loader.get_run_by_id(run_id)
        if run_entry is None:
            result["failures"].append(f"Run {run_id} not found in registry")
        else:
            result["health_checks"]["registry_entry"] = True
            if run_entry.get("status") != "production":
                result["failures"].append(f"Run {run_id} is not in production stage")

            bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)
            result["bundle_path"] = bundle_path
            bundle_validation = validate_bundle(bundle_path)
            result["bundle_validation"] = bundle_validation
            result["health_checks"]["bundle_contract"] = bundle_validation["valid"]
            if not bundle_validation["valid"]:
                result["failures"].extend(bundle_validation["problems"])

            try:
                preview = build_deployment_manifest(
                    run_id=run_id,
                    strategy="production",
                    registry_dir=registry_dir,
                    status="preview",
                )
                result["deployment_preview"] = {
                    "bundle_config_sha256": preview.get("bundle_config_sha256"),
                    "previous_production_run_id": preview.get("previous_production_run_id"),
                    "service_image": preview.get("service_image"),
                }
                result["health_checks"]["deployment_manifest_preview"] = True
            except Exception as exc:
                result["failures"].append(f"deployment_preview_failed:{exc}")
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
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--check-tests", action="store_true")
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument("--check-security", action="store_true")
    args = parser.parse_args()

    result = run_pre_deployment_checks(
        run_id=args.run_id,
        output=args.output,
        registry_dir=args.registry_dir,
        check_tests=args.check_tests,
        check_coverage=args.check_coverage,
        check_security=args.check_security,
    )
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
