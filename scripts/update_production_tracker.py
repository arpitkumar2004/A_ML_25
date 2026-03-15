import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import extract_slo_metrics, write_production_tracker
from src.utils.io import IO


def update_production_tracker(
    run_id: str,
    strategy: str,
    manifest_path: str,
    registry_dir: str = "experiments/registry",
    deployment_manifest_path: str = "experiments/registry/deployment_manifest.json",
    metrics_input: str = "",
    status: str = "active",
) -> dict:
    metrics = None
    if metrics_input:
        payload = IO.load_json(metrics_input)
        metrics = extract_slo_metrics(payload)
        if not metrics and isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            metrics = payload.get("metrics")

    tracker = write_production_tracker(
        run_id=run_id,
        strategy=strategy,
        manifest_path=manifest_path,
        registry_dir=registry_dir,
        deployment_manifest_path=deployment_manifest_path,
        metrics=metrics,
        status=status,
    )
    print(json.dumps(tracker, indent=2))
    return tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Update production deployment tracker")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--deployment-manifest-path", default="experiments/registry/deployment_manifest.json")
    parser.add_argument("--metrics-input", default="")
    parser.add_argument("--status", default="active")
    args = parser.parse_args()

    update_production_tracker(
        run_id=args.run_id,
        strategy=args.strategy,
        manifest_path=args.manifest_path,
        registry_dir=args.registry_dir,
        deployment_manifest_path=args.deployment_manifest_path,
        metrics_input=args.metrics_input,
        status=args.status,
    )


if __name__ == "__main__":
    main()
