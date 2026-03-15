"""
Update deployment manifest with bundle-backed deployment metadata.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import write_deployment_manifest


def update_deployment_manifest(
    run_id: str,
    strategy: str,
    manifest_path: str,
    registry_dir: str = "experiments/registry",
    environment: str = "production",
    deployed_by: str = "automation",
    deployment_url: str = "",
    canary_percent: float | None = None,
    status: str = "active",
    service_image: str = "",
) -> dict:
    manifest = write_deployment_manifest(
        run_id=run_id,
        strategy=strategy,
        manifest_path=manifest_path,
        registry_dir=registry_dir,
        environment=environment,
        deployed_by=deployed_by,
        deployment_url=deployment_url,
        canary_percent=canary_percent,
        status=status,
        service_image=service_image,
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Update deployment manifest")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--strategy", required=True, help="Deployment strategy, e.g. canary or blue_green")
    parser.add_argument("--manifest-path", required=True, help="Path to deployment manifest JSON")
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--deployed-by", default="automation")
    parser.add_argument("--deployment-url", default="")
    parser.add_argument("--canary-percent", type=float, default=None)
    parser.add_argument("--status", default="active")
    parser.add_argument("--service-image", default="")
    args = parser.parse_args()

    update_deployment_manifest(
        run_id=args.run_id,
        strategy=args.strategy,
        manifest_path=args.manifest_path,
        registry_dir=args.registry_dir,
        environment=args.environment,
        deployed_by=args.deployed_by,
        deployment_url=args.deployment_url,
        canary_percent=args.canary_percent,
        status=args.status,
        service_image=args.service_image,
    )


if __name__ == "__main__":
    main()
