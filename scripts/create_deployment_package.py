import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deployment_state import build_deployment_manifest
from src.utils.io import IO
from src.utils.model_bundle import resolve_bundle_path, validate_bundle
from src.utils.registry_loader import RegistryLoader


def _write_service_env(path: str, run_id: str, registry_dir: str, service_image: str, port: int) -> None:
    lines = [
        "# Generated deployment environment for the serving container",
        f"MODEL_RUN_ID={run_id}",
        "MODEL_BUNDLE_PATH=/opt/model-bundle",
        f"REGISTRY_DIR={registry_dir}",
        f"SERVICE_IMAGE={service_image}",
        f"PORT={port}",
        "HOST=0.0.0.0",
        "UVICORN_WORKERS=1",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_deployment_package(
    run_id: str,
    output_dir: str,
    registry_dir: str = "experiments/registry",
    service_image: str = "",
    port: int = 8000,
) -> Dict[str, Any]:
    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)
    bundle_validation = validate_bundle(bundle_path)
    if not bundle_validation["valid"]:
        raise ValueError(f"Bundle validation failed: {bundle_validation['problems']}")

    output_root = Path(output_dir)
    bundle_output_dir = output_root / "model-bundle"
    metadata_dir = output_root / "metadata"
    bundle_output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if bundle_output_dir.exists():
        shutil.rmtree(bundle_output_dir)
    shutil.copytree(bundle_path, bundle_output_dir)

    deployment_manifest = build_deployment_manifest(
        run_id=run_id,
        strategy="serving_release",
        registry_dir=registry_dir,
        environment="container_image",
        deployed_by="package_builder",
        status="packaged",
        service_image=service_image,
    )
    deployment_manifest["packaged_bundle_path"] = "/opt/model-bundle"
    deployment_manifest["packaging_output_dir"] = str(output_root)

    loader = RegistryLoader(registry_dir=registry_dir)
    registry_entry = loader.get_run_by_id(run_id)

    service_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "service_image": service_image,
        "port": port,
        "startup_command": "python -m uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}",
        "bundle_mount_path": "/opt/model-bundle",
        "health_endpoints": ["/healthz", "/readyz", "/metrics/json", "/service/info"],
        "expected_ready_reason_prefix": "ok:bundle=",
        "registry_dir": registry_dir,
    }

    IO.save_json(deployment_manifest, str(metadata_dir / "deployment_manifest.json"), indent=2)
    IO.save_json(registry_entry or {}, str(metadata_dir / "registry_entry.json"), indent=2)
    IO.save_json(service_contract, str(metadata_dir / "service_contract.json"), indent=2)
    _write_service_env(
        path=str(metadata_dir / "service.env"),
        run_id=run_id,
        registry_dir=registry_dir,
        service_image=service_image,
        port=port,
    )

    summary = {
        "run_id": run_id,
        "bundle_path": bundle_path,
        "output_dir": str(output_root),
        "bundle_output_dir": str(bundle_output_dir),
        "metadata_dir": str(metadata_dir),
        "deployment_manifest_path": str(metadata_dir / "deployment_manifest.json"),
        "service_contract_path": str(metadata_dir / "service_contract.json"),
        "service_env_path": str(metadata_dir / "service.env"),
        "service_image": service_image,
        "bundle_validation": bundle_validation,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deployable serving package for a promoted model bundle")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--output-dir", required=True, help="Directory to populate with model-bundle and metadata")
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--service-image", default="")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    create_deployment_package(
        run_id=args.run_id,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        service_image=args.service_image,
        port=args.port,
    )


if __name__ == "__main__":
    main()
