"""
Register trained model in the registry.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def register_model(
    run_id: str,
    stage: str = "staging",
    tags: str = "",
    bundle_path: Optional[str] = None,
) -> str:
    """
    Register model from MLflow run into local registry.
    
    Args:
        run_id: MLflow run ID
        stage: Initial stage (staging, canary, production)
        tags: Comma-separated tags for the model
    
    Returns:
        Path to registry index
    """
    
    from src.registry.model_registry import register_run
    from src.utils.registry_loader import RegistryLoader
    
    registry_dir = "experiments/registry"
    os.makedirs(registry_dir, exist_ok=True)

    loader = RegistryLoader(registry_dir=registry_dir)
    existing = None
    try:
        existing = loader.get_run_by_id(run_id)
    except Exception:
        existing = None

    manifest_path = ""
    resolved_bundle_path = bundle_path
    if existing is not None:
        manifest_path = str(existing.get("manifest_path") or "")
        resolved_bundle_path = resolved_bundle_path or existing.get("bundle_path")

    if not manifest_path:
        candidate_bundle_manifest = os.path.join("experiments", "runs", run_id, "bundle", "manifest.json")
        if os.path.exists(candidate_bundle_manifest):
            manifest_path = candidate_bundle_manifest
            resolved_bundle_path = resolved_bundle_path or os.path.dirname(candidate_bundle_manifest)
        else:
            manifest_path = f"experiments/models/{run_id}/manifest.json"
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            manifest = {
                "run_id": run_id,
                "stage": stage,
                "registered_at": datetime.utcnow().isoformat(),
                "artifacts": []
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
    
    # Parse tags
    tag_dict = {}
    if tags:
        for tag in tags.split(','):
            key, val = tag.strip().split('=', 1)
            tag_dict[key] = val
    
    tag_dict['registered_at'] = datetime.utcnow().isoformat()
    
    # Register in registry
    index_path = register_run(
        run_id=run_id,
        manifest_path=manifest_path,
        stage="train",
        registry_dir=registry_dir,
        status=stage,
        tracking=tag_dict,
        bundle_path=resolved_bundle_path,
    )
    
    print(f"✓ Model {run_id} registered in {stage} stage")
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Register trained model")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--stage", default="staging", help="Initial stage")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--bundle-path", default="", help="Optional immutable bundle path")
    
    args = parser.parse_args()
    
    register_model(
        run_id=args.run_id,
        stage=args.stage,
        tags=args.tags,
        bundle_path=args.bundle_path or None,
    )


if __name__ == "__main__":
    main()
