"""
Update deployment manifest with latest deployment info.
"""

import json
import os
from datetime import datetime
import argparse


def update_deployment_manifest(
    run_id: str,
    stage: str,
    manifest_path: str
) -> str:
    """
    Update deployment manifest with latest info.
    
    Args:
        run_id: MLflow run ID
        stage: Deployment stage
        manifest_path: Path to deployment manifest JSON
    
    Returns:
        Updated manifest path
    """
    
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "deployed_at": datetime.utcnow().isoformat(),
        "status": "active",
        "deployment_strategy": "canary",
        "rollback_available": True
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Deployment manifest updated: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Update deployment manifest")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest-path", required=True)
    
    args = parser.parse_args()
    
    update_deployment_manifest(
        run_id=args.run_id,
        stage=args.stage,
        manifest_path=args.manifest_path
    )


if __name__ == "__main__":
    main()
