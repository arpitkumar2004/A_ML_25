"""
Register trained model in the registry.
"""

import json
import os
from datetime import datetime
import argparse


def register_model(
    run_id: str,
    stage: str = "staging",
    tags: str = ""
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
    
    registry_dir = "experiments/registry"
    os.makedirs(registry_dir, exist_ok=True)
    
    manifest_path = f"experiments/models/{run_id}/manifest.json"
    
    # Create manifest if doesn't exist
    if not os.path.exists(manifest_path):
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
        stage="training",
        registry_dir=registry_dir,
        status=stage,
        tracking=tag_dict
    )
    
    print(f"✓ Model {run_id} registered in {stage} stage")
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Register trained model")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--stage", default="staging", help="Initial stage")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    
    args = parser.parse_args()
    
    register_model(
        run_id=args.run_id,
        stage=args.stage,
        tags=args.tags
    )


if __name__ == "__main__":
    main()
