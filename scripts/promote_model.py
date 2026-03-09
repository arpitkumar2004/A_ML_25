"""
Promote model to target stage in registry.
"""

import json
import os
from datetime import datetime
import argparse


def promote_model(
    run_id: str,
    target_stage: str,
    promoted_by: str = "automation",
    promotion_url: str = ""
) -> str:
    """
    Promote model to target stage.
    
    Args:
        run_id: MLflow run ID
        target_stage: Target stage (staging, canary, production)
        promoted_by: User/entity promoting the model
        promotion_url: Link to promotion workflow (GitHub Actions URL)
    
    Returns:
        Path to updated registry index
    """
    
    from src.registry.model_registry import promote_run, register_run
    
    registry_dir = "experiments/registry"
    os.makedirs(registry_dir, exist_ok=True)
    
    try:
        # First ensure run is registered if not already
        manifest_path = f"experiments/models/{run_id}/manifest.json"
        register_run(
            run_id=run_id,
            manifest_path=manifest_path,
            stage="promotion",
            registry_dir=registry_dir,
            tracking={
                "promoted_by": promoted_by,
                "promotion_time": datetime.utcnow().isoformat(),
                "promotion_url": promotion_url,
                "target_stage": target_stage
            }
        )
        
        # Promote to target stage
        promote_run(run_id, target_stage, registry_dir)
        
        print(f"✓ Model {run_id} promoted to {target_stage}")
        
        # Create promotion record
        promotion_record = {
            "run_id": run_id,
            "target_stage": target_stage,
            "promoted_by": promoted_by,
            "promotion_time": datetime.utcnow().isoformat(),
            "promotion_url": promotion_url
        }
        
        # Append to promotion history
        history_file = os.path.join(registry_dir, "promotion_history.jsonl")
        with open(history_file, 'a') as f:
            f.write(json.dumps(promotion_record) + '\n')
        
        return os.path.join(registry_dir, "index.json")
    
    except Exception as e:
        print(f"✗ Promotion failed: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Promote model to target stage")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--target-stage", required=True, help="Target stage")
    parser.add_argument("--promoted-by", default="automation", help="User promoting model")
    parser.add_argument("--promotion-url", default="", help="Promotion workflow URL")
    
    args = parser.parse_args()
    
    promote_model(
        run_id=args.run_id,
        target_stage=args.target_stage,
        promoted_by=args.promoted_by,
        promotion_url=args.promotion_url
    )


if __name__ == "__main__":
    main()
