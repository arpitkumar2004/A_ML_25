"""
Rollback deployment to previous production model.
"""

import json
import os
from datetime import datetime
import argparse


def rollback_deployment(
    to_previous_production: bool = True,
    reason: str = "Manual rollback",
    registry_dir: str = "experiments/registry",
) -> str:
    """
    Rollback to previous production model.
    
    Args:
        to_previous_production: If True, revert to previously active production model
        reason: Reason for rollback
    
    Returns:
        Path to updated registry
    """
    
    from src.registry.model_registry import rollback_to_run, list_runs
    from src.utils.deployment_state import write_deployment_manifest, write_production_tracker
    
    try:
        runs = list_runs(registry_dir)
        
        # Find previous production (now archived)
        previous_prod = None
        for run in runs:
            if run.get("status") == "archived" and run.get("run_id"):
                previous_prod = run.get("run_id")
                break
        
        if not previous_prod:
            print("✗ No previous production model found to rollback to")
            raise ValueError("No archived production model available")
        
        # Rollback
        rollback_to_run(previous_prod, registry_dir)
        write_deployment_manifest(
            run_id=previous_prod,
            strategy="rollback",
            manifest_path=os.path.join(registry_dir, "deployment_manifest.json"),
            registry_dir=registry_dir,
            status="rolled_back",
        )
        write_production_tracker(
            run_id=previous_prod,
            strategy="rollback",
            manifest_path=os.path.join(registry_dir, "production_tracker.json"),
            registry_dir=registry_dir,
            deployment_manifest_path=os.path.join(registry_dir, "deployment_manifest.json"),
            status="rolled_back",
        )
        
        # Record rollback
        rollback_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "rolled_back_to": previous_prod,
            "reason": reason,
            "triggered_by": "automated_health_check"
        }
        
        history_file = os.path.join(registry_dir, "rollback_history.jsonl")
        with open(history_file, 'a') as f:
            f.write(json.dumps(rollback_record) + '\n')
        
        print(f"✓ Rolled back to {previous_prod}")
        print(f"  Reason: {reason}")
        
        return os.path.join(registry_dir, "index.json")
    
    except Exception as e:
        print(f"✗ Rollback failed: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Rollback deployment")
    parser.add_argument("--to-previous-production", action="store_true", default=True)
    parser.add_argument("--reason", default="Manual rollback", help="Rollback reason")
    parser.add_argument("--registry-dir", default="experiments/registry")
    
    args = parser.parse_args()
    
    rollback_deployment(
        to_previous_production=args.to_previous_production,
        reason=args.reason,
        registry_dir=args.registry_dir,
    )


if __name__ == "__main__":
    main()
