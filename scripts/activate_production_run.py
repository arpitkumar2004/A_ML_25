import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry.model_registry import activate_run
from src.utils.registry_loader import RegistryLoader


def activate_production_run(run_id: str, registry_dir: str = "experiments/registry") -> dict:
    index_path = activate_run(run_id=run_id, registry_dir=registry_dir)
    entry = RegistryLoader(registry_dir=registry_dir).get_run_by_id(run_id)
    result = {
        "run_id": run_id,
        "registry_dir": registry_dir,
        "index_path": index_path,
        "bundle_path": entry.get("bundle_path") if entry else None,
        "status": entry.get("status") if entry else None,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark a production-approved run as the active deployed production run")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--registry-dir", default="experiments/registry")
    args = parser.parse_args()

    activate_production_run(run_id=args.run_id, registry_dir=args.registry_dir)


if __name__ == "__main__":
    main()
