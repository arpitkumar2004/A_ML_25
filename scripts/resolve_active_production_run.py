import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.registry_loader import RegistryLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the active production run from the registry")
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    loader = RegistryLoader(registry_dir=args.registry_dir)
    run_id = loader.get_active_production_run_id()
    entry = loader.get_active_production_entry()
    result = {
        "run_id": run_id,
        "bundle_path": entry.get("bundle_path") if entry else None,
        "status": entry.get("status") if entry else None,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not run_id:
        raise SystemExit("No active production run is set in the registry")


if __name__ == "__main__":
    main()
