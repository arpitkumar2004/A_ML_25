import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.registry_loader import RegistryLoader


def resolve_registry_run(run_id: str, registry_dir: str = "experiments/registry") -> dict:
    loader = RegistryLoader(registry_dir=registry_dir)
    direct_entry = loader.get_run_by_id(run_id)
    resolved_run_id = loader.resolve_bundle_backed_run_id(run_id)
    bundle_backed_candidates = loader.list_bundle_backed_runs(limit=5)

    result = {
        "requested_run_id": run_id,
        "resolved_run_id": resolved_run_id,
        "requested_entry_exists": direct_entry is not None,
        "requested_entry_has_bundle": bool(direct_entry and direct_entry.get("bundle_path")),
        "requested_entry_stage": direct_entry.get("stage") if direct_entry else None,
        "requested_entry_status": direct_entry.get("status") if direct_entry else None,
        "requested_entry_bundle_path": direct_entry.get("bundle_path") if direct_entry else None,
        "recent_bundle_backed_runs": [
            {
                "run_id": entry.get("run_id"),
                "status": entry.get("status"),
                "stage": entry.get("stage"),
                "bundle_path": entry.get("bundle_path"),
            }
            for entry in bundle_backed_candidates
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a registry input to a canonical bundle-backed local run ID")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = resolve_registry_run(run_id=args.run_id, registry_dir=args.registry_dir)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))

    if not result["resolved_run_id"]:
        requested = result["requested_run_id"]
        suggestions = ", ".join([str(item["run_id"]) for item in result["recent_bundle_backed_runs"]]) or "none"
        raise SystemExit(
            f"Run '{requested}' is not a bundle-backed canonical local run id. "
            f"Use one of: {suggestions}"
        )


if __name__ == "__main__":
    main()
