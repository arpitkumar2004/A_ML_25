import argparse
import glob
import json
import math
import os

import pandas as pd


def _find_latest_manifest() -> str:
    manifests = sorted(glob.glob("experiments/registry/*_train_manifest.json"), key=os.path.getmtime)
    if not manifests:
        raise FileNotFoundError("No training manifests found in experiments/registry")
    return manifests[-1]


def validate_training(manifest_path: str, min_smape_improvement: float, max_train_time: int) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    outputs = manifest.get("outputs", {})
    timings = manifest.get("timings_seconds", {})
    report_csv = outputs.get("model_report")

    result = {
        "validation_passed": False,
        "manifest_path": manifest_path,
        "report_csv": report_csv,
        "failures": [],
        "warnings": [],
        "timings_seconds": timings,
        "best_smape": None,
    }

    if not report_csv or not os.path.exists(report_csv):
        result["failures"].append("Training report CSV not found")
        return result

    report_df = pd.read_csv(report_csv)
    if report_df.empty:
        result["failures"].append("Training report is empty")
        return result

    if "smape" not in report_df.columns:
        result["failures"].append("Training report is missing the smape column")
        return result

    best_smape = float(report_df["smape"].min())
    result["best_smape"] = best_smape

    if math.isnan(best_smape) or math.isinf(best_smape):
        result["failures"].append("Best SMAPE is not finite")

    total_time = float(timings.get("total", 0.0) or 0.0)
    if total_time and total_time > max_train_time:
        result["failures"].append(f"Training time {total_time:.2f}s exceeded {max_train_time}s")

    if min_smape_improvement > 0:
        result["warnings"].append(
            "min_smape_improvement was requested, but baseline comparison is not persisted in workflow state; validating absolute report quality only"
        )

    result["validation_passed"] = len(result["failures"]) == 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training outputs before registry registration")
    parser.add_argument("--manifest-path", default="", help="Training manifest JSON path")
    parser.add_argument("--min-smape-improvement", type=float, default=0.02)
    parser.add_argument("--max-train-time", type=int, default=3600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = args.manifest_path or _find_latest_manifest()
    result = validate_training(
        manifest_path=manifest_path,
        min_smape_improvement=args.min_smape_improvement,
        max_train_time=args.max_train_time,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["validation_passed"] else 1)


if __name__ == "__main__":
    main()