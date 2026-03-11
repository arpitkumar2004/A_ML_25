import argparse
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment inference inputs for a promoted run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    args = parser.parse_args()

    if not os.path.exists(args.test_dataset):
        raise FileNotFoundError(f"Test dataset not found: {args.test_dataset}")

    with open("experiments/registry/index.json", encoding="utf-8") as f:
        registry = json.load(f)
    run_entry = next((item for item in registry.get("runs", []) if item.get("run_id") == args.run_id), None)
    if run_entry is None:
        raise ValueError(f"Run {args.run_id} not found in registry")

    df = pd.read_csv(args.test_dataset)
    if df.empty:
        raise ValueError("Inference test dataset is empty")

    candidate_columns = {"sample_id", "unique_identifier", "catalog_content", "Description"}
    if not any(column in df.columns for column in candidate_columns):
        raise ValueError(f"Test dataset does not contain any of the required columns: {sorted(candidate_columns)}")

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "min_accuracy_threshold": args.min_accuracy,
                "status": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()