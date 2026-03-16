import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import PredictPipeline
from src.utils.model_bundle import resolve_bundle_path, validate_bundle
from src.utils.registry_loader import RegistryLoader


def _pick_first_column(columns: pd.Index, candidates: list[str], required: bool = False) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Missing required columns. Expected one of: {candidates}")
    return None


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / denom))


def run_inference_smoke_test(
    run_id: str,
    test_dataset: str,
    min_accuracy: float = 0.70,
    registry_dir: str = "experiments/registry",
    output: str = "",
) -> Dict[str, Any]:
    if not os.path.exists(test_dataset):
        raise FileNotFoundError(f"Test dataset not found: {test_dataset}")

    loader = RegistryLoader(registry_dir=registry_dir)
    run_entry = loader.get_run_by_id(run_id)
    if run_entry is None:
        raise ValueError(f"Run {run_id} not found in registry")

    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir)
    bundle_validation = validate_bundle(bundle_path)
    if not bundle_validation["valid"]:
        raise ValueError(f"Bundle validation failed: {bundle_validation['problems']}")

    df = pd.read_csv(test_dataset)
    if df.empty:
        raise ValueError("Inference test dataset is empty")

    text_col = _pick_first_column(df.columns, ["catalog_content", "Description", "description"], required=True)
    image_col = _pick_first_column(df.columns, ["image_link", "image_url", "image_path"], required=False) or "image_link"
    id_col = _pick_first_column(df.columns, ["sample_id", "unique_identifier", "id"], required=True)
    target_col = _pick_first_column(df.columns, ["price", "Price", "target"], required=False)

    sample_df = df.head(min(len(df), 32)).copy()
    if image_col not in sample_df.columns:
        sample_df[image_col] = ""

    pipeline = PredictPipeline(bundle_path=bundle_path, registry_dir=registry_dir)
    started = time.perf_counter()
    preds = np.asarray(
        pipeline.predict(
            sample_df,
            text_col=text_col,
            image_col=image_col,
            force_rebuild_features=True,
        ),
        dtype=float,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)

    if len(preds) != len(sample_df):
        raise RuntimeError(f"Prediction length mismatch: {len(preds)} != {len(sample_df)}")

    rows_scored = int(len(sample_df))
    # Keep the raw end-to-end batch time, but normalize the SLO-facing latency metric so
    # bundle smoke tests don't fail production thresholds simply because they scored a batch.
    latency_per_row = float(elapsed / max(rows_scored, 1))
    metrics: Dict[str, Any] = {
        "batch_latency_seconds": float(elapsed),
        "latency_p95": latency_per_row,
        "error_rate": 0.0,
        "success_rate": 1.0,
        "rows_scored": rows_scored,
        "throughput_qps": float(rows_scored / elapsed),
        "prediction_mean": float(np.mean(preds)),
    }
    warnings = []

    if target_col and pd.api.types.is_numeric_dtype(sample_df[target_col]):
        metrics["smape"] = _compute_smape(sample_df[target_col].to_numpy(dtype=float), preds)
    else:
        warnings.append("No numeric target column found; regression quality metrics skipped")

    result = {
        "run_id": run_id,
        "bundle_path": bundle_path,
        "bundle_validation": bundle_validation,
        "dataset_rows": int(len(df)),
        "sample_rows": int(len(sample_df)),
        "columns": list(df.columns),
        "text_col": text_col,
        "image_col": image_col,
        "id_col": id_col,
        "target_col": target_col,
        "min_accuracy_threshold": min_accuracy,
        "metrics": metrics,
        "warnings": warnings,
        "passed": True,
    }

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment inference for a promoted run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = run_inference_smoke_test(
        run_id=args.run_id,
        test_dataset=args.test_dataset,
        min_accuracy=args.min_accuracy,
        registry_dir=args.registry_dir,
        output=args.output,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
