import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.parse_features import Parser
from src.features.build_features import FeatureBuilder
from src.features.dimensionality import DimReducer
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline


class StepTimer:
    def __init__(self):
        self.timings = {}

    def run(self, name, fn):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        self.timings[name] = elapsed
        return result


def ensure_dirs(base_dir: Path):
    for p in [
        base_dir / "data",
        base_dir / "cache",
        base_dir / "models",
        base_dir / "oof",
        base_dir / "reports",
        base_dir / "submissions",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def complexity_table(n_train: int, n_test: int):
    return {
        "load_and_sample": "O(n)",
        "parse_features": "O(n * L), L=text length",
        "build_text_tfidf": "O(nnz) approx, nnz=non-zero token counts",
        "build_image_embeddings": "O(n * d_img) if model available; O(n) with zero-fallback",
        "build_numeric": "O(n * p_num)",
        "dimensionality_reduction_pca": "O(n * d * k)",
        "cv_training_linear_kfold": "O(k_folds * (n_train * d^2 + d^3))",
        "inference_predict": "O(n_test * d)",
        "note": f"For this smoke check n_train={n_train}, n_test={n_test}; timing is for operational validation, not model optimization.",
    }


def main():
    parser = argparse.ArgumentParser(description="20/5 smoke test for train+inference pipeline with per-step timings")
    parser.add_argument("--train-path", default="data/raw/train.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="linear", help="Model filter passed to training pipeline")
    args = parser.parse_args()

    base_dir = Path("experiments") / "smoke_check"
    ensure_dirs(base_dir)

    timer = StepTimer()

    train_df = timer.run(
        "load_train_sample",
        lambda: pd.read_csv(args.train_path).sample(n=args.n_train, random_state=args.seed).reset_index(drop=True),
    )
    test_df = timer.run(
        "load_test_sample",
        lambda: pd.read_csv(args.test_path).sample(n=args.n_test, random_state=args.seed).reset_index(drop=True),
    )

    train_df = train_df.rename(
        columns={
            "sample_id": "unique_identifier",
            "catalog_content": "Description",
            "image_link": "image_path",
            "price": "Price",
        }
    )
    test_df = test_df.rename(
        columns={
            "sample_id": "unique_identifier",
            "catalog_content": "Description",
            "image_link": "image_path",
        }
    )

    mini_train_path = base_dir / "data" / f"train_{args.n_train}.csv"
    mini_test_path = base_dir / "data" / f"test_{args.n_test}.csv"
    train_df.to_csv(mini_train_path, index=False)
    test_df.to_csv(mini_test_path, index=False)

    timer.run(
        "sanity_parse_train",
        lambda: Parser.add_parsed_features(train_df.copy(), text_col="Description"),
    )

    # Optional isolated feature+dim timing block before full train pipeline
    feature_cfg = {
        "method": "tfidf",
        "cache_path": str(base_dir / "cache" / "text_embeddings.joblib"),
        "tfidf_max_features": 256,
        "tfidf_ngram_range": (1, 2),
    }
    image_cfg = {"cache_path": str(base_dir / "cache" / "image_embeddings.joblib")}
    numeric_cfg = {"scaler_path": str(base_dir / "cache" / "numeric_scaler.joblib")}

    parsed_df = Parser.add_parsed_features(train_df.copy(), text_col="Description")
    fb = FeatureBuilder(feature_cfg, image_cfg, numeric_cfg, output_cache=str(base_dir / "cache" / "features.joblib"))
    X_raw, _ = timer.run(
        "feature_build_only",
        lambda: fb.build(parsed_df, text_col="Description", image_col="image_path", force_rebuild=True),
    )

    if hasattr(X_raw, "toarray"):
        x_dense = X_raw.toarray()
    elif hasattr(X_raw, "todense"):
        x_dense = X_raw.todense()
    else:
        x_dense = X_raw

    k_components = max(2, min(10, min(len(parsed_df) - 1, x_dense.shape[1] - 1)))
    reducer = DimReducer(
        method="pca",
        n_components=k_components,
        cache_path=str(base_dir / "cache" / "dimred.joblib"),
    )

    timer.run("dim_reduction_only", lambda: reducer.fit_transform(x_dense, use_cache=False))

    train_cfg = {
        "data_path": str(mini_train_path),
        "sample_frac": 1.0,
        "target_col": "Price",
        "text_col": "Description",
        "image_col": "image_path",
        "id_col": "unique_identifier",
        "text_cfg": feature_cfg,
        "image_cfg": image_cfg,
        "numeric_cfg": numeric_cfg,
        "feature_cache": str(base_dir / "cache" / "features.joblib"),
        "dim_cache": str(base_dir / "cache" / "dimred.joblib"),
        "use_dim_cache": False,
        "dim_method": "pca",
        "dim_components": k_components,
        "n_splits": 2,
        "random_state": args.seed,
        "stratify": False,
        "run_stacker": False,
        "force_rebuild": True,
        "models_out": str(base_dir / "models"),
        "oof_out": str(base_dir / "oof"),
        "reports_dir": str(base_dir / "reports"),
        "report_out": str(base_dir / "reports" / "model_comparison.joblib"),
        "report_csv": str(base_dir / "reports" / "model_comparison.csv"),
        "dim_meta_out": str(base_dir / "reports" / "dim_meta.joblib"),
    }

    timer.run("train_pipeline", lambda: run_train_pipeline(train_cfg, model_name=args.model))

    infer_cfg = {
        "input_path": str(mini_test_path),
        "output_path": str(base_dir / "submissions" / f"smoke_submission_{args.n_test}.csv"),
        "text_col": "Description",
        "image_col": "image_path",
        "id_col": "unique_identifier",
        "pred_col": "predicted_price",
        "feature_cache": str(base_dir / "cache" / "features_test.joblib"),
        "dim_cache": str(base_dir / "cache" / "dimred.joblib"),
        "models_dir": str(base_dir / "models"),
        "oof_meta_path": str(base_dir / "oof" / "model_names.joblib"),
        "stacker_path": str(base_dir / "models" / "stacker.joblib"),
        "target_transform": None,
        "round": True,
        "min_value": 0.0,
        "force_rebuild_features": True,
        "text_cfg": {
            "method": "tfidf",
            "cache_path": str(base_dir / "cache" / "text_embeddings_test.joblib"),
            "tfidf_max_features": 256,
            "tfidf_ngram_range": (1, 2),
        },
        "image_cfg": {"cache_path": str(base_dir / "cache" / "image_embeddings_test.joblib")},
        "numeric_cfg": {"scaler_path": str(base_dir / "cache" / "numeric_scaler.joblib")},
    }

    out_path = timer.run("inference_pipeline", lambda: run_inference_pipeline(infer_cfg))

    pred_df = pd.read_csv(out_path)
    status = {
        "output_exists": Path(out_path).exists(),
        "prediction_rows": int(len(pred_df)),
        "expected_rows": int(args.n_test),
        "prediction_columns": list(pred_df.columns),
        "status": "PASS" if len(pred_df) == args.n_test else "FAIL",
    }

    result = {
        "smoke_config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "model": args.model,
            "seed": args.seed,
        },
        "timings_seconds": {k: round(v, 4) for k, v in timer.timings.items()},
        "complexity": complexity_table(args.n_train, args.n_test),
        "validation": status,
    }

    report_json = base_dir / "reports" / "smoke_pipeline_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== PIPELINE SMOKE CHECK RESULT ===")
    print(json.dumps(result, indent=2))
    print(f"\nReport saved to: {report_json}")


if __name__ == "__main__":
    main()
