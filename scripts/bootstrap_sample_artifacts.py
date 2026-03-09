import argparse
import os
import pandas as pd

from src.pipelines.train_pipeline import run_train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap core artifacts using sampled training data")
    parser.add_argument("--input", default="data/raw/train.csv", help="Path to raw training CSV")
    parser.add_argument("--prepared", default="data/interim/train_bootstrap_prepared.csv", help="Prepared mapped CSV path")
    parser.add_argument("--sample-frac", type=float, default=0.02, help="Fraction of training rows to use")
    parser.add_argument("--model", default="Linear", help="Model to train (Linear, RF, LGBM, XGB, Cat)")
    parser.add_argument("--n-splits", type=int, default=2, help="CV folds for bootstrap run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.sample_frac <= 0 or args.sample_frac > 1:
        raise ValueError("--sample-frac must be in (0, 1]")

    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("experiments/oof", exist_ok=True)
    os.makedirs("experiments/reports", exist_ok=True)
    os.makedirs("experiments/registry", exist_ok=True)

    raw_df = pd.read_csv(args.input)

    required_cols = {"sample_id", "catalog_content", "price"}
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw input is missing required columns: {missing}")

    mapped_df = pd.DataFrame(
        {
            "unique_identifier": raw_df["sample_id"],
            "Description": raw_df["catalog_content"].fillna("").astype(str),
            "Price": raw_df["price"].astype(float),
        }
    )
    mapped_df.to_csv(args.prepared, index=False)

    train_cfg = {
        "data_path": args.prepared,
        "sample_frac": args.sample_frac,
        "random_state": args.seed,
        "seed": args.seed,
        "text_col": "Description",
        "image_col": "__missing_image_col__",
        "target_col": "Price",
        "text_cfg": {
            "method": "tfidf",
            "cache_path": "data/processed/text_embeddings.joblib",
            "vectorizer_path": "data/processed/tfidf_vectorizer.joblib",
            "tfidf_max_features": 1024,
            "tfidf_ngram_range": (1, 2),
        },
        "image_cfg": {
            "cache_path": "data/processed/image_embeddings.joblib",
        },
        "numeric_cfg": {
            "scaler_path": "data/processed/numeric_scaler.joblib",
        },
        "selector_cfg": {
            "enabled": True,
            "method": "f_regression",
            "k": 256,
            "min_features": 32,
            "save_path": "data/processed/feature_selector.joblib",
            "random_state": args.seed,
        },
        "feature_cache": "data/processed/features.joblib",
        "force_rebuild": True,
        "dim_method": "pca",
        "dim_components": 32,
        "dim_cache": "data/processed/dimred.joblib",
        "use_dim_cache": True,
        "models_out": "experiments/models",
        "oof_out": "experiments/oof",
        "reports_dir": "experiments/reports",
        "report_out": "experiments/reports/model_comparison.joblib",
        "report_csv": "experiments/reports/model_comparison.csv",
        "registry_dir": "experiments/registry",
        "run_stacker": False,
        "n_splits": args.n_splits,
        "stratify": False,
    }

    summary = run_train_pipeline(train_cfg, model_name=args.model)
    print("BOOTSTRAP_TRAIN_OK", summary)


if __name__ == "__main__":
    main()
