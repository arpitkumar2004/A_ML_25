import os
import pandas as pd

from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.inference.predict import PredictPipeline
from src.data.parse_features import Parser
from src.utils.io import IO


def main() -> None:
    base_dir = "experiments/smoke_check"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/data", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/oof", exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)

    scaler_path = f"{base_dir}/data/numeric_scaler.joblib"
    selector_path = f"{base_dir}/data/feature_selector.joblib"
    feature_cache = f"{base_dir}/data/features.joblib"
    dim_cache = f"{base_dir}/data/dimred.joblib"
    train_smoke_path = f"{base_dir}/data/train_smoke.csv"

    # Map repository raw schema to pipeline-required schema.
    raw_df = pd.read_csv("data/raw/train.csv")
    mapped_df = pd.DataFrame(
        {
            "unique_identifier": raw_df["sample_id"],
            "Description": raw_df["catalog_content"].fillna("").astype(str),
            "Price": raw_df["price"].astype(float),
        }
    )
    mapped_df.to_csv(train_smoke_path, index=False)

    train_cfg = {
        "data_path": train_smoke_path,
        "sample_frac": 0.01,
        "random_state": 42,
        "text_col": "Description",
        "image_col": "__missing_image_col__",
        "target_col": "Price",
        "text_cfg": {
            "method": "tfidf",
            "cache_path": f"{base_dir}/data/text_embeddings.joblib",
            "vectorizer_path": f"{base_dir}/data/tfidf_vectorizer.joblib",
            "tfidf_max_features": 512,
            "tfidf_ngram_range": (1, 2),
        },
        "image_cfg": {
            "cache_path": f"{base_dir}/data/image_embeddings.joblib",
        },
        "numeric_cfg": {
            "scaler_path": scaler_path,
        },
        "selector_cfg": {
            "enabled": True,
            "method": "f_regression",
            "k": 128,
            "min_features": 16,
            "save_path": selector_path,
            "random_state": 42,
        },
        "feature_cache": feature_cache,
        "force_rebuild": True,
        "dim_method": "pca",
        "dim_components": 16,
        "dim_cache": dim_cache,
        "use_dim_cache": True,
        "models_out": f"{base_dir}/models",
        "oof_out": f"{base_dir}/oof",
        "reports_dir": f"{base_dir}/reports",
        "report_out": f"{base_dir}/reports/model_comparison.joblib",
        "report_csv": f"{base_dir}/reports/model_comparison.csv",
        "registry_dir": f"{base_dir}/registry",
        "run_stacker": False,
        "n_splits": 2,
        "stratify": False,
    }

    summary = run_train_pipeline(train_cfg, model_name="Linear")
    print("TRAIN_SUMMARY", summary)

    for required in [scaler_path, selector_path, dim_cache]:
        if not os.path.exists(required):
            raise RuntimeError(f"Missing expected artifact: {required}")

    selector_payload = IO.load_pickle(selector_path)
    selected_count = len(selector_payload.get("selected_indices", []))
    if selected_count <= 0:
        raise RuntimeError("Feature selector artifact has no selected indices.")

    # Build inference features in transform-only mode to verify scaler + selector are consumed.
    df_raw = mapped_df.head(12).copy()
    df_parsed = Parser.add_parsed_features(df_raw.copy(), text_col="Description")

    pp = PredictPipeline(
        bundle_path=summary.get("bundle_path"),
        registry_dir=train_cfg["registry_dir"],
    )

    X_inf, meta_inf = pp._feature_builder.build(
        df_parsed,
        text_col="Description",
        image_col="__missing_image_col__",
        force_rebuild=True,
        mode="inference",
    )
    print("INFER_FEATURE_META", meta_inf)

    if not meta_inf.get("selection", {}).get("applied", False):
        raise RuntimeError("Selector was not applied during inference transform.")

    if X_inf.shape[1] != selected_count:
        raise RuntimeError(
            f"Selector mismatch: X_inf columns={X_inf.shape[1]} selected_count={selected_count}"
        )

    preds = pp.predict(
        df_raw,
        text_col="Description",
        image_col="__missing_image_col__",
        force_rebuild_features=True,
    )

    if len(preds) != len(df_raw):
        raise RuntimeError("Prediction length mismatch.")

    infer_input_path = f"{base_dir}/data/infer_input.csv"
    infer_output_path = f"{base_dir}/submissions/infer_output.csv"
    df_raw.to_csv(infer_input_path, index=False)
    infer_cfg = {
        "run_id": "smoke_infer",
        "input_path": infer_input_path,
        "output_path": infer_output_path,
        "text_col": "Description",
        "image_col": "__missing_image_col__",
        "id_col": "unique_identifier",
        "pred_col": "predicted_price",
        "text_cfg": train_cfg["text_cfg"],
        "image_cfg": train_cfg["image_cfg"],
        "numeric_cfg": train_cfg["numeric_cfg"],
        "selector_cfg": train_cfg["selector_cfg"],
        "bundle_path": summary.get("bundle_path"),
        "registry_dir": f"{base_dir}/registry",
        "latency_log_path": f"{base_dir}/monitoring/latency_events.jsonl",
    }
    _ = run_inference_pipeline(infer_cfg)

    print(
        "SMOKE_OK",
        {
            "n_preds": int(len(preds)),
            "selected_features": int(selected_count),
            "scaler_exists": os.path.exists(scaler_path),
            "selector_exists": os.path.exists(selector_path),
            "dim_exists": os.path.exists(dim_cache),
            "train_manifest_exists": os.path.exists(summary.get("manifest_path", "")),
            "bundle_manifest_exists": os.path.exists(os.path.join(summary.get("bundle_path", ""), "manifest.json")),
            "infer_manifest_exists": os.path.exists(f"{base_dir}/registry/smoke_infer_inference_manifest.json"),
            "latency_log_exists": os.path.exists(f"{base_dir}/monitoring/latency_events.jsonl"),
        },
    )


if __name__ == "__main__":
    main()
