import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.column_aliases import normalize_to_train_schema


def prepare_training_data(input_path: str, output_path: str) -> str:
    raw_df = pd.read_csv(input_path)
    raw_df, _ = normalize_to_train_schema(raw_df)

    required_cols = {"sample_id", "catalog_content", "price"}
    missing = sorted(required_cols.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Raw training data is missing required columns: {missing}")

    prepared_df = pd.DataFrame(
        {
            "sample_id": raw_df["sample_id"],
            "catalog_content": raw_df["catalog_content"].fillna("").astype(str),
            "price": raw_df["price"].astype(float),
            "image_link": raw_df.get("image_link", "").fillna("").astype(str) if "image_link" in raw_df.columns else "",
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prepared_df.to_csv(output_path, index=False)
    print(f"Prepared training data written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train_prepared.csv from raw DVC-backed training data")
    parser.add_argument("--input", default="data/raw/train.csv", help="Input raw training CSV")
    parser.add_argument("--output", default="data/interim/train_prepared.csv", help="Output prepared training CSV")
    args = parser.parse_args()

    prepare_training_data(args.input, args.output)


if __name__ == "__main__":
    main()