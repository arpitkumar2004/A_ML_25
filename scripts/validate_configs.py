import argparse

from main import load_config
from src.utils.config_schema import validate_config_for_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline configuration files")
    parser.add_argument("--train", default="configs/training/final_train.yaml")
    parser.add_argument("--inference", default="configs/inference/inference.yaml")
    parser.add_argument("--features", default="configs/features/all_features.yaml")
    args = parser.parse_args()

    train_cfg = load_config(args.train)
    infer_cfg = load_config(args.inference)
    feat_cfg = load_config(args.features)

    validate_config_for_command("train", train_cfg)
    validate_config_for_command("inference", infer_cfg)
    validate_config_for_command("features", feat_cfg)

    print("CONFIG_VALIDATION_OK")


if __name__ == "__main__":
    main()
