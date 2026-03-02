import argparse
import yaml
import os
import sys
from src.utils.logging_utils import get_logger
from src.utils.seed_everything import seed_everything

logger = get_logger(__name__)


def load_config(config_path: str):
    """Load YAML configuration and flatten nested includes if any."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="🏗️ Price Prediction Competition Runner")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands for pipeline control")

    # ---- TRAIN ----
    train_parser = subparsers.add_parser("train", help="Run model training pipeline")
    train_parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    train_parser.add_argument("--model", type=str, required=False, default=None, help="Optional model name filter (e.g. lgbm, xgb, rf, linear)")
    train_parser.add_argument("--seed", type=int, default=42)

    # ---- INFERENCE ----
    infer_parser = subparsers.add_parser("inference", help="Run inference pipeline")
    infer_parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    infer_parser.add_argument("--model_dir", type=str, required=False, default=None, help="Optional directory containing trained models")

    # ---- FEATURE PIPELINE ----
    feat_parser = subparsers.add_parser("features", help="Run feature building pipeline")
    feat_parser.add_argument("--config", type=str, required=True, help="Path to feature config YAML")

    # ---- ENSEMBLE ----
    ens_parser = subparsers.add_parser("ensemble", help="Run stacking/blending ensemble")
    ens_parser.add_argument("--config", type=str, required=True, help="Path to ensemble config YAML")

    # ---- QUICKRUN ----
    quick_parser = subparsers.add_parser("quickrun", help="Run full demo: train all + UMAP + stacking + compare")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Set seed and log start
    seed_everything(getattr(args, "seed", 42))
    logger.info(f"🚀 Starting pipeline: {args.command}")

    if args.command == "train":
        from src.pipelines.train_pipeline import run_train_pipeline
        config = load_config(args.config)
        train_cfg = config.get("training", config)
        cv_cfg = config.get("cv", {})
        if "n_splits" not in train_cfg and isinstance(cv_cfg, dict):
            train_cfg["n_splits"] = cv_cfg.get("n_splits", 5)
            train_cfg["random_state"] = cv_cfg.get("random_state", 42)
            train_cfg["stratify"] = cv_cfg.get("stratify", False)
        run_train_pipeline(train_cfg, model_name=args.model)

    elif args.command == "inference":
        from src.pipelines.inference_pipeline import run_inference_pipeline
        config = load_config(args.config)
        infer_cfg = config.get("inference", config)
        if args.model_dir:
            infer_cfg["models_dir"] = args.model_dir
        run_inference_pipeline(infer_cfg)

    elif args.command == "features":
        from src.pipelines.feature_pipeline import run_feature_pipeline
        config = load_config(args.config)
        run_feature_pipeline(config.get("features", config))

    elif args.command == "ensemble":
        from src.pipelines.ensemble_pipeline import run_ensemble_pipeline
        config = load_config(args.config)
        run_ensemble_pipeline(config.get("ensemble", config))

    elif args.command == "quickrun":
        from src.experiments.exp_quick_run import main as run_quick_experiment
        run_quick_experiment()

    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
