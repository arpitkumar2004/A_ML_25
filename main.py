import argparse
import yaml
import os
import sys
from src.utils.logging_utils import get_logger
from src.utils.seed_everything import seed_everything
from src.utils.config_schema import validate_config_for_command, flatten_train_config
from src.registry.model_registry import promote_run, rollback_to_run, list_runs, get_active_production

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

    # ---- REGISTRY PROMOTION ----
    promote_parser = subparsers.add_parser("promote", help="Promote a run_id in the model registry")
    promote_parser.add_argument("--run_id", type=str, required=True, help="Run ID to promote")
    promote_parser.add_argument("--stage", type=str, default="production", choices=["staging", "canary", "production"])
    promote_parser.add_argument("--registry_dir", type=str, default="experiments/registry")

    rollback_parser = subparsers.add_parser("rollback", help="Rollback production to a previous run_id")
    rollback_parser.add_argument("--to_run_id", type=str, required=True, help="Run ID to set as production")
    rollback_parser.add_argument("--registry_dir", type=str, default="experiments/registry")

    list_parser = subparsers.add_parser("list-registry", help="List model registry entries")
    list_parser.add_argument("--registry_dir", type=str, default="experiments/registry")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Set seed and log start
    seed_everything(getattr(args, "seed", 42))
    logger.info(f"Starting pipeline: {args.command}")

    if args.command == "train":
        from src.pipelines.train_pipeline import run_train_pipeline
        config = load_config(args.config)
        validate_config_for_command("train", config)
        train_cfg = flatten_train_config(config)
        run_train_pipeline(train_cfg, model_name=args.model)

    elif args.command == "inference":
        from src.pipelines.inference_pipeline import run_inference_pipeline
        config = load_config(args.config)
        validate_config_for_command("inference", config)
        infer_cfg = config.get("inference", config)
        if args.model_dir:
            infer_cfg["models_dir"] = args.model_dir
        run_inference_pipeline(infer_cfg)

    elif args.command == "features":
        from src.pipelines.feature_pipeline import run_feature_pipeline
        config = load_config(args.config)
        validate_config_for_command("features", config)
        run_feature_pipeline(config.get("features", config))

    elif args.command == "ensemble":
        from src.pipelines.ensemble_pipeline import run_ensemble_pipeline
        config = load_config(args.config)
        run_ensemble_pipeline(config.get("ensemble", config))

    elif args.command == "quickrun":
        from src.experiments.exp_quick_run import main as run_quick_experiment
        run_quick_experiment()

    elif args.command == "promote":
        idx_path = promote_run(run_id=args.run_id, target_stage=args.stage, registry_dir=args.registry_dir)
        logger.info(f"Promoted run_id={args.run_id} to stage={args.stage}. index={idx_path}")

    elif args.command == "rollback":
        idx_path = rollback_to_run(run_id=args.to_run_id, registry_dir=args.registry_dir)
        logger.info(f"Rollback complete. run_id={args.to_run_id} is now production. index={idx_path}")

    elif args.command == "list-registry":
        runs = list_runs(registry_dir=args.registry_dir)
        active = get_active_production(registry_dir=args.registry_dir)
        logger.info(f"active_production_run_id={active}")
        for r in runs:
            logger.info(str(r))

    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
