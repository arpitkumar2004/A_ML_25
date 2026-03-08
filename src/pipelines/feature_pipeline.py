# src/pipelines/feature_pipeline.py
from typing import Dict, Any, Optional, Tuple
from ..data.dataset_loader import DatasetLoader
from ..data.parse_features import Parser
from ..features.build_features import FeatureBuilder
from ..utils.logging_utils import LoggerFactory
from ..utils.io import IO

logger = LoggerFactory.get("feature_pipeline")

def run_feature_pipeline(cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Build and return features and metadata. Useful for standalone feature generation.
    cfg should include: data_path, sample_frac, text_cfg, image_cfg, numeric_cfg, feature_cache
    """
    loader = DatasetLoader(cfg["data_path"])
    df = loader.sample(frac=cfg.get("sample_frac", 1.0), random_state=cfg.get("random_state", 42))
    df = Parser.add_parsed_features(df, text_col=cfg.get("text_col", "Description"))

    fb = FeatureBuilder(
        cfg.get("text_cfg", {}),
        cfg.get("image_cfg", {}),
        cfg.get("numeric_cfg", {}),
        selector_cfg=cfg.get("selector_cfg", {}),
        output_cache=cfg.get("feature_cache", "data/processed/features.joblib"),
    )
    X, meta = fb.build(
        df,
        text_col=cfg.get("text_col", "Description"),
        image_col=cfg.get("image_col", "image_path"),
        force_rebuild=cfg.get("force_rebuild", False),
        mode=cfg.get("feature_mode", "train"),
    )
    logger.info("Feature pipeline finished.")
    # persist a small metadata summary
    IO.save_pickle(meta, cfg.get("meta_out", "experiments/reports/feature_meta.joblib"))
    return X, meta
