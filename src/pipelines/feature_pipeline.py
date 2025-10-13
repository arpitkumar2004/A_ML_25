from src.data.dataset_loader import load_train_df
from src.data.parse_features import add_parsed_features
from src.features.build_features import build_features_for_train
from src.utils.logging_utils import get_logger

logger = get_logger("feature_pipeline")

def run_feature_pipeline(cfg):
    df = load_train_df(cfg['data']['train_path'])
    df = add_parsed_features(df, text_col=cfg['data']['text_col'])
    X, vect = build_features_for_train(df, cfg)
    logger.info("Saved features (in-memory return)")
    return X, df

def run_feature_pipeline_full(cfg):
    # Load data
    df = load_train_df(cfg['data']['train_path'])
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Add parsed features if configured
    if cfg.get('parsed', {}).get('features'):
        df = add_parsed_features(df, cfg['data']['text_col'])
        logger.info("Added parsed features.")
    
    # Build features
    X, vect = build_features_for_train(df, cfg)
    logger.info(f"Built features with shape: {X.shape}")
    
    return X, df, vect