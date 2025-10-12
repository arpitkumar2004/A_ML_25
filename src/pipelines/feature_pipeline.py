from ..data.dataset_loader import load_train_df
from ..data.parse_features import add_parsed_features
from ..features.build_features import build_features_for_train
from ..utils.logging_utils import get_logger

logger = get_logger("feature_pipeline")

def run_feature_pipeline(cfg):
    df = load_train_df(cfg['data']['train_path'])
    df = add_parsed_features(df, text_col=cfg['data']['text_col'])
    X, vect = build_features_for_train(df, cfg)
    logger.info("Saved features (in-memory return)")
    return X, df
