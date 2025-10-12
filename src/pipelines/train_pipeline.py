"""High-level training pipeline that orchestrates everything."""
def main(cfg_path):
    print('Training pipeline reading', cfg_path)

from ..data.dataset_loader import load_train_df
from ..data.parse_features import add_parsed_features
from ..features.build_features import build_features_for_train
from ..training.trainer import train_lgbm_cv
import numpy as np
import joblib
from ..utils.logging_utils import get_logger

logger = get_logger("train_pipeline")

def run_train_pipeline(cfg):
    logger.info("Loading train data")
    df = load_train_df(cfg['data']['train_path'])
    df = add_parsed_features(df, text_col=cfg['data']['text_col'])
    # target transform
    target_col = cfg['data']['target_col']
    y = df[target_col].values.astype(float)
    if cfg['training'].get('target_transform') == 'log1p':
        y_trans = np.log1p(y)
    else:
        y_trans = y
    logger.info("Building features")
    X, vect = build_features_for_train(df, cfg)
    logger.info("Training model (CV)")
    models, oof = train_lgbm_cv(X, y_trans, cfg)
    joblib.dump(oof, "experiments/oof_predictions/oof.npy")
    logger.info("Done training")
    # save models
    for i, model in enumerate(models):
        joblib.dump(model, f"experiments/models/model_fold{i}.pkl")
    joblib.dump(vect, "experiments/models/vectorizer.pkl")
    logger.info("Models and vectorizer saved")
        