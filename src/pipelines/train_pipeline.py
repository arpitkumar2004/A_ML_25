"""High-level training pipeline that orchestrates everything."""
def main(cfg_path):
    print('Training pipeline reading', cfg_path)

from src.data.dataset_loader import load_train_df
from src.data.parse_features import add_parsed_features
from src.features.build_features import build_features_for_train
from src.training.trainer import train_lgbm_cv
import numpy as np
import joblib
from src.utils.logging_utils import get_logger

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

def run_train_pipeline_full(cfg):
    # Load data
    df = load_train_df(cfg['data']['train_path'])
    logger.info(f"Loaded training data with shape: {df.shape}")
    
    # Add parsed features if configured
    if cfg.get('parsed', {}).get('features'):
        df = add_parsed_features(df, cfg['data']['text_col'])
        logger.info("Added parsed features.")
    
    # Build features
    X, vect = build_features_for_train(df, cfg)
    y = df[cfg['data']['target_col']].values
    logger.info(f"Built features with shape: {X.shape}")
    
    # Train model with cross-validation
    models, oof = train_lgbm_cv(X, y, cfg)
    
    # Save models and vectorizer
    joblib.dump(models, cfg['training'].get('model_save_path', 'experiments/models/lgbm_models.pkl'))
    joblib.dump(vect, cfg['training'].get('vectorizer_save_path', 'experiments/models/tfidf_vectorizer.pkl'))
    logger.info("Saved trained models and vectorizer.")
    
    return models, oof

if __name__ == "__main__":
    import sys
    import yaml
    if len(sys.argv) != 2:
        print("Usage: python train_pipeline.py <config_path>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    run_train_pipeline(cfg)
    
    print('Training pipeline reading', cfg_path)
    

        