# Ensemble pipeline combining multiple models

from src.data.dataset_loader import load_train_df
from src.data.parse_features import add_parsed_features
from src.features.build_features import build_features_for_train
from src.training.trainer import train_lgbm_cv, train_base_model
from src.inference.predict import load_model, predict_with_model
from src.utils.logging_utils import get_logger
import numpy as np
import joblib
import os

logger = get_logger("ensemble_pipeline")

def run_ensemble_pipeline(cfg):
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
    logger.info(f"Built features with shape: {X.shape}")
    
    logger.info("Training model (CV)")
    models, oof = train_lgbm_cv(X, y_trans, cfg)
    joblib.dump(oof, "experiments/oof_predictions/oof.npy")
    logger.info("Done training")
    # save models
    for i, model in enumerate(models):
        joblib.dump(model, f"experiments/models/model_fold{i}.pkl")
    joblib.dump(vect, "experiments/models/vectorizer.pkl")
    logger.info("Models and vectorizer saved")
    
    return models, oof

def run_ensemble_pipeline_full(cfg):
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

def run_ensemble_inference(cfg):
    logger.info("Running ensemble inference pipeline")
    df = load_train_df(cfg['data']['inference_path'])
    vect = load_tfidf(cfg['features'].get('vectorizer_path', 'experiments/models/vectorizer.pkl'))
    texts = df[cfg['data']['text_col']].astype(str).tolist()
    X_text = vect.transform(texts)
    logger.info(f"Built text features with shape: {X_text.shape}")
    
    # Load models
    model_paths = cfg['model'].get('model_paths', [f"experiments/models/model_fold{i}.pkl" for i in range(cfg['training'].get('cv_folds', 3))])
    models = [load_model(path) for path in model_paths]
    logger.info(f"Loaded {len(models)} trained models.")
    
    # Predict with each model and average
    all_preds = np.zeros((X_text.shape[0], len(models)))
    for i, model in enumerate(models):
        preds = predict_with_model(model, X_text)
        all_preds[:, i] = preds
        logger.info(f"Model {i} predictions done.")
    
    final_preds = np.mean(all_preds, axis=1)
    logger.info(f"Generated final predictions with shape: {final_preds.shape}")
    
    return final_preds

if __name__ == "__main__":
    import sys
    import yaml
    if len(sys.argv) != 2:
        print("Usage: python ensemble_pipeline.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    run_ensemble_pipeline_full(cfg)
