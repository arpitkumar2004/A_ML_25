# Making complete pipeline work with CV and non-CV training options
import numpy as np
import joblib
import os
from src.data.dataset_loader import DatasetLoader
from src.features.build_features import build_features_for_train
from src.training.trainer import train_model
from src.inference.predict import load_model, predict_with_model
from src.utils.logging_utils import get_logger
import yaml
import argparse
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline

logger = get_logger("main")

def main(cfg_path):
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.info("Loading training data")
    df = DatasetLoader(cfg).load_train_df(cfg['data']['train_path'])
    
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
    models, oof = train_model(X, y_trans, cfg)
    joblib.dump(oof, "experiments/oof_predictions/oof.npy")
    logger.info("Done training")
    # save models       
    
    for i, model in enumerate(models):
        joblib.dump(model, f"experiments/models/model_fold{i}.pkl")
    joblib.dump(vect, "experiments/models/vectorizer.pkl")
    logger.info("Models and vectorizer saved")
    
    return models, oof


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'])
    parser.add_argument('--config', '-c', default='configs/model/lgbm.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == 'train':
        run_train_pipeline(cfg)
    else:
        run_inference_pipeline(cfg)

if __name__ == "__main__":
    main()



