from src.data.dataset_loader import load_train_df
from src.features.text_embeddings import load_tfidf
from src.inference.predict import load_model, predict_with_model
from src.utils.logging_utils import get_logger
import numpy as np

logger = get_logger("inference_pipeline")

def run_inference_pipeline(cfg):
    logger.info("Running inference pipeline (demo)")
    df = load_train_df(cfg['data']['train_path'])
    vect = load_tfidf()
    texts = df[cfg['data']['text_col']].astype(str).tolist()
    X_text = vect.transform(texts)
    # load model
    model = load_model("experiments/models/lgbm_model.pkl")
    preds = predict_with_model(model, X_text)
    print("Sample preds:", preds[:5])
    
    return preds

def run_inference_pipeline_full(cfg):
    # Load data
    df = load_train_df(cfg['data']['inference_path'])
    logger.info(f"Loaded inference data with shape: {df.shape}")
    
    # Build features
    vect = load_tfidf(cfg['features'].get('vectorizer_path', 'experiments/models/vectorizer.pkl'))
    texts = df[cfg['data']['text_col']].astype(str).tolist()
    X_text = vect.transform(texts)
    logger.info(f"Built text features with shape: {X_text.shape}")
    
    # Load model
    model = load_model(cfg['model'].get('model_path', 'experiments/models/lgbm_model.pkl'))
    logger.info("Loaded trained model.")
    
    # Predict
    preds = predict_with_model(model, X_text)
    logger.info(f"Generated predictions with shape: {preds.shape}")
    
    return preds

