from ..data.dataset_loader import load_train_df
from ..features.text_embeddings import load_tfidf
from ..inference.predict import load_model, predict_with_model
from ..utils.logging_utils import get_logger
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
