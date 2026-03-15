# quick experiment usage example
import numpy as np
import pandas as pd
from src.data.dataset_loader import DatasetLoader
from src.features.build_features import FeatureBuilder
from src.models.lgb_model import LGBModel
from src.training.trainer import Trainer

# load and sample small fraction for quick test
loader = DatasetLoader(path="data/raw/train.csv")
df = loader.sample(frac=0.1)

# build features (use defaults)
fb = FeatureBuilder(
    text_cfg={"method":"tfidf", "model_name":"all-MiniLM-L6-v2", "cache_path":"data/processed/text_embeddings.joblib"},
    image_cfg={"cache_path":"data/processed/image_embeddings.joblib"},
    numeric_cfg={"scaler_path":"data/processed/numeric_scaler.joblib"},
    output_cache="data/processed/features.joblib"
)
y = df["Price"].values
X, meta = fb.build(df, text_col="Description", image_col="image_path", force_rebuild=False, y=y, mode="train")

trainer = Trainer(output_dir="experiments/models", n_splits=2, random_state=42, stratify=False)
models, oof, metrics = trainer.run_cv(LGBModel, model_params={"params": {"n_estimators":100, "learning_rate":0.1}}, X=X, y=y)
print(metrics)
