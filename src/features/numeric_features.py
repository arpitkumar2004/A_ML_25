# src/features/numeric_features.py
from typing import Optional, List
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory

logger = LoggerFactory.get("numeric_features")


class NumericBuilder:
    """
    Build numeric features from dataframe and scale them.
    Keeps a StandardScaler for transform/fit persistence.
    """
    def __init__(self, scaler_path: Optional[str] = "data/processed/numeric_scaler.joblib"):
        self.scaler_path = scaler_path
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if df[c].dtype in [int, float] and c not in ["Price"]]
        X = df[numeric_cols].fillna(0.0).astype(float).values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        IO.save_pickle(self.scaler, self.scaler_path)
        logger.info(f"Fitted scaler and saved to {self.scaler_path}")
        return Xs, numeric_cols

    def transform(self, df: pd.DataFrame, numeric_cols: List[str]):
        if self.scaler is None:
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = IO.load_pickle(self.scaler_path)
            else:
                raise RuntimeError("Scaler not fitted.")
        X = df[numeric_cols].fillna(0.0).astype(float).values
        Xs = self.scaler.transform(X)
        return Xs
