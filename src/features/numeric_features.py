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
        self.numeric_cols_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if df[c].dtype in [int, float] and c not in ["Price"]]
        X = df[numeric_cols].fillna(0.0).astype(float).values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.numeric_cols_ = list(numeric_cols)
        IO.save_pickle({"scaler": self.scaler, "numeric_cols": self.numeric_cols_}, self.scaler_path)
        logger.info(f"Fitted scaler and saved to {self.scaler_path}")
        return Xs, numeric_cols

    def transform(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
        if self.scaler is None:
            if self.scaler_path and os.path.exists(self.scaler_path):
                payload = IO.load_pickle(self.scaler_path)
                # Backward compatibility: older artifacts stored the scaler directly.
                if isinstance(payload, dict) and "scaler" in payload:
                    self.scaler = payload["scaler"]
                    self.numeric_cols_ = payload.get("numeric_cols")
                else:
                    self.scaler = payload
            else:
                raise RuntimeError("Scaler not fitted.")

        if numeric_cols is None:
            if self.numeric_cols_ is None:
                raise RuntimeError("numeric_cols not provided and scaler artifact lacks fitted numeric_cols.")
            numeric_cols = list(self.numeric_cols_)

        X = df[numeric_cols].fillna(0.0).astype(float).values
        Xs = self.scaler.transform(X)
        return Xs, list(numeric_cols)
