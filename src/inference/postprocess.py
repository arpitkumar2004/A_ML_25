# src/inference/postprocess.py
import numpy as np
import pandas as pd
from typing import Optional


class Postprocessor:
    """
    Simple postprocessing utilities:
      - invert transformations (log1p)
      - rounding / clipping
    """

    @staticmethod
    def invert_log1p(preds: np.ndarray) -> np.ndarray:
        """Assumes preds were produced on log1p scale; returns original price scale."""
        return np.expm1(preds)

    @staticmethod
    def clip_min(preds: np.ndarray, min_value: float = 0.0) -> np.ndarray:
        p = preds.copy()
        p[p < min_value] = min_value
        return p

    @staticmethod
    def round_to_cents(preds: np.ndarray) -> np.ndarray:
        return np.round(preds, 2)

    @staticmethod
    def to_submission_df(ids, preds: np.ndarray, id_col: str = "sample_id", pred_col: str = "predicted_price") -> pd.DataFrame:
        return pd.DataFrame({id_col: ids, pred_col: preds})

    @staticmethod
    def to_submission_csv(df: pd.DataFrame, path: Optional[str] = None) -> None:
        df.to_csv(path, index=False)