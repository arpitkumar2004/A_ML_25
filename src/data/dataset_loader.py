# src/data/dataset_loader.py
from typing import Optional
import pandas as pd
import os
from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory


class DatasetLoader:
    """
    Load train/test CSVs and apply minimal checks.
    Usage:
        loader = DatasetLoader(path="data/raw/train.csv")
        df = loader.load()
    """
    def __init__(self, path: str, required_columns: Optional[list] = None, logger=None):
        self.path = path
        self.required_columns = required_columns or ["unique_identifier", "Description", "Price"]
        self.logger = logger or LoggerFactory.get("DatasetLoader")

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            self.logger.error(f"File not found: {self.path}")
            raise FileNotFoundError(self.path)
        df = IO.read_csv(self.path)
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            self.logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        self.logger.info(f"Loaded {len(df)} rows from {self.path}")
        return df

    def sample(self, frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
        df = self.load()
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
            self.logger.info(f"Sampled {len(df)} rows (frac={frac})")
        return df


def load_train_df(path: str) -> pd.DataFrame:
    """Backward-compatible helper for loading training data."""
    loader = DatasetLoader(path=path, required_columns=["unique_identifier", "Description", "Price"])
    return loader.load()


def load_test_df(path: str) -> pd.DataFrame:
    """Backward-compatible helper for loading test/inference data."""
    loader = DatasetLoader(path=path, required_columns=["unique_identifier", "Description"])
    return loader.load()
