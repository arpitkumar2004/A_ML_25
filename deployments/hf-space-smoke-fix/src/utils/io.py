# src/utils/io.py
from typing import Any, Optional
import joblib
import json
import os
import pandas as pd


class IO:
    """
    IO helper for saving/loading artifacts, arrays, and dataframes.
    Methods are static for convenience.
    """
    @staticmethod
    def ensure_dir(path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def save_pickle(obj: Any, path: str) -> None:
        IO.ensure_dir(path)
        joblib.dump(obj, path)

    @staticmethod
    def load_pickle(path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return joblib.load(path)

    @staticmethod
    def save_json(obj: Any, path: str, indent: int = 2) -> None:
        IO.ensure_dir(path)
        with open(path, "w", encoding="utf8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def load_json(path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def read_csv(path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def to_csv(df, path: str, index: bool = False) -> None:
        IO.ensure_dir(path)
        df.to_csv(path, index=index)

    @staticmethod
    def read_parquet(path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_parquet(path, **kwargs)
    
    @staticmethod
    def to_parquet(df, path: str, index: bool = False) -> None:
        IO.ensure_dir(path)
        df.to_parquet(path, index=index)    
        
    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> list:
        if not os.path.exists(directory):
            raise FileNotFoundError(directory)
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extension is None or filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files
    
    @staticmethod
    def file_exists(path: str) -> bool:
        return os.path.exists(path)
    
    @staticmethod
    def remove_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
    