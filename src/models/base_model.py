# src/models/base_model.py
from typing import Any
import abc
import joblib
import os


class BaseModel(abc.ABC):
    """
    Abstract interface for model wrappers.
    All model wrappers should inherit from this and implement fit/predict/save/load.
    """

    @abc.abstractmethod
    def fit(self, X, y, eval_set=None):
        ...

    @abc.abstractmethod
    def predict(self, X):
        ...

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return joblib.load(path)
