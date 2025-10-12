"""BaseModel interface."""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): pass
    @abstractmethod
    def predict(self, X): pass

