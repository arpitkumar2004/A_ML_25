# src/utils/seed_everything.py
from typing import Optional
import os
import random
import numpy as np

try:
    import torch
except Exception:
    torch = None


class Seed:
    """
    Utility to set seeds for reproducibility.
    Usage:
        Seed.set(42)
    """
    @staticmethod
    def set(seed: int = 42, use_torch: bool = True) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        if use_torch and torch is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # for determinism (may slow down)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False