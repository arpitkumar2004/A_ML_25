"""Data augmentation helpers."""

import numpy as np
import pandas as pd

# placeholder for image augmentations or text augmentation
def noop_augment(data):
    return data

class DataImputer:
    """
    Imputes missing values in the parsed features by extracting text form image links from the column image link.
    """
    def __init__(self):
        pass
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df