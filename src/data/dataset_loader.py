"""Dataset loader utilities."""
import os
import pandas as pd

class DatasetLoader:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load_train_df(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Train csv not found: {path}")
        df = pd.read_csv(path)
        # Basic sanity checks
        required = ['sample_id', 'catalog_content', 'price']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"Missing required column: {r}")
        return df
    
    def load_test_df(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test csv not found: {path}")
        df = pd.read_csv(path)
        # Basic sanity checks
        required = ['sample_id', 'catalog_content', 'image_link']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"Missing required column: {r}")
        return df
    
    def load_sample_out(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample output csv not found: {path}")
        df = pd.read_csv(path)
        # Basic sanity checks
        required = ['sample_id', 'price']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"Missing required column: {r}")
        return df
        
    def save_submission(self, df, path: str="data\submissions\submission.csv"):
        df.to_csv(path, index=False)
        print(f"Submission saved to {path}")