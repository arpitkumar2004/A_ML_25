"""Dataset loader utilities."""
import os
import pandas as pd

class DatasetLoader:
    def __init__(self, path):
        self.path = path
        
    def load(self):
        return pd.read_csv(self.path)
    
    def save(self, df):
        df.to_csv(self.path, index=False)
        print(f"Dataset saved to {self.path}")
        
class DatasetSplitter:
    def __init__(self, df):
        self.df = df
        
    def train_test_split(self, test_size=0.2, random_state=42):
        train_df = self.df.sample(frac=1 - test_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)
        return train_df, test_df
    
    def stratified_split(self, target_col, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(self.df, test_size=test_size, 
                                             stratify=self.df[target_col], 
                                             random_state=random_state)
        return train_df, test_df
    
    
class DataScaling:
    def __init__(self, df):
        self.df = df
        
    def scale_numeric(self, cols):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        return self.df
    def min_max_scale(self, cols):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        return self.df
    def robust_scale(self, cols):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        return self.df
    
class DataTransfromation:
    def __init__(self, df):
        self.df = df
        
    def log_transform(self, cols):
        for col in cols:
            self.df[col] = self.df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        return self.df
    
    def sqrt_transform(self, cols):
        for col in cols:
            self.df[col] = self.df[col].apply(lambda x: np.sqrt(x) if x >= 0 else 0)
        return self.df
    
    def boxcox_transform(self, cols):
        from scipy import stats
        for col in cols:
            self.df[col], _ = stats.boxcox(self.df[col] + 1)  # Adding 1 to avoid zero values
        return self.df