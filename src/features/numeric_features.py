import numpy as np

def build_numeric_features(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = ['desc_clean_len', 'parsed_ounces']
    X = df[numeric_cols].fillna(0).values.astype(float)
    return X

