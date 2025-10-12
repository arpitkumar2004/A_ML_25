"""Feature parsing helpers (counts, weights)."""
import numpy as np
import pandas as pd
import re

def parse_ounces(text):
    # returns total ounces found in text (simple)
    if not isinstance(text, str):
        return 0.0
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*oz', text, flags=re.I)
    vals = [float(m) for m in matches]
    return sum(vals)

def add_parsed_features(df, text_col='Description'):
    df = df.copy()
    df["desc_clean_len"] = df[text_col].astype(str).apply(len)
    df["parsed_ounces"] = df[text_col].astype(str).apply(parse_ounces)
    return df
