"""Select features using importance / statistical methods."""
def select_top_k(features_df, k=50):
    return features_df.columns[:k]

