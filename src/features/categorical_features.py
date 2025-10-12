"""Categorical feature encoders."""
def one_hot_encode(df, cols):
    return df.join(pd.get_dummies(df[cols], drop_first=True))

