"""Numeric transformations."""
def scale_numeric(df, cols):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

