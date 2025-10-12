"""Common metrics (RMSE, MAE, RMSLE)."""
def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

