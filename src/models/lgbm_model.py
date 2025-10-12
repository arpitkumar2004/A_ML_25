import lightgbm as lgb

class LGBMModel:
    def __init__(self, params, num_boost_round=1000):
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = lgb.LGBMRegressor(**params, n_estimators=num_boost_round)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        # LightGBM 4.x: early stopping via callbacks
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric="rmse",
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration_)
