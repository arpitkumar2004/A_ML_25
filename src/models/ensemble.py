import numpy as np
class SimpleEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [1.0/len(models)]*len(models)
        else:
            self.weights = weights

    def predict(self, Xs):
        # Xs: list of inputs for each model (or same input)
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        weighted = (preds.T * np.array(self.weights)).sum(axis=1)
        return weighted

