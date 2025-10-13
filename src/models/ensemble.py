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


class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, Xs, y):
        # Xs: list of inputs for each base model
        base_preds = []
        for m, x in zip(self.base_models, Xs):
            m.fit(x, y)
            base_preds.append(m.predict(x))
        base_preds = np.vstack(base_preds).T  # n_rows x n_models
        self.meta_model.fit(base_preds, y)
        return self

    def predict(self, Xs):
        base_preds = []
        for m, x in zip(self.base_models, Xs):
            base_preds.append(m.predict(x))
        base_preds = np.vstack(base_preds).T  # n_rows x n_models
        return self.meta_model.predict(base_preds)
    
class VotingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds.astype(int))
        return majority_vote
    
class AveragingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        avg_preds = preds.mean(axis=0)
        return avg_preds
    
class WeightedAveragingEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        weighted_avg = np.average(preds, axis=0, weights=self.weights)
        return weighted_avg
    
class MaxVotingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        max_vote = np.max(preds, axis=0)
        return max_vote
    
class MinVotingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        min_vote = np.min(preds, axis=0)
        return min_vote
    
class MedianEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = []
        for m, x in zip(self.models, Xs):
            preds.append(m.predict(x))
        preds = np.vstack(preds)  # n_models x n_rows
        median_preds = np.median(preds, axis=0)
        return median_preds