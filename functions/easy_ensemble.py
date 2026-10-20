from functools import reduce
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
from numpy.random import default_rng

class EasyEnsemble:
    def __init__(self, base_model, n_folds, random_state=None, n_jobs = 1, verbose=False):
        self.base_model = base_model
        self.n_folds = n_folds
        self.rng = default_rng(seed=random_state)
        self.models = None
        self.verbose = verbose
        self.n_jobs = n_jobs
    
    def _compute_fold(self, indices, data, *args, **kwargs):
        # deepcopy: a shallow copy shares the wrapped estimator between folds
        model = deepcopy(self.base_model)
        model.fit(data[indices], *args, **kwargs)
        return {"model": model, "training_indices": indices}
    
    def fit(self, data, *args, **kwargs):
        indices_1 = np.flatnonzero(data.y == 1)
        indices_0 = np.flatnonzero(data.y == 0)
        if self.verbose:
            print("{} negative samples and {} positive samples".format(len(indices_0), len(indices_1)))
        # Draw every fold here: parallel workers each get a copy of self.rng in
        # the same state and would all draw the same sample.
        folds = [
            self.rng.permutation(np.hstack((indices_1, self.rng.choice(indices_0, size=len(indices_1)))))
            for _ in range(self.n_folds)
        ]
        self.models = Parallel(
            n_jobs=self.n_jobs,
            verbose=10 if self.verbose else 0
        )(delayed(self._compute_fold)(indices, data, *args, **kwargs) for indices in folds)
    
    def predict(self, data, *args, **kwargs):
        # Use a lambda to reduce the predictions.
        # We could use np.add but it wouldn't work with pandas DataFrames
        predictions = reduce(
            lambda a, b: a + b,
            Parallel(
                n_jobs=self.n_jobs,
                verbose=10 if self.verbose else 0
            )(delayed(model["model"].predict)(data, *args, **kwargs) for model in self.models)
        )
        return predictions / len(self.models)
