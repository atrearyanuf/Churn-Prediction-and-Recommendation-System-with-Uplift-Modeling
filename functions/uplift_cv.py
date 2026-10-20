"""Repeated cross-validation of uplift learners, built on functions/benchmark.py.

A single validation split is too noisy to choose between uplift models on this
data; averaging out-of-fold Qini over several splits is more stable.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions.benchmark import benchmark
from functions.dataset import Dataset
from functions.eval_measures import aipw_pseudo_outcomes, dr_qini_coefficient, ipw_weights, qini_coefficient
from functions.models import make_outcome_model
from functions.targeting import retention_uplift


class PropensityModel:
    """P(treated | X): scaled logistic regression, clipped away from 0 and 1 so
    inverse-propensity weights stay bounded."""

    def __init__(self, clip=0.05):
        self.clip = clip
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, data):
        self.model.fit(data.X, data.t)
        return self

    def predict(self, data):
        return np.clip(self.model.predict_proba(data.X)[:, 1], self.clip, 1 - self.clip)


class OutcomeModels:
    """P(retained | X) without and with the offer: one churn model per arm.
    predict returns an (n, 2) array (mu0, mu1) for doubly-robust effect estimates."""

    def fit(self, data):
        self.models = [make_outcome_model().fit(data.X[data.t == arm], data.y[data.t == arm]) for arm in (0, 1)]
        return self

    def predict(self, data):
        return np.column_stack([1 - m.predict_proba(data.X)[:, 1] for m in self.models])


class UpliftLearner:
    """Fit/predict a CausalML meta-learner on a Dataset; predict returns retention uplift.

    With needs_propensity (X-Learner) a PropensityModel is fitted on the same
    training data and passed to the learner at fit and predict time.
    """

    def __init__(self, learner, needs_propensity=False):
        self.learner = learner
        self.propensity = PropensityModel() if needs_propensity else None

    def _extra(self, data):
        return {} if self.propensity is None else {"p": self.propensity.predict(data)}

    def fit(self, data):
        if self.propensity is not None:
            self.propensity.fit(data)
        self.learner.fit(data.X, treatment=data.t, y=data.y, **self._extra(data))
        return self

    def predict(self, data):
        return retention_uplift(self.learner.predict(data.X, **self._extra(data)))


def cross_validate_uplift(X, t, y, learners, k_folds=5, n_repeats=2, seed=0):
    """Out-of-fold Qini on retention: unweighted, inverse-propensity weighted and
    doubly robust.

    Each split refits every learner, plus the propensity and outcome models used
    for the weights and DR effects, on the training folds only. Returns one row
    per (model, split).
    """
    data = Dataset(X=np.asarray(X), t=np.asarray(t), y=np.asarray(y))
    models = {**learners, "_propensity": PropensityModel(), "_outcomes": OutcomeModels()}
    results = benchmark(data, models, k_folds=k_folds, n_repeats=n_repeats, seed=seed)

    rows = []
    for i, split in enumerate(results):
        test = data[split["test_indices"]]
        propensity = split["results"]["_propensity"]["pred"]
        weights = ipw_weights(test.t, propensity)
        mu = split["results"]["_outcomes"]["pred"]
        effects = aipw_pseudo_outcomes(1 - test.y, test.t, mu[:, 0], mu[:, 1], propensity)
        for name in learners:
            uplift = split["results"][name]["pred"]
            rows.append({
                "model": name,
                "split": i,
                "qini": qini_coefficient(1 - test.y, test.t, uplift),
                "ipw_qini": qini_coefficient(1 - test.y, test.t, uplift, weights),
                "dr_qini": dr_qini_coefficient(test.t, uplift, effects),
            })
    return pd.DataFrame(rows)


def summarize_cv(cv):
    """Per model: mean Qini and IPW Qini, mean and spread of DR Qini, and the
    share of splits where DR Qini beats random targeting. Best mean DR Qini first."""
    by_model = cv.groupby("model")
    return pd.DataFrame({
        "mean_qini": by_model["qini"].mean(),
        "mean_ipw_qini": by_model["ipw_qini"].mean(),
        "mean_dr_qini": by_model["dr_qini"].mean(),
        "std_dr_qini": by_model["dr_qini"].std(),
        "share_dr_qini_above_0": by_model["dr_qini"].apply(lambda s: (s > 0).mean()),
    }).sort_values("mean_dr_qini", ascending=False)
