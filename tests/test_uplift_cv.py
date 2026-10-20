import numpy as np
from causalml.inference.meta import BaseTClassifier, BaseXClassifier
from xgboost import XGBClassifier, XGBRegressor

from functions.dataset import Dataset
from functions.uplift_cv import PropensityModel, UpliftLearner, cross_validate_uplift, summarize_cv
from tests.conftest import simulate_campaign


def small_xgb(cls=XGBClassifier):
    return cls(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0, n_jobs=1)


class NoiseLearner:
    """Ignores the data: random scores."""

    def fit(self, data):
        self.rng = np.random.default_rng(0)

    def predict(self, data):
        return self.rng.normal(size=len(data.y))


class MemorizingLearner:
    """Scores 1 for customers it was trained on, 0 for unseen ones."""

    def fit(self, data):
        self.seen = {row.tobytes() for row in data.X}

    def predict(self, data):
        return np.array([float(row.tobytes() in self.seen) for row in data.X])


def test_cv_scores_every_learner_on_every_split():
    X, t, y, _ = simulate_campaign(1500, seed=0)
    cv = cross_validate_uplift(X, t, y, {"noise": NoiseLearner(), "memo": MemorizingLearner()},
                               k_folds=3, n_repeats=2)

    assert len(cv) == 12
    assert cv.groupby("model")["split"].nunique().to_dict() == {"memo": 6, "noise": 6}


def test_cv_scores_only_out_of_fold_customers():
    # A learner that saw a test customer during training would score it 1.
    X, t, y, _ = simulate_campaign(1500, seed=0)
    cv = cross_validate_uplift(X, t, y, {"memo": MemorizingLearner()}, k_folds=3, n_repeats=1)

    np.testing.assert_allclose(cv["qini"], 0, atol=1e-12)
    np.testing.assert_allclose(cv["ipw_qini"], 0, atol=1e-12)


def test_cv_ranks_a_real_learner_above_noise():
    X, t, y, _ = simulate_campaign(6000, seed=1)
    learners = {"T-Learner": UpliftLearner(BaseTClassifier(learner=small_xgb())), "noise": NoiseLearner()}
    summary = summarize_cv(cross_validate_uplift(X, t, y, learners, k_folds=3, n_repeats=1))

    assert summary.index[0] == "T-Learner"
    assert summary.loc["T-Learner", "mean_ipw_qini"] > 0
    assert summary.loc["T-Learner", "share_ipw_qini_above_0"] == 1.0


def test_uplift_learner_returns_retention_uplift():
    X, t, y, _ = simulate_campaign(2000, seed=2)
    data = Dataset(X=X, t=t, y=y)
    learner = BaseTClassifier(learner=small_xgb())
    wrapped = UpliftLearner(learner)
    wrapped.fit(data)

    np.testing.assert_allclose(wrapped.predict(data), -learner.predict(X).ravel())


def test_x_learner_gets_fold_propensity_scores():
    X, t, y, _ = simulate_campaign(2000, seed=3)
    data = Dataset(X=X, t=t, y=y)
    wrapped = UpliftLearner(BaseXClassifier(outcome_learner=small_xgb(), effect_learner=small_xgb(XGBRegressor)),
                            needs_propensity=True)
    wrapped.fit(data)

    assert wrapped.predict(data).shape == (2000,)
    assert wrapped.propensity.model[-1].coef_.shape == (1, 5)  # fitted on this data's features


def test_propensity_model_is_clipped():
    X = np.r_[np.full((50, 1), -10.0), np.full((50, 1), 10.0)]
    t = np.r_[np.zeros(50, int), np.ones(50, int)]
    data = Dataset(X=X, t=t, y=np.zeros(100, int))

    p = PropensityModel(clip=0.05).fit(data).predict(data)
    assert p.min() == 0.05 and p.max() == 0.95
