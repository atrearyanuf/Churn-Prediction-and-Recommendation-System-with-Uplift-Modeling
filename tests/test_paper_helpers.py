"""Helpers adapted from the dataset paper's code: functions/benchmark.py,
dataset.py, easy_ensemble.py, wrappers.py and the profit curves in
eval_measures.py."""
import numpy as np
import pandas as pd
import pytest

from functions.benchmark import benchmark, generate_splits
from functions.dataset import Dataset
from functions.easy_ensemble import EasyEnsemble
from functions.eval_measures import calibrate_score, extract_cb, uplift_curve
from functions.wrappers import RandomForestWrapper, TLearnerWrapper, URFCWrapper


class Recorder:
    """Minimal model that remembers its training labels in a mutable attribute."""

    def __init__(self):
        self.seen = {}

    def fit(self, data):
        self.seen["y"] = data.y.copy()

    def predict(self, data):
        return np.full(len(data.y), self.seen["y"].mean())


def opposite_arms(n, t_dtype, seed=0):
    """Control customers churn when x0 > 0, treated customers when x0 < 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = (rng.random(n) < 0.5).astype(t_dtype)
    y = np.where(t == 1, X[:, 0] < 0, X[:, 0] > 0).astype(int)
    return Dataset(X=X, y=y, t=t)


# -- Dataset ---------------------------------------------------------------

def test_dataset_indexing_keeps_arrays_aligned_and_copies():
    data = Dataset(X=np.arange(10).reshape(5, 2), y=np.arange(5))
    subset = data[[4, 0]]
    subset.y[0] = 99

    np.testing.assert_array_equal(subset.X, [[8, 9], [0, 1]])
    assert data.y[4] == 4


def test_dataset_train_test_split_keeps_arrays_aligned():
    data = Dataset(X=np.arange(20).reshape(10, 2), y=np.arange(10))
    train, test = data.train_test_split(test_size=0.3, random_state=0)

    assert len(train.y) == 7 and len(test.y) == 3
    for part in (train, test):
        np.testing.assert_array_equal(part.X[:, 0] // 2, part.y)


# -- benchmark ---------------------------------------------------------------

def test_every_customer_is_tested_exactly_once_per_repeat():
    splits = generate_splits(103, k_folds=5, n_repeats=2, rng=np.random.default_rng(0))

    assert len(splits) == 10
    for repeat in (splits[:5], splits[5:]):
        np.testing.assert_array_equal(np.sort(np.concatenate(repeat)), np.arange(103))


def test_benchmark_trains_on_the_complement_of_each_test_fold():
    data = Dataset(X=np.zeros((50, 1)), y=np.arange(50) % 2)
    template = Recorder()
    results = benchmark(data, {"recorder": template}, k_folds=5, n_repeats=1, seed=0)

    assert template.seen == {}  # each split fits its own copy
    for split in results:
        model = split["results"]["recorder"]["model"]
        assert len(model.seen["y"]) == 50 - len(split["test_indices"])

    again = benchmark(data, {"recorder": Recorder()}, past_results=results)
    for old, new in zip(results, again):
        np.testing.assert_array_equal(old["test_indices"], new["test_indices"])


# -- EasyEnsemble ------------------------------------------------------------

@pytest.fixture
def imbalanced():
    # Positives first, so sampling from the wrong index range shows up.
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    return Dataset(X=np.arange(100).reshape(100, 1), y=y)


def test_each_fold_is_balanced_and_negatives_come_from_negative_rows(imbalanced):
    ensemble = EasyEnsemble(Recorder(), n_folds=20, random_state=0)
    ensemble.fit(imbalanced)

    for fold in ensemble.models:
        labels = imbalanced.y[fold["training_indices"]]
        assert labels.sum() == 10 and len(labels) == 20


def test_each_member_is_trained_on_its_own_fold(imbalanced):
    ensemble = EasyEnsemble(Recorder(), n_folds=5, random_state=0)
    ensemble.fit(imbalanced)

    for fold in ensemble.models:
        np.testing.assert_array_equal(fold["model"].seen["y"], imbalanced.y[fold["training_indices"]])


def test_parallel_folds_draw_different_samples(imbalanced):
    ensemble = EasyEnsemble(Recorder(), n_folds=4, random_state=0, n_jobs=2)
    ensemble.fit(imbalanced)

    samples = {tuple(np.sort(f["training_indices"])) for f in ensemble.models}
    assert len(samples) == 4


def test_easy_ensemble_is_reproducible_and_averages_members(imbalanced):
    a = EasyEnsemble(Recorder(), n_folds=3, random_state=1)
    b = EasyEnsemble(Recorder(), n_folds=3, random_state=1)
    a.fit(imbalanced)
    b.fit(imbalanced)

    for fa, fb in zip(a.models, b.models):
        np.testing.assert_array_equal(fa["training_indices"], fb["training_indices"])
    np.testing.assert_allclose(a.predict(imbalanced), 0.5)  # every fold is half positives


# -- wrappers --------------------------------------------------------------

@pytest.mark.parametrize("t_dtype", [int, bool])
def test_random_forest_wrapper_learns_the_control_arm(t_dtype):
    data = opposite_arms(2000, t_dtype)
    model = RandomForestWrapper(n_estimators=20, random_state=0)
    model.fit(data)

    p = model.predict(data)
    assert p[data.X[:, 0] > 0].mean() > 0.8 > 0.2 > p[data.X[:, 0] < 0].mean()


@pytest.mark.parametrize("t_dtype", [int, bool])
def test_t_learner_wrapper_fits_one_model_per_arm(t_dtype):
    data = opposite_arms(2000, t_dtype)
    model = TLearnerWrapper(n_estimators=20, random_state=0)
    model.fit(data)

    pred = model.predict(data)
    assert list(pred.columns) == ["control", "target"]
    high = data.X[:, 0] > 0
    assert pred["control"][high].mean() > 0.8 and pred["target"][high].mean() < 0.2


def test_uplift_forest_wrapper_ranks_where_treatment_raises_the_outcome():
    data = opposite_arms(3000, int)
    model = URFCWrapper(n_estimators=10, max_depth=3, random_state=0)
    model.fit(data)

    uplift = np.asarray(model.predict(data)).ravel()
    assert uplift[data.X[:, 0] < 0].mean() > uplift[data.X[:, 0] > 0].mean()


# -- profit curves -----------------------------------------------------------

def test_extract_cb_reads_single_and_per_customer_matrices():
    cb = np.array([[1, 2], [3, 4]])
    assert extract_cb(cb) == (1, 2, 3, 4)
    per_customer = np.stack([cb, cb + 10])
    np.testing.assert_array_equal(extract_cb(per_customer)[3], [4, 14])


def test_uplift_curve_counts_the_top_k_customers():
    # Ranked by score: (y, t) = (0,1), (0,0), (1,0), (1,1)
    y = np.array([1, 0, 0, 1])
    t = np.array([1, 0, 1, 0])
    score = np.array([0.1, 0.4, 0.9, 0.2])

    profit = uplift_curve(y, t, score)["profit"].to_numpy()
    # Retention rate treated minus control among the top k, times k
    # (an arm with no customers yet counts as rate 0).
    expected = [1 * (1 - 0), 2 * (1 - 1), 3 * (1 - 0.5), 4 * (0.5 - 0.5)]
    np.testing.assert_allclose(profit, expected)


def test_calibrate_score_undoes_resampling_prior_shift():
    score = np.array([0.1, 0.3, 0.5, 0.7])
    np.testing.assert_allclose(calibrate_score(score, score.mean()), score)

    corrected = calibrate_score(score, p=0.05)
    assert np.all(np.diff(corrected) > 0)
    assert np.all((0 < corrected) & (corrected < score))
