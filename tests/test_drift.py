import numpy as np
import pandas as pd
import pytest

from churn_uplift.drift import build_reference, drift_report, psi


def test_psi_is_zero_for_identical_distributions_and_grows_with_shift():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == 0
    assert psi([0.5, 0.5], [0.6, 0.4]) < psi([0.5, 0.5], [0.9, 0.1])


def test_psi_matches_the_textbook_formula():
    expected, actual = np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.3, 0.4])
    assert psi(expected, actual) == pytest.approx(np.sum((actual - expected) * np.log(actual / expected)))


def test_psi_handles_empty_bins():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


def training_like(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"PC1": rng.normal(shift, 1, n), "PC2": rng.normal(0, 1, n),
                         "FACTOR1": rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])})


@pytest.fixture
def reference():
    train = training_like(5000, seed=0)
    return build_reference(train, {"churn_probability": np.random.default_rng(1).beta(1, 20, 5000)})


def test_same_distribution_shows_no_drift(reference):
    report = drift_report(reference, training_like(3000, seed=2),
                          {"churn_probability": np.random.default_rng(3).beta(1, 20, 3000)}, threshold=0.2)

    assert set(report["column"]) == {"PC1", "PC2", "FACTOR1", "churn_probability"}
    assert (report["status"] == "ok").all()


def test_shifted_feature_and_unseen_category_are_flagged(reference):
    batch = training_like(3000, seed=2, shift=1.0)
    batch.loc[:299, "FACTOR1"] = -1   # category the model never saw
    report = drift_report(reference, batch, {"churn_probability": np.random.default_rng(3).beta(1, 20, 3000)},
                          threshold=0.2).set_index("column")

    assert report.loc["PC1", "status"] == "drift"
    assert report.loc["FACTOR1", "status"] == "drift"
    assert report.loc["PC2", "status"] == "ok"
    assert report.index[0] in ("PC1", "FACTOR1")   # sorted by PSI, largest first


def test_prediction_columns_are_labelled(reference):
    report = drift_report(reference, training_like(100, seed=4), {"churn_probability": np.full(100, 0.05)}, 0.2)
    assert report.set_index("column").loc["churn_probability", "kind"] == "prediction"
