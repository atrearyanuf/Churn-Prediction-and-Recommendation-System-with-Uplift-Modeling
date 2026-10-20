import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from functions.models import (
    encode_factors, make_churn_models, make_outcome_model, make_propensity_model,
    make_uplift_forest, make_uplift_learners, split_dataset,
)


def frame():
    return pd.DataFrame({"PC1": [0.5, -1.0, 2.0, 0.1], "FACTOR1": ["V2", "V10", "V2", "V1"],
                         "y": [0, 1, 0, 0], "t": [1, 0, 1, 1]})


def test_factors_are_label_encoded_in_sorted_order_and_numbers_untouched():
    encoded, encoders = encode_factors(frame())

    assert encoded["FACTOR1"].tolist() == [2, 1, 2, 0]   # sorted: V1, V10, V2
    assert encoded["PC1"].tolist() == frame()["PC1"].tolist()
    assert list(encoders) == ["FACTOR1"]


def test_unseen_categories_become_minus_one_when_reusing_encoders():
    _, encoders = encode_factors(frame())
    new = frame().assign(FACTOR1=["V2", "V99", "V1", "V1"])

    encoded, _ = encode_factors(new, encoders)
    assert encoded["FACTOR1"].tolist() == [2, -1, 0, 0]


def test_split_is_60_20_20_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    n = 5000
    X, y, t = rng.normal(size=(n, 2)), (rng.random(n) < 0.05).astype(int), (rng.random(n) < 0.75).astype(int)
    parts = split_dataset(X, y, t)

    sizes = [len(parts[s][1]) for s in ("train", "val", "test")]
    assert sizes == [3000, 1000, 1000]
    rows = np.concatenate([parts[s][0][:, 0] for s in ("train", "val", "test")])
    assert np.array_equal(np.sort(rows), np.sort(X[:, 0]))
    for s in ("train", "val", "test"):
        assert parts[s][1].mean() == pytest.approx(y.mean(), abs=0.003)
        assert parts[s][2].mean() == pytest.approx(t.mean(), abs=0.005)


def test_split_is_deterministic():
    X, y, t = np.arange(200).reshape(100, 2), np.arange(100) % 2, (np.arange(100) // 2) % 2
    a, b = split_dataset(X, y, t), split_dataset(X, y, t)
    for s in ("train", "val", "test"):
        np.testing.assert_array_equal(a[s][0], b[s][0])


def test_factories_return_fresh_unfitted_models():
    assert make_churn_models(1.0)["XGBoost"] is not make_churn_models(1.0)["XGBoost"]
    assert list(make_churn_models(1.0)) == ["RandomForest", "XGBoost", "LightGBM"]
    assert list(make_uplift_learners()) == ["T-Learner", "S-Learner", "X-Learner", "DR-Learner"]
    assert make_propensity_model().get_params()["max_iter"] == 1000


def all_xgboost_models():
    objs = [make_churn_models(1.0)["XGBoost"], make_outcome_model()]
    for learner in make_uplift_learners().values():
        objs += [v for v in vars(learner).values() if isinstance(v, (XGBClassifier, XGBRegressor))]
        for v in vars(learner).values():
            if isinstance(v, dict):
                objs += [m for m in v.values() if isinstance(m, (XGBClassifier, XGBRegressor))]
    return objs


def test_no_xgboost_model_uses_column_sampling():
    # colsample_bytree gives different results on macOS and Linux.
    models = all_xgboost_models()
    assert len(models) >= 6
    for model in models:
        assert model.get_params().get("colsample_bytree") in (None, 1, 1.0)


def test_uplift_forest_uses_string_treatment_labels():
    assert make_uplift_forest().control_name == "0"


def test_dr_learner_cross_fitting_is_reproducible():
    from tests.conftest import simulate_campaign
    X, t, y, _ = simulate_campaign(3000, seed=0)
    p = np.full(len(t), t.mean())
    first, second = make_uplift_learners()["DR-Learner"], make_uplift_learners()["DR-Learner"]
    first.fit(X, treatment=t, y=y, p=p)
    second.fit(X, treatment=t, y=y, p=p)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
