from types import SimpleNamespace

import numpy as np
import pytest


def simulate_campaign(n, seed, treated_share=0.75):
    """Synthetic retention campaign with a known, heterogeneous effect on churn.

    Customers with x1 > 0 are persuadable (the offer cuts their churn odds);
    the rest are sleeping dogs (the offer raises their churn odds).
    Returns X, t, y (y = 1 means churn) and the true churn effect per customer.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    t = (rng.random(n) < treated_share).astype(int)
    persuadable = X[:, 1] > 0

    def churn_prob(treated):
        logit = -1.5 + 0.8 * X[:, 0] + np.where(persuadable, -1.5, 0.8) * treated
        return 1 / (1 + np.exp(-logit))

    y = (rng.random(n) < churn_prob(t)).astype(int)
    churn_effect = churn_prob(1) - churn_prob(0)
    return X, t, y, churn_effect


@pytest.fixture
def campaign():
    return simulate_campaign(4000, seed=0)


def simulate_confounded_campaign(n, seed):
    """Campaign where high-risk customers (large x0) are treated more often.

    Returns a namespace with X, the true propensity, t, y (y = 1 means churn),
    the true retention probabilities without/with the offer (mu0, mu1) and the
    true retention uplift (x1 > 0 persuadable, otherwise sleeping dog).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    propensity = 1 / (1 + np.exp(-1.5 * X[:, 0]))
    t = (rng.random(n) < propensity).astype(int)
    persuadable = X[:, 1] > 0

    def churn_prob(treated):
        logit = -1.5 + 1.2 * X[:, 0] + np.where(persuadable, -1.0, 0.5) * treated
        return 1 / (1 + np.exp(-logit))

    y = (rng.random(n) < churn_prob(t)).astype(int)
    mu0, mu1 = 1 - churn_prob(0), 1 - churn_prob(1)
    return SimpleNamespace(X=X, propensity=propensity, t=t, y=y, mu0=mu0, mu1=mu1, uplift=mu1 - mu0)


def synthetic_customers(n, seed):
    """Campaign data with the real dataset's layout: PC* numbers, FACTOR* categories, y, t."""
    X, t, y, _ = simulate_campaign(n, seed)
    rng = np.random.default_rng(seed + 100)
    frame = {f"PC{i + 1}": X[:, i] for i in range(X.shape[1])}
    frame["FACTOR1"] = np.where(X[:, 2] > 0.5, "V1", np.where(X[:, 2] > -0.5, "V2", "V3"))
    frame["FACTOR2"] = rng.choice(["A", "B"], n)
    frame["y"], frame["t"] = y, t
    import pandas as pd
    return pd.DataFrame(frame)
