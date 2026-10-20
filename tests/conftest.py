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
