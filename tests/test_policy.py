import numpy as np
import pytest

from functions.eval_measures import aipw_pseudo_outcomes
from functions.policy import evaluate_policies, policy_profit
from tests.conftest import simulate_confounded_campaign

VALUE, COST = 1000, 40


def true_profit(treat, c):
    return float(np.sum(treat * (VALUE * c.uplift - COST)))


def test_contacting_no_one_earns_nothing_and_everyone_pays_every_offer():
    effects = np.array([0.10, -0.05, 0.02])

    assert policy_profit(np.zeros(3, bool), effects, VALUE, COST) == 0
    assert policy_profit(np.ones(3, bool), effects, VALUE, COST) == pytest.approx(100 - 50 + 20 - 3 * COST)


def test_targeting_true_persuadables_beats_both_blanket_policies():
    c = simulate_confounded_campaign(20000, seed=0)
    oracle = VALUE * c.uplift > COST

    profits = {name: policy_profit(treat, c.uplift, VALUE, COST)
               for name, treat in {"oracle": oracle, "all": np.ones(len(c.t), bool),
                                   "none": np.zeros(len(c.t), bool)}.items()}
    assert profits["oracle"] > max(profits["all"], profits["none"])


def test_dr_profit_estimate_is_unbiased_despite_confounding():
    errors = []
    for seed in range(10):
        c = simulate_confounded_campaign(50000, seed)
        policy = c.X[:, 1] > 0   # contact the true persuadable segment
        psi = aipw_pseudo_outcomes(1 - c.y, c.t, c.mu0, c.mu1, c.propensity)
        errors.append(policy_profit(policy, psi, VALUE, COST) / true_profit(policy, c) - 1)

    assert abs(np.mean(errors)) < 0.05


def test_evaluate_policies_reports_profit_with_intervals():
    c = simulate_confounded_campaign(4000, seed=1)
    psi = aipw_pseudo_outcomes(1 - c.y, c.t, c.mu0, c.mu1, c.propensity)
    policies = {"none": np.zeros(len(c.t), bool), "persuadables": c.X[:, 1] > 0}

    table = evaluate_policies(policies, c.t, psi, VALUE, COST, n_boot=200)

    assert list(table["policy"]) == ["none", "persuadables"]
    assert table.loc[0, ["profit", "profit_low", "profit_high"]].tolist() == [0, 0, 0]
    row = table.loc[1]
    assert row["contacted"] == policies["persuadables"].sum()
    assert row["profit_low"] <= row["profit"] <= row["profit_high"]
    assert row["profit"] == pytest.approx(policy_profit(policies["persuadables"], psi, VALUE, COST))


def test_policy_must_have_one_decision_per_customer():
    with pytest.raises(ValueError, match="one decision per customer"):
        policy_profit(np.ones(2, bool), np.zeros(3), VALUE, COST)
