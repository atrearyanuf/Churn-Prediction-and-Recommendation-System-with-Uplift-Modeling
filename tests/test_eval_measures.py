import numpy as np
import pandas as pd
import pytest
from causalml.metrics import get_qini, qini_score

from functions.eval_measures import (
    aipw_pseudo_outcomes, bootstrap_dr_qini, bootstrap_qini, dr_qini_coefficient,
    ipw_weights, qini_coefficient, qini_curve,
)
from tests.conftest import simulate_confounded_campaign


def test_qini_matches_causalml_for_untied_scores(campaign):
    X, t, y, effect = campaign
    score = effect + np.random.default_rng(1).normal(scale=0.05, size=len(y))
    df = pd.DataFrame({"y": y, "w": t, "model": score})

    reference = get_qini(df, outcome_col="y", treatment_col="w")["model"].to_numpy()

    np.testing.assert_allclose(qini_curve(y, t, score), reference)
    assert qini_coefficient(y, t, score) == pytest.approx(
        qini_score(df, outcome_col="y", treatment_col="w", normalize=False)["model"])


def test_constant_score_is_exactly_random_targeting(campaign):
    _, t, y, _ = campaign
    curve = qini_curve(y, t, np.zeros(len(y)))

    np.testing.assert_allclose(curve, np.linspace(0, curve[-1], len(curve)))
    assert qini_coefficient(y, t, np.zeros(len(y))) == pytest.approx(0, abs=1e-12)


def test_ties_do_not_depend_on_row_order(campaign):
    X, t, y, effect = campaign
    score = np.round(effect, 1)  # few distinct values, many ties
    perm = np.random.default_rng(2).permutation(len(y))

    assert qini_coefficient(y, t, score) == pytest.approx(
        qini_coefficient(y[perm], t[perm], score[perm]))


def test_ranking_by_true_benefit_beats_random_and_reverse_loses(campaign):
    _, t, y, effect = campaign
    retained = 1 - y

    assert qini_coefficient(retained, t, -effect) > 0
    assert qini_coefficient(retained, t, effect) < 0


def test_churn_and_retention_framings_give_the_same_qini(campaign):
    # Ranking by churn effect against churn ~= ranking by retention uplift against
    # retention, so the earlier Qini values were not sign-flipped. Not exact:
    # Qini rescales control outcomes by n_t / n_c within each top-k, and that
    # ratio differs between the top and bottom of the ranking (~0.5% here,
    # ~1.2% with 7 coarse tie blocks).
    _, t, y, effect = campaign
    for score in (effect, np.round(effect, 1)):
        assert qini_coefficient(y, t, score) == pytest.approx(
            qini_coefficient(1 - y, t, -score), rel=0.02)


@pytest.mark.parametrize("y, t, score, message", [
    ([0, 1], [1, 1], [0.1, 0.2], "both treated and control"),
    ([0, 1], [0, 2], [0.1, 0.2], "binary"),
    ([0, 1], [0, 1], [0.1, np.nan], "finite"),
    ([0, 1, 1], [0, 1], [0.1, 0.2], "same length"),
])
def test_qini_rejects_invalid_input(y, t, score, message):
    with pytest.raises(ValueError, match=message):
        qini_curve(y, t, score)


@pytest.mark.parametrize("weights", [[1.0, -1.0], [1.0, np.inf], [1.0]])
def test_qini_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        qini_curve([0, 1], [0, 1], [0.1, 0.2], weights)


def test_unit_weights_give_the_standard_qini(campaign):
    _, t, y, effect = campaign
    np.testing.assert_allclose(qini_curve(1 - y, t, -effect, np.ones(len(y))),
                               qini_curve(1 - y, t, -effect))


def test_ipw_weights_invert_the_propensity():
    np.testing.assert_allclose(ipw_weights([1, 0], [0.8, 0.8]), [1.25, 5.0])
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ipw_weights([1, 0], [1.0, 0.5])


def true_qini(t, score, uplift):
    """Qini from the known effects: n_t(k) * mean true uplift among the top k."""
    order = np.argsort(-score, kind="stable")
    k = np.arange(1, len(t) + 1)
    curve = np.r_[0, np.cumsum(uplift[order]) / k * np.cumsum(t[order])]
    return float((curve - np.linspace(0, curve[-1], len(curve))).mean())


def test_ipw_qini_removes_confounding_bias():
    # High-risk customers are treated more often, so among the top-ranked the
    # treated churn more for reasons unrelated to the offer. IPW is unbiased but
    # noisy (~10% per sample here), so compare average errors over 10 samples.
    naive_err, ipw_err = [], []
    for seed in range(10):
        c = simulate_confounded_campaign(50000, seed)
        X, propensity, t, y, uplift = c.X, c.propensity, c.t, c.y, c.uplift
        score = X[:, 0] + X[:, 1]
        truth = true_qini(t, score, uplift)
        naive_err.append(qini_coefficient(1 - y, t, score) / truth - 1)
        ipw_err.append(qini_coefficient(1 - y, t, score, ipw_weights(t, propensity)) / truth - 1)

    assert np.mean(naive_err) < -0.3
    assert abs(np.mean(ipw_err)) < 0.1


def test_weighted_bootstrap_resamples_weights_with_rows():
    c = simulate_confounded_campaign(4000, seed=1)
    propensity, t, y, uplift = c.propensity, c.t, c.y, c.uplift
    w = ipw_weights(t, propensity)
    estimate, lower, upper = bootstrap_qini(1 - y, t, uplift, w, n_boot=200)

    assert estimate == qini_coefficient(1 - y, t, uplift, w)
    assert 0 < lower <= estimate <= upper


def test_bootstrap_is_reproducible_and_brackets_the_estimate(campaign):
    _, t, y, effect = campaign
    result = bootstrap_qini(1 - y, t, -effect, n_boot=200, seed=3)

    assert result == bootstrap_qini(1 - y, t, -effect, n_boot=200, seed=3)
    estimate, lower, upper = result
    assert lower <= estimate <= upper


def test_bootstrap_detects_real_signal(campaign):
    _, t, y, effect = campaign
    _, lower, _ = bootstrap_qini(1 - y, t, -effect, n_boot=200)

    assert lower > 0


def test_bootstrap_interval_covers_zero_for_a_useless_model(campaign):
    _, t, y, _ = campaign
    noise = np.random.default_rng(4).normal(size=len(y))
    _, lower, upper = bootstrap_qini(1 - y, t, noise, n_boot=200)

    assert lower < 0 < upper


# -- doubly-robust Qini ------------------------------------------------------

def test_aipw_pseudo_outcomes_are_unbiased_for_the_effect():
    c = simulate_confounded_campaign(200000, seed=2)
    psi = aipw_pseudo_outcomes(1 - c.y, c.t, c.mu0, c.mu1, c.propensity)

    assert psi.mean() == pytest.approx(c.uplift.mean(), abs=0.005)


def test_dr_qini_with_true_effects_is_the_true_qini(campaign):
    _, t, y, effect = campaign
    uplift = -effect
    score = uplift + np.random.default_rng(5).normal(scale=0.05, size=len(t))

    assert dr_qini_coefficient(t, score, uplift) == pytest.approx(true_qini(t, score, uplift))


def test_dr_qini_constant_score_is_random_targeting(campaign):
    _, t, y, effect = campaign
    assert dr_qini_coefficient(t, np.zeros(len(t)), -effect) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("outcome_models, propensity_model", [
    ("true", "true"),
    ("wrong", "true"),   # double robustness: a correct propensity is enough ...
    ("true", "wrong"),   # ... and so are correct outcome models
])
def test_dr_qini_is_unbiased_if_either_model_is_right(outcome_models, propensity_model):
    errors = []
    for seed in range(10):
        c = simulate_confounded_campaign(50000, seed)
        score = c.X[:, 0] + c.X[:, 1]
        mu0, mu1 = (c.mu0, c.mu1) if outcome_models == "true" else (np.full(len(c.t), 0.7), np.full(len(c.t), 0.7))
        p = c.propensity if propensity_model == "true" else np.full(len(c.t), 0.5)
        psi = aipw_pseudo_outcomes(1 - c.y, c.t, mu0, mu1, p)
        errors.append(dr_qini_coefficient(c.t, score, psi) / true_qini(c.t, score, c.uplift) - 1)

    assert abs(np.mean(errors)) < 0.1


def test_dr_qini_is_less_noisy_than_ipw_with_good_outcome_models():
    dr, ipw = [], []
    for seed in range(10):
        c = simulate_confounded_campaign(50000, seed)
        score = c.X[:, 0] + c.X[:, 1]
        truth = true_qini(c.t, score, c.uplift)
        psi = aipw_pseudo_outcomes(1 - c.y, c.t, c.mu0, c.mu1, c.propensity)
        dr.append(dr_qini_coefficient(c.t, score, psi) / truth)
        ipw.append(qini_coefficient(1 - c.y, c.t, score, ipw_weights(c.t, c.propensity)) / truth)

    assert np.std(dr) < np.std(ipw)


def test_dr_bootstrap_is_reproducible_and_brackets_the_estimate():
    c = simulate_confounded_campaign(4000, seed=3)
    psi = aipw_pseudo_outcomes(1 - c.y, c.t, c.mu0, c.mu1, c.propensity)
    result = bootstrap_dr_qini(c.t, c.uplift, psi, n_boot=200, seed=1)

    assert result == bootstrap_dr_qini(c.t, c.uplift, psi, n_boot=200, seed=1)
    assert result[1] <= result[0] <= result[2]
    assert result[0] == dr_qini_coefficient(c.t, c.uplift, psi)


@pytest.mark.parametrize("mu0, p, message", [
    ([0.5, np.nan], [0.5, 0.5], "finite"),
    ([0.5, 0.5], [0.0, 0.5], "strictly between 0 and 1"),
    ([0.5], [0.5, 0.5], "same length"),
])
def test_aipw_rejects_invalid_input(mu0, p, message):
    with pytest.raises(ValueError, match=message):
        aipw_pseudo_outcomes([1, 0], [1, 0], mu0, [0.5, 0.5], p)
