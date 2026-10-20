import numpy as np
import pandas as pd
import pytest
from causalml.metrics import get_qini, qini_score

from functions.eval_measures import bootstrap_qini, qini_coefficient, qini_curve


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
