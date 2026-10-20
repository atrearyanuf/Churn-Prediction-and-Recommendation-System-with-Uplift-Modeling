"""Power analysis for the offer's effect on churn (two-sided test of two proportions).

Effects are absolute reductions in churn rate, e.g. 0.01 = one percentage point
fewer churners with the offer than without it. Uses Cohen's h with
statsmodels' normal-approximation power calculations.
"""
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def _check(base_rate, *counts):
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must lie strictly between 0 and 1")
    if any(n <= 0 for n in counts):
        raise ValueError("need at least one customer in each arm")


def _effect_size(base_rate, effect):
    if not 0 < effect < base_rate:
        raise ValueError("effect must be a churn reduction between 0 and base_rate")
    return abs(proportion_effectsize(base_rate - effect, base_rate))


def minimum_detectable_effect(base_rate, n_control, n_treated, alpha=0.05, power=0.8):
    """Smallest churn reduction detected with probability `power`."""
    _check(base_rate, n_control, n_treated)
    h = NormalIndPower().solve_power(nobs1=n_treated, ratio=n_control / n_treated, alpha=alpha, power=power)
    treated_rate = np.sin(np.arcsin(np.sqrt(base_rate)) - h / 2) ** 2
    return float(base_rate - treated_rate)


def power_for_effect(base_rate, effect, n_control, n_treated, alpha=0.05):
    """Probability of detecting a churn reduction of `effect`."""
    _check(base_rate, n_control, n_treated)
    return float(NormalIndPower().power(_effect_size(base_rate, effect), nobs1=n_treated,
                                        ratio=n_control / n_treated, alpha=alpha))


def required_sample_size(base_rate, effect, treated_share, alpha=0.05, power=0.8):
    """Total customers needed to detect `effect` when `treated_share` of them get the offer."""
    _check(base_rate, treated_share, 1 - treated_share)
    n_treated = NormalIndPower().solve_power(_effect_size(base_rate, effect), ratio=(1 - treated_share) / treated_share,
                                             alpha=alpha, power=power)
    return int(np.ceil(n_treated / treated_share))
