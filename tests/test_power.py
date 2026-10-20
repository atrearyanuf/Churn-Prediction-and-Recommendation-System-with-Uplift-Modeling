import numpy as np
import pytest
from scipy.stats import norm

from functions.power import minimum_detectable_effect, power_for_effect, required_sample_size


def simulated_power(base_rate, effect, n_control, n_treated, n_sims=20000, alpha=0.05, seed=0):
    """Share of simulated experiments where a two-proportion z-test detects the effect."""
    rng = np.random.default_rng(seed)
    control = rng.binomial(n_control, base_rate, n_sims) / n_control
    treated = rng.binomial(n_treated, base_rate - effect, n_sims) / n_treated
    pooled = (control * n_control + treated * n_treated) / (n_control + n_treated)
    z = (control - treated) / np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treated))
    return np.mean(np.abs(z) > norm.ppf(1 - alpha / 2))


def test_mde_is_detected_80_percent_of_the_time_in_simulation():
    mde = minimum_detectable_effect(0.036, n_control=2886, n_treated=9010)
    assert simulated_power(0.036, mde, 2886, 9010) == pytest.approx(0.8, abs=0.03)


def test_mde_sample_size_and_power_are_consistent():
    mde = minimum_detectable_effect(0.036, n_control=2886, n_treated=9010)

    assert power_for_effect(0.036, mde, n_control=2886, n_treated=9010) == pytest.approx(0.8, abs=1e-4)
    assert required_sample_size(0.036, mde, treated_share=9010 / 11896) == pytest.approx(11896, rel=0.001)


def test_more_customers_detect_smaller_effects():
    small = minimum_detectable_effect(0.036, n_control=500, n_treated=1500)
    large = minimum_detectable_effect(0.036, n_control=5000, n_treated=15000)
    assert large < small


def test_smaller_effects_need_more_customers():
    assert required_sample_size(0.036, 0.001, 0.75) > required_sample_size(0.036, 0.01, 0.75)


@pytest.mark.parametrize("kwargs, message", [
    (dict(base_rate=0.0, n_control=10, n_treated=10), "base_rate"),
    (dict(base_rate=0.1, n_control=0, n_treated=10), "customer"),
])
def test_mde_rejects_invalid_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        minimum_detectable_effect(**kwargs)


def test_effect_must_be_a_reduction_below_the_base_rate():
    with pytest.raises(ValueError, match="effect"):
        required_sample_size(0.036, 0.05, 0.75)
