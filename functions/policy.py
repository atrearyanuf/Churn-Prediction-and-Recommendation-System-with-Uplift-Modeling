"""Estimate what a targeting policy would earn, from campaign data.

Profit is measured against contacting no one: every contacted customer costs
the offer and earns customer_value times their retention effect. Using
doubly-robust per-customer effects (eval_measures.aipw_pseudo_outcomes) keeps
the estimate unbiased when treatment probability varies with customer features.
"""
import numpy as np
import pandas as pd

from functions.eval_measures import bootstrap_interval


def policy_profit(treat, effects, customer_value, offer_cost):
    """Estimated incremental profit of contacting the customers where `treat` is True."""
    treat = np.asarray(treat, dtype=bool)
    effects = np.asarray(effects, dtype=float)
    if treat.shape != effects.shape:
        raise ValueError("policy needs one decision per customer")
    return float(np.sum(treat * (customer_value * effects - offer_cost)))


def evaluate_policies(policies, t, effects, customer_value, offer_cost, n_boot=1000, alpha=0.05, seed=0):
    """One row per policy: customers contacted, estimated extra customers retained,
    and estimated profit with a percentile bootstrap interval."""
    t, effects = np.asarray(t), np.asarray(effects, dtype=float)
    rows = []
    for name, treat in policies.items():
        treat = np.asarray(treat, dtype=bool)
        low, high = bootstrap_interval(
            lambda i: policy_profit(treat[i], effects[i], customer_value, offer_cost), t, n_boot, alpha, seed)
        rows.append({
            "policy": name,
            "contacted": int(treat.sum()),
            "extra_retained": float(np.sum(effects[treat])),
            "profit": policy_profit(treat, effects, customer_value, offer_cost),
            "profit_low": low,
            "profit_high": high,
        })
    return pd.DataFrame(rows)
