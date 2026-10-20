"""Turn uplift estimates into retention decisions.

The outcome is churn (y = 1 is bad). CausalML learners return the churn effect
tau(x) = P(churn | treated) - P(churn | control), so a customer the campaign
keeps has tau < 0. Everything here works with retention uplift = -tau, where
higher is better: persuadables have positive retention uplift, sleeping dogs
(customers the offer pushes away) have negative retention uplift.
"""
import numpy as np
import pandas as pd


def retention_uplift(churn_effect):
    """Convert a CausalML churn effect (one treatment group) to retention uplift."""
    effect = np.asarray(churn_effect, dtype=float)
    if effect.ndim == 2 and effect.shape[1] == 1:
        effect = effect[:, 0]
    if effect.ndim != 1:
        raise ValueError("expected one treatment effect per customer")
    return -effect


def segment(retention_uplift, threshold=0.02):
    """Label customers by retention uplift: Persuadable, Sleeping_Dog or Unaffected."""
    u = np.asarray(retention_uplift, dtype=float)
    return np.where(u > threshold, "Persuadable",
                    np.where(u < -threshold, "Sleeping_Dog", "Unaffected"))


def recommend(churn_probability, uplift, customer_value, offer_cost, threshold=0.02):
    """Rank customers by the expected profit of a retention offer.

    `uplift` is retention uplift (see retention_uplift). expected_profit =
    customer_value * uplift - offer_cost, and the offer is recommended only when
    that is positive. Rows keep their position in the input as `customer_index`
    and are sorted by expected profit, best first.
    """
    uplift = np.asarray(uplift, dtype=float)
    churn_probability = np.asarray(churn_probability, dtype=float)
    if uplift.ndim != 1 or churn_probability.shape != uplift.shape:
        raise ValueError("churn_probability and uplift must have one value per customer")
    profit = customer_value * uplift - offer_cost
    recs = pd.DataFrame({
        "customer_index": np.arange(len(uplift)),
        "churn_probability": churn_probability,
        "retention_uplift": uplift,
        "expected_profit": profit,
        "segment": segment(uplift, threshold),
        "recommended_intervention": np.where(profit > 0, "Retention_Offer", "No_Action"),
    })
    return recs.sort_values("expected_profit", ascending=False, kind="stable").reset_index(drop=True)
