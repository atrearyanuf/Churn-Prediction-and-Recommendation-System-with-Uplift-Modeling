import numpy as np
import pytest

from functions.targeting import recommend, retention_uplift, segment


def test_retention_uplift_is_reduction_in_churn():
    # Offer lowers churn by 10 points -> retention uplift +0.10.
    np.testing.assert_allclose(retention_uplift([-0.10, 0.05]), [0.10, -0.05])


def test_retention_uplift_accepts_causalml_column_output():
    np.testing.assert_allclose(retention_uplift(np.array([[-0.1], [0.2]])), [0.1, -0.2])


def test_retention_uplift_rejects_multiple_treatment_groups():
    with pytest.raises(ValueError, match="one treatment effect"):
        retention_uplift(np.zeros((3, 2)))


def test_segments_follow_retention_sign():
    labels = segment(retention_uplift([-0.10, 0.10, 0.01, -0.01]), threshold=0.02)

    assert labels.tolist() == ["Persuadable", "Sleeping_Dog", "Unaffected", "Unaffected"]


def test_offer_only_when_expected_profit_is_positive():
    recs = recommend(
        churn_probability=[0.3, 0.3, 0.9, 0.9],
        uplift=[0.10, 0.02, -0.10, 0.0],
        customer_value=1000, offer_cost=40,
    ).set_index("customer_index")

    np.testing.assert_allclose(recs["expected_profit"].sort_index(), [60, -20, -140, -40])
    assert recs["recommended_intervention"].sort_index().tolist() == [
        "Retention_Offer", "No_Action", "No_Action", "No_Action"]


def test_high_churn_risk_alone_never_earns_an_offer():
    # The old priority score rewarded churn probability; a sleeping dog with
    # 90% churn risk must still not be targeted.
    recs = recommend([0.9], [-0.2], customer_value=1000, offer_cost=40)

    assert recs.loc[0, "recommended_intervention"] == "No_Action"
    assert recs.loc[0, "segment"] == "Sleeping_Dog"


def test_recommendations_are_ranked_and_traceable():
    recs = recommend([0.1, 0.2, 0.3], [-0.05, 0.30, 0.10], customer_value=500, offer_cost=10)

    assert recs["customer_index"].tolist() == [1, 2, 0]
    assert recs["expected_profit"].is_monotonic_decreasing
    np.testing.assert_allclose(recs["churn_probability"], [0.2, 0.3, 0.1])


def test_recommend_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one value per customer"):
        recommend([0.1, 0.2], [0.1], customer_value=100, offer_cost=1)
    with pytest.raises(ValueError, match="one value per customer"):
        recommend([0.1], [[0.1]], customer_value=100, offer_cost=1)
