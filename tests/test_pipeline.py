"""End-to-end check of the uplift pipeline's sign convention on a campaign
where the right answer is known: a learner trained without oversampling must
target persuadables and leave sleeping dogs alone."""
import numpy as np
from causalml.inference.meta import BaseTClassifier
from xgboost import XGBClassifier

from functions.eval_measures import bootstrap_qini
from functions.targeting import recommend, retention_uplift
from tests.conftest import simulate_campaign


def test_t_learner_targets_persuadables_on_known_campaign():
    X_train, t_train, y_train, _ = simulate_campaign(20000, seed=10)
    X_test, t_test, y_test, true_effect = simulate_campaign(5000, seed=11)

    learner = BaseTClassifier(learner=XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0, n_jobs=1))
    learner.fit(X_train, treatment=t_train, y=y_train)
    uplift = retention_uplift(learner.predict(X_test))

    _, lower, _ = bootstrap_qini(1 - y_test, t_test, uplift, n_boot=200)
    assert lower > 0

    recs = recommend(np.full(len(y_test), 0.5), uplift, customer_value=1000, offer_cost=40, validated=lower > 0)
    targeted = recs.loc[recs["recommended_intervention"] == "Retention_Offer", "customer_index"]
    assert len(targeted) > 0
    assert (true_effect[targeted] < 0).mean() > 0.9  # almost all are true persuadables
