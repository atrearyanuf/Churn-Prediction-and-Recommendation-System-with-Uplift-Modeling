"""Train, save and apply the churn and uplift models.

Training follows the notebook step by step, using the same shared definitions
(functions/models.py): churn model chosen by validation AUC and calibrated on
validation, uplift model chosen by cross-validated DR Qini and trained on the
training set, safety gate from the validation DR Qini interval.
"""
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from churn_uplift.drift import build_reference, drift_report
from functions.data import load_dataset
from functions.dataset import Dataset
from functions.eval_measures import aipw_pseudo_outcomes, bootstrap_dr_qini
from functions.models import (
    NEEDS_PROPENSITY, RANDOM_STATE, encode_factors, make_churn_models, make_propensity_model,
    make_uplift_learners, split_dataset,
)
from functions.targeting import recommend, retention_uplift
from functions.uplift_cv import OutcomeModels, UpliftLearner, cross_validate_uplift, summarize_cv

BUNDLE = "model.joblib"
METRICS = "metrics.json"


@dataclass
class Config:
    data_path: str | None   # training data file; None downloads the published dataset
    cache: str              # where the published dataset is cached
    artifacts: str
    offer_cost: float
    customer_value: float
    cv_folds: int
    cv_repeats: int
    psi_threshold: float
    drift_report: str

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            c = tomllib.load(f)
        return cls(
            data_path=c["data"].get("path"), cache=c["data"]["cache"], artifacts=c["artifacts"]["dir"],
            offer_cost=c["business"]["offer_cost"], customer_value=c["business"]["customer_value"],
            cv_folds=c["cv"]["folds"], cv_repeats=c["cv"]["repeats"],
            psi_threshold=c["drift"]["psi_threshold"], drift_report=c["drift"]["report"],
        )


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)


def load_training_data(config):
    return read_table(config.data_path) if config.data_path else load_dataset(config.cache)


def _propensity(model, X):
    return np.clip(model.predict_proba(X)[:, 1], 0.05, 0.95)


def _uplift(name, model, X, propensity):
    extra = {"p": propensity} if name in NEEDS_PROPENSITY else {}
    return retention_uplift(model.predict(X, **extra))


def train(df, config):
    """Fit everything on `df` (feature columns plus y and t), save the bundle and
    metrics under config.artifacts, and return the metrics."""
    encoded, encoders = encode_factors(df)
    features = [c for c in encoded.columns if c not in ("y", "t")]
    parts = split_dataset(encoded[features].to_numpy(dtype=float), encoded["y"].to_numpy(), encoded["t"].to_numpy())
    (X_tr, y_tr, t_tr), (X_va, y_va, t_va) = parts["train"], parts["val"]
    scaler = StandardScaler().fit(X_tr)
    S_tr, S_va = scaler.transform(X_tr), scaler.transform(X_va)
    propensity = make_propensity_model().fit(S_tr, t_tr)
    p_tr, p_va = _propensity(propensity, S_tr), _propensity(propensity, S_va)

    churn_models = make_churn_models((y_tr == 0).sum() / (y_tr == 1).sum())
    churn_auc = {name: float(roc_auc_score(y_va, m.fit(S_tr, y_tr).predict_proba(S_va)[:, 1]))
                 for name, m in churn_models.items()}
    churn_name = max(churn_auc, key=churn_auc.get)
    churn_model = CalibratedClassifierCV(FrozenEstimator(churn_models[churn_name]), method="sigmoid").fit(S_va, y_va)

    learners = {name: UpliftLearner(learner, needs_propensity=name in NEEDS_PROPENSITY)
                for name, learner in make_uplift_learners().items()}
    cv = summarize_cv(cross_validate_uplift(np.vstack([S_tr, S_va]), np.r_[t_tr, t_va], np.r_[y_tr, y_va], learners,
                                            k_folds=config.cv_folds, n_repeats=config.cv_repeats, seed=RANDOM_STATE))
    uplift_name = cv.index[0]
    uplift_model = make_uplift_learners()[uplift_name]
    uplift_model.fit(S_tr, treatment=t_tr, y=y_tr, **({"p": p_tr} if uplift_name in NEEDS_PROPENSITY else {}))
    uplift_va = _uplift(uplift_name, uplift_model, S_va, p_va)

    mu = OutcomeModels().fit(Dataset(X=S_tr, y=y_tr, t=t_tr)).predict(Dataset(X=S_va))
    effects = aipw_pseudo_outcomes(1 - y_va, t_va, mu[:, 0], mu[:, 1], p_va)
    dr_qini, dr_low, dr_high = bootstrap_dr_qini(t_va, uplift_va, effects)
    validated = bool(dr_low > 0)

    reference = build_reference(pd.DataFrame(X_tr, columns=features), {
        "churn_probability": churn_model.predict_proba(S_va)[:, 1], "retention_uplift": uplift_va})
    bundle = {"features": features, "encoders": encoders, "scaler": scaler, "propensity": propensity,
              "churn_name": churn_name, "churn_model": churn_model, "uplift_name": uplift_name,
              "uplift_model": uplift_model, "validated": validated, "reference": reference}
    metrics = {
        "churn_model": churn_name, "churn_validation_auc": churn_auc,
        "uplift_model": uplift_name, "cross_validation": cv.round(4).to_dict("index"),
        "validation_dr_qini": {"estimate": dr_qini, "low": dr_low, "high": dr_high},
        "validated": validated, "training_customers": int(len(y_tr)),
    }
    out = Path(config.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / BUNDLE)
    (out / METRICS).write_text(json.dumps(metrics, indent=2))
    return metrics


def load_bundle(config):
    return joblib.load(Path(config.artifacts) / BUNDLE)


def _prepare(bundle, customers):
    """Encoded feature table in training column order; unseen categories become -1."""
    missing = [c for c in bundle["features"] if c not in customers.columns]
    if missing:
        raise ValueError(f"missing feature columns: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    encoded, _ = encode_factors(customers[bundle["features"]], bundle["encoders"])
    try:
        values = encoded.to_numpy(dtype=float)
    except ValueError:
        raise ValueError("feature columns must be numeric (FACTOR* columns may be text)") from None
    if not np.isfinite(values).all():
        raise ValueError("feature values must be finite")
    return encoded, bundle["scaler"].transform(values)


def _predictions(bundle, scaled):
    churn = bundle["churn_model"].predict_proba(scaled)[:, 1]
    uplift = _uplift(bundle["uplift_name"], bundle["uplift_model"], scaled, _propensity(bundle["propensity"], scaled))
    return churn, uplift


def score(bundle, customers, config):
    """Recommendations for a batch of customers, best expected profit first.
    A `customer_id` column, if present, is carried through."""
    _, scaled = _prepare(bundle, customers)
    churn, uplift = _predictions(bundle, scaled)
    recs = recommend(churn, uplift, config.customer_value, config.offer_cost, validated=bundle["validated"])
    if "customer_id" in customers.columns:
        recs.insert(1, "customer_id", customers["customer_id"].to_numpy()[recs["customer_index"]])
    return recs


def monitor(bundle, customers, config):
    """Drift report for a batch against the training data; True if any column drifted."""
    encoded, scaled = _prepare(bundle, customers)
    churn, uplift = _predictions(bundle, scaled)
    report = drift_report(bundle["reference"], encoded, {"churn_probability": churn, "retention_uplift": uplift},
                          config.psi_threshold)
    return report, bool((report["status"] == "drift").any())
