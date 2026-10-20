"""Data drift between the training data and a new batch, measured with the
population stability index (PSI). Rule of thumb: below 0.1 stable, 0.1 to 0.2
worth watching, above 0.2 a real shift."""
import numpy as np
import pandas as pd

EPS = 1e-4  # stands in for empty bins, so PSI stays finite
WATCH = 0.1


def psi(expected, actual):
    """PSI between two vectors of bin shares."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_shares(values, edges):
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / counts.sum()


def _categorical_shares(values, categories):
    # The extra last bin holds categories absent from the training data.
    index = {c: i for i, c in enumerate(categories)}
    counts = np.bincount([index.get(v, len(categories)) for v in values], minlength=len(categories) + 1)
    return counts / counts.sum()


def _profile(values, categorical, kind):
    values = np.asarray(values)
    if categorical:
        categories = np.unique(values)
        return {"kind": kind, "categories": categories, "shares": _categorical_shares(values, categories)}
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))[1:-1])   # deciles
    return {"kind": kind, "edges": edges, "shares": _numeric_shares(values, edges)}


def build_reference(features, predictions):
    """Bin shares of every training feature (FACTOR* columns as categories,
    others in deciles) and of every prediction column."""
    reference = {c: _profile(features[c], c.startswith("FACTOR"), "feature") for c in features.columns}
    reference.update({name: _profile(values, False, "prediction") for name, values in predictions.items()})
    return reference


def drift_report(reference, features, predictions, threshold):
    """PSI per column for a new batch, largest first, with status ok / watch / drift."""
    columns = {**{c: features[c] for c in features.columns}, **predictions}
    rows = []
    for name, profile in reference.items():
        values = np.asarray(columns[name])
        shares = (_categorical_shares(values, profile["categories"]) if "categories" in profile
                  else _numeric_shares(values, profile["edges"]))
        value = psi(profile["shares"], shares)
        rows.append({"column": name, "kind": profile["kind"], "psi": value,
                     "status": "drift" if value >= threshold else "watch" if value >= WATCH else "ok"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
