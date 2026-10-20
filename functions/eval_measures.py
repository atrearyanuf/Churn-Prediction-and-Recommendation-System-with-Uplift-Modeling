import numpy as np
import pandas as pd

def extract_cb(CB):
    if len(CB.shape) == 2:
        (CB_00, CB_01), (CB_10, CB_11) = CB
    else:
        CB_00 = CB[:, 0, 0]
        CB_01 = CB[:, 0, 1]
        CB_10 = CB[:, 1, 0]
        CB_11 = CB[:, 1, 1]
    return CB_00, CB_01, CB_10, CB_11

def uplift_curve(y, t, score, use_churn_convention=True):
    if use_churn_convention:
        return profit_curve(y, t, score, np.array([[1, 1], [0, 0]]))
    else:
        return profit_curve(y, t, score, np.array([[0, 0], [1, 1]]))

def profit_curve(y, t, score, CB=np.array([[1, 1], [0, 0]])):
    """Generalization of the uplift curve to arbitrary cost-benefits.
    """
    score_ranking = np.argsort(score)[::-1]
    y = y[score_ranking]
    t = t[score_ranking]
    N = y.shape[0]
    score = score[score_ranking]
    CB_00, CB_01, CB_10, CB_11 = extract_cb(CB)

    grid = pd.DataFrame(data={
        "k": np.arange(1, N + 1),
        "n_1": np.cumsum(t),
        "n_0": np.cumsum(1 - t),
        "score": score,
        "r_00": np.cumsum((1 - y) * (1 - t) * CB_00),
        "r_01": np.cumsum((1 - y) * t * CB_01),
        "r_10": np.cumsum(y * (1 - t) * CB_10),
        "r_11": np.cumsum(y * t * CB_11)
    })
    # Avoid NaNs, they will be replaced by zeros
    grid.loc[grid["n_0"] == 0, ["n_0"]] = 1
    grid.loc[grid["n_1"] == 0, ["n_1"]] = 1

    grid["profit"] = (
        ((grid.r_01 + grid.r_11) / grid.n_1) -
        ((grid.r_00 + grid.r_10) / grid.n_0)
    ) * grid["k"]

    return grid

def calibrate_score(score, p):

    p_s = np.mean(score)
    ratio = p * (p_s - 1) / (p_s * (p - 1))
    return ratio * score / (ratio * score - score + 1)

def ipw_weights(t, propensity):
    """Inverse-propensity weights: 1/p for treated, 1/(1 - p) for control customers."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(propensity, dtype=float)
    if t.shape != p.shape or not ((p > 0) & (p < 1)).all():
        raise ValueError("propensity must lie strictly between 0 and 1, one per customer")
    return t / p + (1 - t) / (1 - p)

def qini_curve(y, t, score, weights=None):
    """Qini curve (Radcliffe 2007): incremental positive outcomes among the top-k
    customers ranked by `score` (highest first), for k = 0..N.

    q(k) = n_t(k) * (treated outcome rate - control outcome rate) among the top k.
    With `weights` (e.g. ipw_weights) both rates are weighted averages, which
    corrects the comparison when treatment probability varies with customer
    features; weights=None is the standard, unweighted Qini.

    Customers with tied scores form one block and the curve is interpolated
    linearly across the block, so the result never depends on row order (a
    constant score gives exactly the random-targeting line). Points before the
    first control customer are interpolated, as in causalml.metrics.get_qini.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if not (y.ndim == t.ndim == score.ndim == w.ndim == 1 and len(y) == len(t) == len(score) == len(w)):
        raise ValueError("y, t, score and weights must be 1-D arrays of the same length")
    if not np.isin(t, (0, 1)).all():
        raise ValueError("t must be binary (0 = control, 1 = treated)")
    if not (t == 0).any() or not (t == 1).any():
        raise ValueError("Qini needs both treated and control customers")
    if not (np.isfinite(y).all() and np.isfinite(score).all()):
        raise ValueError("y and score must be finite")
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("weights must be positive and finite")

    order = np.argsort(-score, kind="stable")
    y, t, score, w = y[order], t[order], score[order], w[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    w_t = np.cumsum(w * t)
    w_c = np.cumsum(w * (1 - t))
    rate_t = np.divide(np.cumsum(w * y * t), w_t, out=np.zeros(len(y)), where=w_t > 0)

    k = np.arange(1, len(y) + 1)
    block_end = np.r_[score[1:] != score[:-1], True]
    anchor = block_end & (n_c > 0)
    rate_c = np.cumsum(w * y * (1 - t))[anchor] / w_c[anchor]
    q = n_t[anchor] * (rate_t[anchor] - rate_c)
    return np.interp(np.arange(len(y) + 1), np.r_[0, k[anchor]], np.r_[0.0, q])

def qini_coefficient(y, t, score, weights=None):
    """Mean gap between the model's Qini curve and the random-targeting line,
    in incremental positive outcomes. 0 = no better than random.

    Unnormalised on purpose: normalising divides by the total gain, which is
    close to 0 when the average treatment effect is small and blows the score
    up. Unweighted, it equals causalml.metrics.qini_score(..., normalize=False)
    for untied scores.
    """
    q = qini_curve(y, t, score, weights)
    return float((q - np.linspace(0.0, q[-1], len(q))).mean())

def bootstrap_qini(y, t, score, weights=None, n_boot=1000, alpha=0.05, seed=0):
    """Qini coefficient with a percentile bootstrap confidence interval.

    Treated and control customers are resampled separately, so every replicate
    keeps the original arm sizes. Returns (estimate, lower, upper); an interval
    that contains 0 means the model is not distinguishable from random targeting.
    """
    y, t, score = np.asarray(y), np.asarray(t), np.asarray(score)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    estimate = qini_coefficient(y, t, score, w)
    rng = np.random.default_rng(seed)
    treated, control = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    replicates = np.empty(n_boot)
    for b in range(n_boot):
        i = np.r_[rng.choice(treated, len(treated)), rng.choice(control, len(control))]
        replicates[b] = qini_coefficient(y[i], t[i], score[i], w[i])
    lower, upper = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lower), float(upper)
