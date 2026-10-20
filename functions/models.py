"""Data preparation and model definitions shared by the notebook and the
pipeline (churn_uplift/), so both train exactly the same models.

No XGBoost model uses colsample_bytree: its column sampling uses a C++
standard-library shuffle that gives different results on macOS and Linux.
"""
from causalml.inference.meta import BaseDRClassifier, BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42


def encode_factors(df, encoders=None):
    """Label-encode the FACTOR* columns (sorted category order).

    With `encoders` from an earlier call, categories not seen then become -1.
    Returns the encoded copy and the encoders.
    """
    df = df.copy()
    columns = [c for c in df.columns if c.startswith("FACTOR")]
    if encoders is None:
        encoders = {c: LabelEncoder().fit(df[c].astype(str)) for c in columns}
    for c in columns:
        codes = {label: i for i, label in enumerate(encoders[c].classes_)}
        df[c] = df[c].astype(str).map(codes).fillna(-1).astype(int)
    return df, encoders


def split_dataset(X, y, t, random_state=RANDOM_STATE):
    """60/20/20 train/validation/test split, stratified on churn x treatment so
    every part keeps the rare treated churners. Returns {part: (X, y, t)}."""
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=random_state, stratify=y * 2 + t)
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.25, random_state=random_state, stratify=y_temp * 2 + t_temp)
    return {"train": (X_train, y_train, t_train), "val": (X_val, y_val, t_val), "test": (X_test, y_test, t_test)}


def make_propensity_model(random_state=RANDOM_STATE):
    """P(treated | X) on standardised features."""
    return LogisticRegression(max_iter=1000, C=1.0, random_state=random_state, n_jobs=-1)


def make_churn_models(scale_pos_weight, random_state=RANDOM_STATE):
    """Churn classifiers, each weighted for the class imbalance."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20, class_weight="balanced",
            random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, objective="binary:logistic",
            eval_metric="auc", scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            objective="binary", class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1),
    }


def _xgb_params(random_state):
    return dict(n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=random_state, n_jobs=-1)


def _regularised_xgb(cls, random_state):
    # Shallow trees with large leaves: the control arm has only ~60 churners.
    return cls(max_depth=3, min_child_weight=10, **_xgb_params(random_state))


def make_uplift_learners(random_state=RANDOM_STATE):
    """Uplift learners, each predicting the churn effect P(churn|T) - P(churn|C).

    X- and DR-Learner take propensity scores (p) at fit time; the DR-Learner
    cross-fits its outcome and effect models over 3 folds. (CausalML's causal
    forest was tried and dropped: one fit on 4,000 customers took 95-400 s.)
    """
    params = _xgb_params(random_state)
    return {
        "T-Learner": BaseTClassifier(learner=XGBClassifier(max_depth=6, **params)),
        "S-Learner": BaseSClassifier(learner=XGBClassifier(max_depth=8, min_child_weight=5, **params)),
        "X-Learner": BaseXClassifier(
            outcome_learner=XGBClassifier(max_depth=6, **params),
            effect_learner=XGBRegressor(max_depth=4, **params)),
        "DR-Learner": BaseDRClassifier(
            learner=_regularised_xgb(XGBClassifier, random_state),
            treatment_effect_learner=XGBRegressor(max_depth=3, min_child_weight=20, **params)),
    }


NEEDS_PROPENSITY = {"X-Learner", "DR-Learner"}


def make_outcome_model(random_state=RANDOM_STATE):
    """Churn model fitted per arm to build doubly-robust (AIPW) effect estimates."""
    return _regularised_xgb(XGBClassifier, random_state)


def make_uplift_forest(random_state=RANDOM_STATE):
    """CausalML's native uplift random forest; treatment labels must be '0'/'1'."""
    return UpliftRandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50, evaluationFunction="KL", control_name="0",
        random_state=random_state)
