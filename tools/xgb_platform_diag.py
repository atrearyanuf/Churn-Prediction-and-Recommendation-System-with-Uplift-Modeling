"""TEMPORARY diagnostic: which XGBoost settings give platform-independent results?"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import platform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from functions.data import load_dataset

df = load_dataset("data/churn_uplift_anonymized.parquet")
for c in [c for c in df if c.startswith("FACTOR")]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
feat = [c for c in df if c not in ("t", "y")]
X, y, t = df[feat].values, df.y.values, df.t.values
Xtmp, Xte, ytmp, yte, ttmp, tte = train_test_split(X, y, t, test_size=.2, random_state=42, stratify=y * 2 + t)
Xtr, Xva, ytr, yva, ttr, tva = train_test_split(Xtmp, ytmp, ttmp, test_size=.25, random_state=42, stratify=ytmp * 2 + ttmp)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xva = sc.transform(Xva)
spw = (ytr == 0).sum() / (ytr == 1).sum()
print(platform.machine(), platform.system())
for name, kw in {
    "notebook (hist, subsample .8, colsample .8)": dict(subsample=0.8, colsample_bytree=0.8),
    "hist, no row/col sampling": dict(subsample=1.0, colsample_bytree=1.0),
    "hist, subsample .8 only": dict(subsample=0.8, colsample_bytree=1.0),
    "hist, colsample .8 only": dict(subsample=1.0, colsample_bytree=0.8),
    "exact, subsample .8, colsample .8": dict(subsample=0.8, colsample_bytree=0.8, tree_method="exact"),
    "exact, no sampling": dict(subsample=1.0, colsample_bytree=1.0, tree_method="exact"),
}.items():
    m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective="binary:logistic",
                      scale_pos_weight=spw, random_state=42, n_jobs=1, **kw).fit(Xtr, ytr)
    print(f"{name:<45} val AUC {roc_auc_score(yva, m.predict_proba(Xva)[:, 1]):.6f}")
