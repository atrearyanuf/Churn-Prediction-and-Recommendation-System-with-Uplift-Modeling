import json

from tests.test_compare_notebook_outputs import write_notebook
from tools.check_pipeline_matches_notebook import mismatches

NOTEBOOK_TEXT = [
    " Best Model: XGBoost\n   AUC:       0.8423\n",
    "  X-Learner          Qini=  -2.38 [ -14.64,   +9.68]  IPW=  +1.82 [ -15.75,  +18.18]  DR=  +7.17 [ -13.93,  +28.34]\n"
    "  Selected by cross-validation: X-Learner (the copy trained on the training set is used from here on)\n",
]


def metrics(**changes):
    m = {"churn_model": "XGBoost", "churn_validation_auc": {"XGBoost": 0.84225},
         "uplift_model": "X-Learner", "validation_dr_qini": {"estimate": 7.171, "low": -13.934, "high": 28.339}}
    m.update(changes)
    return m


def write(tmp_path, m):
    nb = write_notebook(tmp_path / "nb.ipynb", NOTEBOOK_TEXT)
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    return nb, tmp_path / "metrics.json"


def test_matching_pipeline_passes(tmp_path):
    assert mismatches(*write(tmp_path, metrics())) == []


def test_different_model_choice_or_numbers_are_reported(tmp_path):
    found = mismatches(*write(tmp_path, metrics(uplift_model="T-Learner", churn_validation_auc={"XGBoost": 0.80})))
    assert len(found) == 2
    assert any("uplift model" in f for f in found) and any("AUC" in f for f in found)
