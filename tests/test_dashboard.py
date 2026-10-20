"""Smoke test: the dashboard renders from the CSVs the notebook writes."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from functions.targeting import recommend

DASHBOARD = str(Path(__file__).resolve().parents[1] / "streamlit_dashboard.py")


def test_dashboard_renders_notebook_outputs(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    pd.DataFrame({"y": [0, 1, 0, 0], "t": [1, 1, 0, 1]}).to_parquet(
        tmp_path / "data" / "churn_uplift_anonymized.parquet", index=False)
    recommend([0.2, 0.8, 0.4, 0.6], [0.10, -0.05, 0.0, 0.03],
              customer_value=1600, offer_cost=40, validated=True).to_csv(
        tmp_path / "outputs" / "customer_recommendations.csv", index=False)
    pd.DataFrame({"Model": ["T-Learner"], "Qini": [1.0]}).to_csv(
        tmp_path / "outputs" / "uplift_comparison_results.csv", index=False)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert app.metric[4].label == "Avg Retention Uplift"
    table = app.dataframe[1].value
    assert table["expected_profit"].is_monotonic_decreasing


def test_dashboard_explains_missing_notebook_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert "Run Churn_Uplift_COMPLETE.ipynb first" in app.error[0].value
