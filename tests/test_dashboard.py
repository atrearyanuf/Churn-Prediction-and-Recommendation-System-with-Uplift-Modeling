"""Smoke test: the dashboard renders from the CSVs the notebook writes."""
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from functions.targeting import recommend

DASHBOARD = str(Path(__file__).resolve().parents[1] / "streamlit_dashboard.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    # The dashboard caches its CSVs; without this each test would see the
    # previous test's files.
    st.cache_data.clear()


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


def write_outputs(root, validated=True):
    (root / "data").mkdir()
    (root / "outputs").mkdir()
    pd.DataFrame({"y": [0, 1], "t": [1, 0]}).to_parquet(root / "data" / "churn_uplift_anonymized.parquet", index=False)
    recommend([0.2, 0.8], [0.1, -0.05], customer_value=1600, offer_cost=40, validated=validated).to_csv(
        root / "outputs" / "customer_recommendations.csv", index=False)
    pd.DataFrame({"Model": ["T-Learner"], "Qini": [1.0]}).to_csv(
        root / "outputs" / "uplift_comparison_results.csv", index=False)


def test_dashboard_shows_the_latest_drift_report(tmp_path, monkeypatch):
    write_outputs(tmp_path)
    pd.DataFrame({"column": ["PC1", "PC2"], "kind": ["feature", "feature"], "psi": [0.35, 0.01],
                  "status": ["drift", "ok"]}).to_csv(tmp_path / "outputs" / "drift_report.csv", index=False)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert app.dataframe[2].value["column"].tolist() == ["PC1", "PC2"]
    assert any("1 of 2 columns drifted" in c.value for c in app.caption)


def test_dashboard_explains_when_the_safety_gate_blocks_offers(tmp_path, monkeypatch):
    # A customer with positive expected profit but No_Action means the gate is on.
    write_outputs(tmp_path, validated=False)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert "Safety gate on" in app.warning[0].value


def test_dashboard_shows_no_gate_note_when_offers_are_allowed(tmp_path, monkeypatch):
    write_outputs(tmp_path, validated=True)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert len(app.warning) == 0


def test_dashboard_explains_how_to_get_a_drift_report(tmp_path, monkeypatch):
    write_outputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(DASHBOARD).run()

    assert not app.exception
    assert "python -m churn_uplift monitor" in app.info[0].value
