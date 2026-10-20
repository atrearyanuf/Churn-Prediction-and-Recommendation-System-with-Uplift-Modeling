"""The batch pipeline: train -> score -> monitor, on a small synthetic campaign."""
import json

import numpy as np
import pandas as pd
import pytest

from churn_uplift.__main__ import main
from churn_uplift.pipeline import Config, load_bundle, monitor, score, train
from tests.conftest import synthetic_customers


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    data = synthetic_customers(3000, seed=0)
    data.to_parquet(root / "campaign.parquet", index=False)
    config_file = root / "config.toml"
    config_file.write_text(f"""
[data]
path = "{root / 'campaign.parquet'}"
cache = "{root / 'cache.parquet'}"
[artifacts]
dir = "{root / 'artifacts'}"
[business]
offer_cost = 40
customer_value = 1600
[cv]
folds = 2
repeats = 1
[drift]
psi_threshold = 0.2
report = "{root / 'drift.csv'}"
""")
    config = Config.load(config_file)
    metrics = train(data, config)
    return root, config_file, config, metrics


def new_customers(n=500, seed=7):
    return synthetic_customers(n, seed).drop(columns=["y", "t"]).assign(customer_id=lambda d: [f"C{i}" for i in range(len(d))])


def test_training_saves_model_and_metrics(trained):
    root, _, config, metrics = trained
    bundle = load_bundle(config)

    assert metrics["uplift_model"] in ("T-Learner", "S-Learner", "X-Learner", "DR-Learner")
    assert metrics["churn_model"] in ("RandomForest", "XGBoost", "LightGBM")
    assert isinstance(metrics["validated"], bool) and bundle["validated"] == metrics["validated"]
    assert json.loads((root / "artifacts" / "metrics.json").read_text()) == metrics


def test_scoring_ranks_customers_and_keeps_their_ids(trained):
    _, _, config, _ = trained
    recs = score(load_bundle(config), new_customers(), config)

    assert len(recs) == 500 and recs["expected_profit"].is_monotonic_decreasing
    assert recs["customer_id"].str.startswith("C").all()
    assert recs["churn_probability"].between(0, 1).all()


def test_scoring_respects_the_safety_gate(trained):
    _, _, config, _ = trained
    bundle = {**load_bundle(config), "validated": False}
    assert (score(bundle, new_customers(), config)["recommended_intervention"] == "No_Action").all()


def test_scoring_rejects_missing_feature_columns(trained):
    _, _, config, _ = trained
    with pytest.raises(ValueError, match="missing feature columns: PC3"):
        score(load_bundle(config), new_customers().drop(columns=["PC3"]), config)


def test_unseen_categories_are_scored_and_flagged_as_drift(trained):
    _, _, config, _ = trained
    batch = new_customers()
    batch.loc[:199, "FACTOR2"] = "Z"   # category absent from training
    assert len(score(load_bundle(config), batch, config)) == 500

    report, drifted = monitor(load_bundle(config), batch, config)
    assert drifted and report.set_index("column").loc["FACTOR2", "status"] == "drift"


def test_monitor_finds_no_drift_in_a_similar_batch(trained):
    _, _, config, _ = trained
    report, drifted = monitor(load_bundle(config), new_customers(2000, seed=11), config)
    assert not drifted and (report["status"] != "drift").all()


def test_cli_scores_and_monitors_files(trained, tmp_path):
    root, config_file, _, _ = trained
    batch = new_customers()
    batch.to_csv(tmp_path / "batch.csv", index=False)
    output = tmp_path / "scored.csv"

    assert main(["--config", str(config_file), "score", str(tmp_path / "batch.csv"), "--output", str(output)]) == 0
    assert len(pd.read_csv(output)) == 500

    shifted = batch.assign(PC1=batch["PC1"] + 2)
    shifted.to_csv(tmp_path / "shifted.csv", index=False)
    assert main(["--config", str(config_file), "monitor", str(tmp_path / "shifted.csv")]) == 1
    assert (root / "drift.csv").exists()


def test_cli_reports_bad_input_without_a_traceback(trained, tmp_path, capsys):
    _, config_file, _, _ = trained
    new_customers().drop(columns=["PC1"]).to_csv(tmp_path / "bad.csv", index=False)

    assert main(["--config", str(config_file), "score", str(tmp_path / "bad.csv")]) == 2
    assert "missing feature columns: PC1" in capsys.readouterr().err
