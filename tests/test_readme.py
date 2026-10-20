"""The README's numbers must come from the notebook output, and the files it
names must exist. Guards against the README drifting from the code again."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
# Written by the notebook and git-ignored, so absent on a fresh clone.
GENERATED = ("data/", "outputs/customer_recommendations.csv", "outputs/uplift_comparison_results.csv")


def _notebook_output():
    nb = json.loads((ROOT / "Churn_Uplift_COMPLETE.ipynb").read_text())
    return "".join("".join(o.get("text", "")) for c in nb["cells"]
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")


VALIDATION, TEST = _notebook_output().split("SECTION 9: FINAL TEST SET EVALUATION")
NUM = r"\s*([+-]\d+\.\d+)"
LEARNERS = r"(T-Learner|S-Learner|X-Learner|DR-Learner|UpliftRandomForest)"
QINI = re.compile(rf"^\s+{LEARNERS}\s+Qini={NUM}\s+\[{NUM},{NUM}\]\s+IPW={NUM}\s+\[{NUM},{NUM}\]"
                  rf"\s+DR={NUM}\s+\[{NUM},{NUM}\]", re.M)
CV = re.compile(rf"^\s+{LEARNERS}\s+mean Qini={NUM}\s+mean IPW={NUM}\s+mean DR={NUM}"
                r"\s+\(std\s+(\d+\.\d+)\)\s+DR Qini > 0 in (\d+%) of splits", re.M)


def readme_rows(*cells):
    return [line for line in README.splitlines()
            if line.startswith("|") and all(c in line for c in cells)]


def test_uplift_results_match_notebook():
    validation = QINI.findall(VALIDATION)
    test = QINI.findall(TEST)
    assert len(validation) == 5 and len(test) == 1

    for name, qini, low, high, ipw, ipw_low, ipw_high, dr, dr_low, dr_high in validation:
        assert readme_rows(name, "validation", qini, f"{low} to {high}", ipw, f"{ipw_low} to {ipw_high}",
                           dr, f"{dr_low} to {dr_high}"), name
    name, qini, low, high, ipw, ipw_low, ipw_high, dr, dr_low, dr_high = test[0]
    assert readme_rows(name, "test", qini, f"{low} to {high}", ipw, f"{ipw_low} to {ipw_high}", dr, f"{dr_low} to {dr_high}")
    assert f"test DR Qini {dr} (95% CI {dr_low} to {dr_high})" in README
    assert f"test IPW Qini {ipw} (95% CI {ipw_low} to {ipw_high})" in README
    assert f"test Qini {qini} (95% CI {low} to {high})" in README


def test_cross_validation_results_match_notebook():
    rows = CV.findall(VALIDATION)
    assert len(rows) == 4
    for name, mean_qini, mean_ipw, mean_dr, std_dr, share in rows:
        assert readme_rows(name, mean_qini, mean_ipw, mean_dr, std_dr, share), name
    selected = re.search(r"Selected by cross-validation: (\S+)", VALIDATION).group(1)
    assert rows[0][0] == selected
    assert readme_rows(selected, "selected by cross-validation", "test")

    mean, se = re.search(r"mean DR Qini is ([+-]\d+\.\d+) ± (\d+\.\d+)", VALIDATION).groups()
    assert f"best mean out-of-fold DR Qini is {mean} ({selected}), with a standard error of {se}" in README


def test_power_analysis_matches_notebook():
    rows = re.findall(r"^\s+(whole dataset|test set|top 20% of test set)\s+\(([\d,]+) control / ([\d,]+) treated\): "
                      r"(\d+\.\d+) pp", VALIDATION, re.M)
    assert len(rows) == 3
    for label, n_control, n_treated, mde in rows:
        assert readme_rows(label, f"{n_control} / {n_treated}", f"{mde} pp"), label
    observed, power = re.search(r"Power to detect the observed (\d+\.\d+) pp with the whole dataset: (\d+%)",
                                VALIDATION).groups()
    needed = re.search(r"Customers needed to detect it \(\d+% treated\): ([\d,]+)", VALIDATION).group(1)

    whole_mde = rows[0][3]
    assert f"the smallest churn reduction this data can detect is {whole_mde} pp" in README
    assert f"observed reduction is {observed} pp, which gives {power} power; detecting it would take {needed} customers" in README
    assert f"has {power} power to detect it, and detecting it with 80% power would take {needed} customers" in README


def test_policy_evaluation_matches_notebook():
    rows = re.findall(r"^\s+(Contact no one|Contact everyone|Model picks \([\w-]+\))\s+contacted=\s*([\d,]+)"
                      r"\s+extra retained=\s*([+-]\d+\.\d)\s+profit=(\$[+-][\d,]+)\s+95% CI \[(\$[+-][\d,]+), (\$[+-][\d,]+)\]",
                      TEST, re.M)
    assert len(rows) == 3
    for policy, contacted, retained, profit, low, high in rows:
        assert readme_rows(policy, contacted, retained, profit, f"{low} to {high}"), policy
    for policy, _, _, profit, low, high in rows[1:]:
        readme_profit = profit.replace("$-", "-$").replace("$+", "$")
        readme_low, readme_high = (v.replace("$-", "-$").replace("$+", "+$") for v in (low, high))
        assert f"{readme_profit} (95% CI {readme_low} to {readme_high})" in README, policy


def test_safety_gate_matches_notebook():
    offered, benefit = re.search(r"The model would offer (\d+) customers, estimating a net benefit of (\$[\d,]+)",
                                 VALIDATION).groups()
    blocked = "SAFETY GATE" in VALIDATION

    assert f"would send offers to {offered} of the" in README
    assert f"{offered} of the 2,379 validation customers are worth an offer, for an estimated net benefit of {benefit}" in README
    assert blocked == ("the safety gate recommends **No_Action** for everyone" in README)


def test_churn_results_match_notebook():
    table = re.findall(r"^(RandomForest|XGBoost|LightGBM)" + r"\s+(\d\.\d+)" * 5, VALIDATION, re.M)
    assert len(table) == 3
    for name, *metrics in table:
        readme_name = {"RandomForest": "Random Forest"}.get(name, name)
        assert readme_rows(readme_name, "validation", *(f"{float(m):.3f}" for m in metrics)), name

    test = [f"{float(re.search(rf'{label}:\s+(\d\.\d+)', TEST).group(1)):.3f}"
            for label in ("AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall")]
    assert readme_rows("XGBoost", "test", *test)
    assert f"test AUC {test[0]}, PR-AUC {test[1]}" in README and f"F1 {test[2]}" in README


def test_calibration_results_match_notebook():
    brier = re.search(r"Brier score:\s+raw (\d\.\d+)\s+calibrated (\d\.\d+)", TEST).groups()
    mean = re.search(r"Mean predicted churn: raw (\d\.\d+)\s+calibrated (\d\.\d+)\s+actual (\d\.\d+)", TEST).groups()

    assert readme_rows("Brier score", *brier)
    assert readme_rows("Mean predicted churn", *mean)
    assert f"mean predicted churn on test is {mean[1]}, against an actual {mean[2]}" in README


def test_assumptions_and_diagnostics_match_notebook():
    propensity_auc = re.search(r"Propensity model AUC \(validation\): (\d\.\d+)", VALIDATION).group(1)
    break_even = re.search(r"Break-even retention uplift: (\d\.\d+)", VALIDATION).group(1)

    assert README.count(f"propensity AUC {propensity_auc}") == 2
    assert readme_rows("Break-even retention uplift", break_even)


def test_readme_mentions_only_existing_files():
    inline = re.findall(r"`([\w./-]+(?:\.(?:py|ipynb|txt|ini|go|csv|png)|/))`", README)
    links = re.findall(r"\]\((?!https?:|#|mailto:)([^)]+)\)", README)
    paths = set(inline + links)
    assert paths
    missing = [p for p in paths if not p.startswith(GENERATED) and not (ROOT / p).exists()]
    assert not missing


def test_project_structure_lists_only_existing_files():
    tree = README.split("## Project Structure", 1)[1].split("```")[1]
    names = set(re.findall(r"[\w.-]+\.(?:py|ipynb|txt|ini|go|mod|sum)\b", tree))
    present = set()
    for directory, subdirs, files in os.walk(ROOT):
        subdirs[:] = [d for d in subdirs if not d.startswith(".") and d != "__pycache__"]
        present.update(files)
    assert names
    assert names <= present, names - present
