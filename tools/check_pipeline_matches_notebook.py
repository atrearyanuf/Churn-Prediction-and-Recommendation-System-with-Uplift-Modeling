"""Fail if `python -m churn_uplift train` did not reproduce the notebook.

Usage: python -m tools.check_pipeline_matches_notebook NOTEBOOK.ipynb artifacts/metrics.json

Both build their models from functions/models.py, so the churn model and its
validation AUC, the selected uplift model and its validation DR Qini interval
must agree to the precision the notebook prints.
"""
import json
import re
import sys

from tools.compare_notebook_outputs import printed_output


def mismatches(notebook, metrics_file):
    text = "".join(printed_output(notebook))
    with open(metrics_file) as f:
        m = json.load(f)
    churn, auc = re.search(r"Best Model: (\S+)\n\s+AUC:\s+(\d\.\d+)", text).groups()
    uplift = re.search(r"Selected by cross-validation: (\S+)", text).group(1)
    num = r"\s*([+-]\d+\.\d+)"
    dr = re.search(rf"^\s+{re.escape(uplift)}\s+Qini=.*DR={num}\s+\[{num},{num}\]", text, re.M).groups()

    found = []
    if m["churn_model"] != churn:
        found.append(f"churn model: notebook {churn}, pipeline {m['churn_model']}")
    elif f"{m['churn_validation_auc'][churn]:.4f}" != auc:
        found.append(f"churn validation AUC: notebook {auc}, pipeline {m['churn_validation_auc'][churn]:.4f}")
    if m["uplift_model"] != uplift:
        found.append(f"uplift model: notebook {uplift}, pipeline {m['uplift_model']}")
    else:
        pipeline_dr = tuple(f"{m['validation_dr_qini'][k]:+.2f}" for k in ("estimate", "low", "high"))
        if pipeline_dr != dr:
            found.append(f"validation DR Qini: notebook {dr}, pipeline {pipeline_dr}")
    return found


if __name__ == "__main__":
    found = mismatches(sys.argv[1], sys.argv[2])
    print("\n".join(found) if found else "Pipeline reproduces the notebook's models and validation results.")
    sys.exit(1 if found else 0)
