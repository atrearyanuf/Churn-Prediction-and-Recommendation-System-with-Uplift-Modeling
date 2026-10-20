#  Churn Prediction & Recommendation System with Uplift Modeling

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![tests](https://github.com/atrearyanuf/Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling/actions/workflows/tests.yml/badge.svg)](https://github.com/atrearyanuf/Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling/actions/workflows/tests.yml)

**A causal-inference pipeline that moves customer retention from "who will churn?" to "who will respond to an offer?", and says plainly when the data cannot answer the second question.**

[Overview](#overview) • [Results](#results) • [Installation](#installation) • [Usage](#usage) • [Testing](#testing) • [Limitations](#limitations)

</div>

---

## Overview

Churn models rank customers by **who will churn**. A retention budget needs to know **who will stay because of the offer**. Contacting a customer who would have stayed anyway wastes the offer, and some customers are pushed to leave by being contacted ("sleeping dogs").

This project builds both stages on a real retention campaign:

1. **Churn prediction**: Random Forest, XGBoost and LightGBM estimate P(churn).
2. **Uplift modeling**: T-, S- and X-Learners (CausalML, XGBoost base learners) and an Uplift Random Forest estimate how much the offer changes each customer's churn probability.
3. **Recommendation**: an offer is recommended only when its expected value exceeds its cost.

### Academic Context

- **Course:** CAI6826 - Artificial Intelligence Systems
- **Institution:** University of Florida
- **Author:** Aryan Atre
- **Year:** Spring 2026

### Key Results

All numbers below are printed by `Churn_Uplift_COMPLETE.ipynb`. `tests/test_readme.py` fails if they drift apart, and a weekly CI job re-runs the notebook on Linux to check that it still prints them.

- **Churn prediction:** XGBoost reaches test AUC 0.802, PR-AUC 0.197 (random ranking: 0.035) and F1 0.249 on a 28:1 class imbalance. After calibration its mean predicted churn on test is 0.0333, against an actual 0.0345.
- **Uplift modeling:** no learner reliably beats random targeting. In repeated cross-validation the best mean out-of-fold DR Qini is +0.70 (X-Learner), with a standard error of 2.10. The selected X-Learner scores test DR Qini -1.58 (95% CI -16.47 to +18.48), test IPW Qini -4.02 (95% CI -13.94 to +8.62) and test Qini -2.45 (95% CI -11.61 to +7.71).
- **Power:** with all 11,896 customers, the smallest churn reduction this data can detect is 1.04 pp. The offer's observed reduction is 0.28 pp, which gives 11% power; detecting it would take 190,285 customers. So the null result is what the data can show, not a modeling failure.
- **Policy value (test set, doubly robust):** contacting everyone earns -$60,585 (95% CI -$149,232 to +$59,750), the X-Learner's picks -$36,994 (95% CI -$100,607 to +$57,976), and contacting no one $0.
- **Recommendation:** the X-Learner's estimates would send offers to 778 of the 2,379 validation customers. Because the model fails validation, a safety gate recommends **no offers**: spending the budget on an unvalidated model is not justified.
- **Why:** the offer lowers churn by only 0.25 percentage points on average (3.64% control vs 3.39% treated in validation), the training set has 244 churners (63 in the control arm), and treatment probability depends on customer features (propensity AUC 0.631).

---

## How It Works

```
 Dataset (Orange Belgium, 11,896 customers, 178 anonymised features)
    │  label-encode categoricals, 60/20/20 split stratified on (churn × treatment)
    ▼
 ┌──────────────────────────┐        ┌──────────────────────────────────┐
 │ Stage 1: churn models    │        │ Stage 2: uplift models           │
 │  Random Forest, XGBoost, │        │  T / S / X / DR-Learner,         │
 │  LightGBM → P(churn)     │        │  Uplift Random Forest            │
 │  threshold tuned for F1, │        │  → churn effect τ(x)             │
 │  Platt-calibrated        │        │                                  │
 └────────────┬─────────────┘        └────────────────┬─────────────────┘
              │                                       │ retention uplift = −τ(x)
              ▼                                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Evaluation: Qini, IPW and doubly-robust (DR) Qini, 95% bootstrap CI    │
 │ Selection:  repeated cross-validation by DR Qini (5 folds × 2 repeats) │
 │ Targeting:  expected profit = value × retention uplift − offer cost    │
 │             offer only if profit > 0 and the model beats random        │
 └────────────┬───────────────────────────────────────────────────────────┘
              ▼
   outputs/ (CSV + plots)  →  Streamlit dashboard,  SHAP explanations
```

### Sign convention

The outcome is churn (`y = 1` is bad). CausalML learners return the **churn effect** τ(x) = P(churn | offer) − P(churn | no offer), so a customer the offer keeps has τ(x) < 0. The pipeline converts this once, right at the model output, to **retention uplift** = −τ(x), where higher is better:

| Retention uplift | Segment | Meaning |
|------------------|---------|---------|
| > +0.02 | Persuadable | Offer lowers their churn |
| < −0.02 | Sleeping_Dog | Offer raises their churn: never target |
| otherwise | Unaffected | Offer makes little difference |

### Evaluation choices

- **Qini is unnormalised.** It is measured in extra customers retained compared with random targeting, averaged over all targeting depths. CausalML's default normalisation divides by the total gain. That gain is close to 0 here, which inflated earlier scores to values like −1.90.
- **Ties are handled.** Customers with equal scores form one block, so a constant score gives exactly 0, independent of row order.
- **Confidence intervals** come from 1,000 bootstrap resamples within each arm. An interval containing 0 means "not distinguishable from random targeting".
- **No oversampling for uplift models.** Oversampling each arm to 15% churners inflates churn odds by a different factor per arm, which biases the estimated effect.
- **Propensity weighting (IPW Qini).** Treatment probability varies with customer features, so the plain Qini compares treated and control customers who differ in more than the offer. Weighting each customer by 1/p if treated, or 1/(1 − p) if not, corrects this. On a simulated confounded campaign, the plain Qini is off by −54% on average and the weighted one by −0.3%.
- **Doubly-robust (DR) Qini.** Per-arm outcome models, trained on other customers, predict each customer's retention with and without the offer; the AIPW formula combines them with the propensity score. The resulting effect estimates are unbiased if either the outcome models or the propensity model are right (both cases are tested), and they replace treated-vs-control rate differences in the Qini.
- **DR-Learner.** CausalML's doubly-robust learner, cross-fitted over 3 folds with a fixed seed and shallow, large-leaf trees for the small control arm. CausalML's causal forest was tried and dropped: one fit on 4,000 customers took 95–400 s.
- **Model selection by repeated cross-validation.** A single validation split cannot separate the models (CI widths around ±12). The pipeline selects the model with the highest mean out-of-fold DR Qini over 5 folds × 2 repeats of train + validation. The copy trained on the training set is then evaluated once on test.
- **Safety gate.** Offers are recommended only if the selected model's validation DR Qini CI lies above 0. Otherwise every customer gets No_Action, and the model's estimates are kept in the output for analysis.
- **Platform-independent XGBoost.** XGBoost's column subsampling (`colsample_bytree`) uses a C++ standard-library shuffle that gives different results on macOS and Linux, so it is not used; row subsampling is reproducible. The weekly notebook job runs on Linux and compares its output with results produced on macOS.
- **Calibrated churn probabilities.** Class weighting inflates predicted churn. Platt scaling, fitted on validation, maps the predictions back to churn rates. It is monotone, so AUC and threshold decisions are unchanged. The recommendations and the dashboard show calibrated probabilities.

---

## Results

### Churn prediction

The decision threshold is chosen for F1 on the validation set.

| Model | Split | AUC | PR-AUC | F1 | Precision | Recall |
|-------|-------|-----|--------|----|-----------|--------|
| Random Forest | validation | 0.822 | 0.196 | 0.250 | 0.197 | 0.341 |
| LightGBM | validation | 0.825 | 0.176 | 0.294 | 0.344 | 0.256 |
| **XGBoost** (selected by AUC) | validation | 0.842 | 0.194 | 0.295 | 0.222 | 0.439 |
| **XGBoost** | **test** | **0.802** | **0.197** | **0.249** | **0.186** | **0.378** |

PR-AUC of a random ranking equals the churn rate (0.035). XGBoost's validation AUC of 0.842 falls to 0.802 on test, so selecting on a single validation split flatters it.

| Calibration on test | Raw | Calibrated | Actual |
|---------------------|-----|------------|--------|
| Mean predicted churn | 0.0479 | 0.0333 | 0.0345 |
| Brier score | 0.0320 | 0.0309 | |

### Uplift modeling

Qini is reported on retention, in extra customers retained compared with random targeting, with a 95% bootstrap CI.

**Repeated cross-validation** (out-of-fold, 10 splits of train + validation), used for model selection:

| Model | Mean Qini | Mean IPW Qini | Mean DR Qini | Std DR Qini | Splits with DR Qini > 0 |
|-------|-----------|---------------|--------------|-------------|-------------------------|
| X-Learner | -0.95 | +0.46 | +0.70 | 6.63 | 50% |
| S-Learner | -0.17 | +0.03 | +0.19 | 2.51 | 20% |
| DR-Learner | -1.33 | +0.20 | -1.04 | 5.07 | 30% |
| T-Learner | -1.51 | -1.87 | -3.81 | 6.03 | 30% |

**Validation and test** (95% bootstrap CIs):

| Model | Split | Qini | 95% CI | IPW Qini | 95% CI | DR Qini | 95% CI |
|-------|-------|------|--------|----------|--------|---------|--------|
| T-Learner | validation | -6.47 | -17.19 to +5.98 | -10.09 | -30.11 to +8.15 | -9.97 | -31.04 to +8.89 |
| S-Learner | validation | +0.00 | +0.00 to +0.00 | +0.00 | +0.00 to +0.00 | +0.00 | +0.00 to +0.00 |
| X-Learner | validation | -2.38 | -14.64 to +9.68 | +1.82 | -15.75 to +18.18 | +7.17 | -13.93 to +28.34 |
| DR-Learner | validation | -3.69 | -13.73 to +6.99 | +1.53 | -11.92 to +16.60 | +0.74 | -14.39 to +17.63 |
| UpliftRandomForest | validation | -4.10 | -14.30 to +6.74 | -10.91 | -31.16 to +7.59 | -14.30 | -35.20 to +3.36 |
| **X-Learner** (selected by cross-validation) | **test** | **-2.45** | **-11.61 to +7.71** | **-4.02** | **-13.94 to +8.62** | **-1.58** | **-16.47 to +18.48** |

No learner reliably beats random targeting. The selected X-Learner's cross-validated mean DR Qini (+0.70) is well within one standard error (2.10) of 0, and every test interval contains 0. The S-Learner's single model never splits on the treatment column, so its scores are constant and its Qini is exactly 0. The DR intervals are wide because the outcome models are weak: the control arm has only 63 churners.

### Power analysis

Smallest churn reduction a two-sided test detects with 80% power at alpha 0.05 (control churn rate 3.64%):

| Customers | Control / treated | Detectable reduction |
|-----------|-------------------|----------------------|
| whole dataset | 2,886 / 9,010 | 1.04 pp |
| test set | 577 / 1,803 | 2.09 pp |
| top 20% of test set | 115 / 360 | 3.46 pp |

The offer's observed churn reduction is 0.28 pp: the whole dataset has 11% power to detect it, and detecting it with 80% power would take 190,285 customers. The detectable effect on the test set, where models are evaluated, is about 7 times the observed one.

### Policy evaluation

Estimated profit on the 2,380 test customers compared with contacting no one, using the doubly-robust per-customer effects:

| Policy | Contacted | Extra customers retained | Profit | 95% CI |
|--------|-----------|--------------------------|--------|--------|
| Contact no one | 0 | +0.0 | $+0 | $+0 to $+0 |
| Contact everyone | 2,380 | +21.6 | $-60,585 | $-149,232 to $+59,750 |
| Model picks (X-Learner) | 767 | -3.9 | $-36,994 | $-100,607 to $+57,976 |

Contacting no one has the best point estimate, which is also the decision the safety gate makes. Every interval contains 0, so the data cannot rank the policies with confidence.

### Business decision

| Assumption | Value |
|------------|-------|
| Offer cost | $40 per contacted customer |
| Value of a retained customer | $1,600 |
| Break-even retention uplift | 0.025 |

By the X-Learner's estimates, 778 of the 2,379 validation customers are worth an offer, for an estimated net benefit of $61,425. The model does not beat random targeting on validation (its DR Qini CI contains 0), so the safety gate recommends **No_Action** for everyone. The estimates stay in `outputs/customer_recommendations.csv` for analysis, but no budget is spent on them. The dollar values are planning assumptions, not data from the campaign. Change `OFFER_COST` and `CUSTOMER_VALUE` in section 8 of the notebook to use your own.

> An earlier version of this README reported a Qini of 0.33, 58 targeted customers and a 1,204% ROI. Those figures came from a normalised Qini, a targeting rule with the effect sign inverted, and oversampled training data. They are superseded by the results above.

---

## Installation

### Prerequisites

- Python 3.12

### Steps

```bash
git clone https://github.com/atrearyanuf/Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling.git
cd Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling

# venv (or: conda create -n churn-uplift python=3.12 && conda activate churn-uplift)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt   # runtime dependencies + pytest
```

`requirements.txt` pins the exact versions the results were produced with. A fresh environment built from it reproduces the notebook output.

---

## Usage

### 1. Run the pipeline notebook

```bash
jupyter notebook Churn_Uplift_COMPLETE.ipynb
# or, headless:
jupyter nbconvert --to notebook --execute --inplace Churn_Uplift_COMPLETE.ipynb
```

The first cell downloads the dataset from [OpenML (dataset 45580)](https://www.openml.org/d/45580), falling back to the original Dropbox copy, and caches it in `data/churn_uplift_anonymized.parquet`. Every copy, downloaded or cached, is checked against a fingerprint of the data these results were produced with, so a changed source fails loudly instead of silently changing the numbers. The full run takes about two minutes on a laptop. It writes:

- `outputs/customer_recommendations.csv`: one row per validation customer with churn probability, retention uplift, expected profit, segment and recommendation
- `outputs/uplift_comparison_results.csv`: Qini with confidence interval for each uplift model
- plots in `outputs/` (ROC, confusion matrix, Qini curves, model comparison, SHAP)

### 2. Launch the dashboard

From the repository root, after running the notebook:

```bash
streamlit run streamlit_dashboard.py
```

It shows the KPIs, the churn-probability distribution, the recommendation counts, the uplift model comparison, a filterable recommendation table and the latest drift report from `python -m churn_uplift monitor`. If the notebook outputs are missing, it says so instead of crashing.

### 3. Go chart server (optional)

```bash
go run .        # needs Go 1.23+; serves http://localhost:8081
```

It shows a bar chart of the 20 customers with the highest retention uplift, read from `outputs/customer_recommendations.csv` on every request.

### 4. Batch pipeline: train, score, monitor

The `churn_uplift` package runs the notebook's modeling steps as a command-line pipeline. It builds every model from the same definitions (`functions/models.py`), so its models match the notebook's; CI checks this on every change. Settings (data, artifact paths, offer cost, customer value, cross-validation and drift threshold) are in `config.toml`.

```bash
python -m churn_uplift train                      # fit and save models to artifacts/ (about 1-2 minutes)
python -m churn_uplift score customers.csv        # recommendations -> outputs/scored_customers.csv
python -m churn_uplift monitor customers.csv      # drift report -> outputs/drift_report.csv
```

- **train** chooses the churn model by validation AUC, calibrates it, chooses the uplift model by cross-validated DR Qini, applies the safety gate, and saves `artifacts/model.joblib` plus `artifacts/metrics.json`.
- **score** reads a CSV or parquet file with the training feature columns (an optional `customer_id` column is carried through). Missing columns are rejected with a clear error, and categories not seen in training are scored as unknown.
- **monitor** compares a new batch with the training data using the population stability index (PSI) for every feature and for the predicted churn probability and retention uplift. It exits with code 1 if any PSI reaches the threshold (0.2), so a scheduler or CI job can alert on it.

### 5. Docker

```bash
docker build -t churn-uplift .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" churn-uplift train
docker run --rm -v "$PWD/artifacts:/app/artifacts" -v "$PWD/outputs:/app/outputs" -v "$PWD/batch:/app/batch" \
  churn-uplift score batch/customers.csv
```

The image runs the pipeline as a non-root user. CI builds it and runs train, score and monitor inside it on every pipeline change.

### 6. Use the targeting functions directly

```python
from functions.eval_measures import bootstrap_qini, ipw_weights
from functions.targeting import recommend, retention_uplift

uplift = retention_uplift(learner.predict(X))           # CausalML churn effect -> retention uplift
qini, low, high = bootstrap_qini(1 - y, t, uplift)       # evaluate on retention (1 - churn)
ipw_qini, low, high = bootstrap_qini(1 - y, t, uplift, ipw_weights(t, propensity))
recs = recommend(churn_probability, uplift, customer_value=1600, offer_cost=40)
```

---

## Testing

```bash
pytest          # Python suite
go test ./...   # Go chart server
```

Both run on every push via GitHub Actions (`.github/workflows/tests.yml`).

A second workflow (`.github/workflows/notebook.yml`) re-runs the whole notebook from scratch, including the dataset download. It runs every Monday and whenever the notebook, `functions/` or `requirements.txt` change. `tools/compare_notebook_outputs.py` then fails the run if any printed result differs from the committed notebook, which catches code changes committed without re-running the notebook, and data sources that disappear or change.

A third workflow (`.github/workflows/pipeline.yml`) trains the batch pipeline and fails unless `tools/check_pipeline_matches_notebook.py` finds the same churn model and validation AUC, the same uplift model and the same validation DR Qini interval as the committed notebook. It also builds the Docker image and runs the pipeline inside it.

The Python suite (`tests/`) checks:

- **Qini implementation:** it matches `causalml.metrics.get_qini` and `qini_score(normalize=False)`, ignores row order for tied scores, gives exactly 0 for a constant score and rejects invalid input.
- **Bootstrap intervals:** they are reproducible, contain 0 for a useless model and exclude 0 for a real signal.
- **DR Qini:** unbiased when either the outcome models or the propensity model are right, and less noisy than IPW with good outcome models; with the true effects it equals the true Qini.
- **Power analysis:** a two-proportion z-test detects the computed minimum effect in 81.6% of 20,000 simulated campaigns (target 80%), and sample size, power and MDE agree with each other.
- **Policy evaluation:** contacting no one earns 0, and the doubly-robust profit estimate is unbiased on a simulated confounded campaign.
- **Shared models:** encoding, the 60/20/20 split and factories match the notebook; no XGBoost model uses column sampling; DR-Learner cross-fitting is reproducible.
- **IPW Qini:** unit weights reproduce the standard Qini. On a simulated confounded campaign, the weighted Qini is unbiased, while the unweighted one is off by about half.
- **Cross-validation:** every learner is scored on every split, and only on out-of-fold customers (a learner that memorises its training data scores exactly 0). A real learner ranks above noise.
- **Targeting:** the retention sign convention holds, offers go out only when profitable, sleeping dogs are never targeted, even at high churn risk, and nothing is offered when the model is not validated.
- **End to end:** on a synthetic campaign with a known effect, a T-Learner trained without oversampling targets true persuadables more than 90% of the time.
- **Dataset loading:** sources are tried in order, a source with different content is skipped, a modified cache is rejected, and the cache keeps floats exact.
- **Dataset-paper helpers:** cross-validation splits cover every customer, EasyEnsemble folds are balanced and independent (including in parallel), and the wrappers handle 0/1 treatment flags.
- **Dashboard:** it renders from notebook outputs and explains missing files.
- **Batch pipeline:** training saves the model and metrics; scoring ranks customers, keeps their IDs, respects the safety gate, rejects missing columns and handles unseen categories; the CLI returns exit code 1 on drift and 2 on bad input without a traceback.
- **Drift monitoring:** PSI matches the textbook formula and survives empty bins; a similar batch shows no drift; a shifted feature or an unseen category is flagged.
- **Pipeline check:** a different model choice or validation number is reported.
- **Notebook comparison tool:** it ignores timings, stderr, widgets and last-digit float noise (relative 1e-6), and reports changed results by cell.
- **README:** every number above matches the notebook output, and every file path it mentions exists.

The Go tests check the CSV parsing (columns found by name, bad input rejected) and the HTTP handler.

---

## Project Structure

```
├── Churn_Uplift_COMPLETE.ipynb   # End-to-end pipeline (data → models → evaluation → recommendations → SHAP)
├── streamlit_dashboard.py        # Dashboard over the notebook's outputs
├── functions/
│   ├── eval_measures.py          # Qini, IPW and DR Qini + bootstrap CI (used); profit curves (from the dataset paper)
│   ├── targeting.py              # Retention-uplift convention, segments, expected-profit recommendations
│   ├── uplift_cv.py              # Repeated cross-validation of uplift learners (out-of-fold Qini, IPW Qini)
│   ├── data.py                   # Dataset download (OpenML, Dropbox fallback) with fingerprint check
│   ├── models.py                 # Shared encoding, split and model definitions (notebook + pipeline)
│   ├── power.py                  # Power analysis for the offer's effect on churn
│   ├── policy.py                 # Doubly-robust profit of targeting policies
│   └── benchmark.py, dataset.py, easy_ensemble.py, wrappers.py
│                                 # Benchmark helpers from the dataset paper's code (bugs fixed); used by uplift_cv.py
├── tests/                        # pytest suite
├── churn_uplift/                 # Batch pipeline: python -m churn_uplift {train,score,monitor}
│   ├── __main__.py               # Command line
│   ├── pipeline.py               # Train, save, score
│   └── drift.py                  # Population stability index (PSI) drift report
├── config.toml                   # Pipeline settings
├── Dockerfile                    # Pipeline image (non-root)
├── tools/
│   ├── compare_notebook_outputs.py        # CI: a fresh notebook run prints the committed results
│   └── check_pipeline_matches_notebook.py # CI: the pipeline reproduces the notebook's models
├── outputs/                      # Plots (tracked) and CSVs (generated, git-ignored)
├── end_user.go, go.mod, go.sum   # Go chart server: top customers by retention uplift
├── end_user_test.go              # Go tests
├── requirements.txt              # Pinned runtime dependencies
├── requirements-dev.txt          # + pytest
├── pytest.ini
├── docs/                         # Course reports, slides and architecture diagrams (PDF, PPTX, HTML)
└── LICENSE
```

---

## Limitations

- **Treatment is only partly random.** According to the dataset paper, the control group was drawn at random from customers already selected as high churn risk. Within this data, treatment probability still varies with customer features (propensity AUC 0.631), so uplift estimates rely on there being no unmeasured confounding.
- **Small, rare signal.** There are 244 churners in the training set (63 in the control arm), and the average effect is about 0.25 percentage points. Confidence intervals are wide.
- **IPW and DR estimates rely on the propensity and outcome models.** If features that drive both treatment and churn are missing, the correction is incomplete; with 63 control churners the outcome models are weak, so DR intervals are wide.
- **Churn model selection uses one validation split.** The churn model and its F1 threshold are still chosen on a single validation set, and its probabilities are calibrated on the same set.
- **Features are anonymised.** SHAP explanations refer to principal components, which carry no business meaning.

## Roadmap

- [x] Two-stage churn + uplift pipeline (T/S/X-Learner, Uplift Random Forest)
- [x] Tie-aware Qini with bootstrap confidence intervals
- [x] Expected-profit targeting with an explicit sign convention
- [x] SHAP explanations for the churn model
- [x] Streamlit dashboard
- [x] pytest suite and pinned, reproducible environment
- [x] Repeated cross-validation for uplift model selection
- [x] Propensity-aware evaluation (inverse-propensity-weighted Qini)
- [x] Probability calibration and PR-AUC for the churn model
- [x] Doubly-robust Qini and DR-Learner
- [x] Power analysis and policy evaluation
- [x] Batch pipeline (train, score, monitor) with config, saved models, drift monitoring and Docker
- [x] Continuous integration (GitHub Actions: pytest and go test on every push)
- [x] Repaired the dataset-paper helpers and the Go chart server

---

## 📖 Citation

```bibtex
@misc{atre2026churn,
  author = {Atre, Aryan},
  title = {Churn Prediction and Recommendation System with Uplift Modeling:
           A Causal Inference Approach to Customer Retention},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/atrearyanuf/Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling}},
  note = {CAI6826 - Artificial Intelligence Systems, University of Florida}
}
```

---

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

The helper modules in `functions/` other than `functions/targeting.py`, `functions/uplift_cv.py`, `functions/data.py`, `functions/models.py`, `functions/power.py`, `functions/policy.py` and the Qini functions in `functions/eval_measures.py` come from the dataset paper's code repository (see Acknowledgments).

---

##  Acknowledgments

### Dataset

- T. Verhelst, D. Mercier, J. Shrestha, G. Bontempi. *A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling.* ECML PKDD 2023 Workshop on Uplift Modeling and Causal Machine Learning for Operational Decision Making. [arXiv:2312.07206](https://arxiv.org/abs/2312.07206). Data provided by Orange Belgium, licensed CC BY-NC-ND 4.0 and published as [OpenML dataset 45580](https://www.openml.org/d/45580); code at [TheoVerhelst/Churn-Uplift-Dataset-Paper](https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper).

### Libraries & Frameworks

- **[XGBoost](https://xgboost.readthedocs.io/)** and **[LightGBM](https://lightgbm.readthedocs.io/)**: gradient boosting
- **[CausalML](https://causalml.readthedocs.io/)**: uplift modeling (Uber)
- **[SHAP](https://shap.readthedocs.io/)**: model interpretability
- **[Streamlit](https://streamlit.io/)**: dashboard
- **[scikit-learn](https://scikit-learn.org/)**: machine learning utilities

### Academic References

- Künzel et al. (2019): Meta-learners for estimating heterogeneous treatment effects
- Gutierrez & Gérardy (2017): Causal inference and uplift modeling, a review
- Radcliffe (2007): Using control groups to target on predicted lift (Qini)
- Lundberg & Lee (2017): A unified approach to interpreting model predictions (SHAP)

### Special Thanks

- **University of Florida**: CAI6826 course infrastructure
- **Dr. Silva and Professor Aslam**: project guidance and feedback

---

##  Contact

**Aryan Atre**
- University of Florida - Artificial Intelligence Systems
- Email: atrearyan@ufl.edu
- LinkedIn: [linkedin.com/in/aryan-atre](https://www.linkedin.com/in/aryan-atre)
- GitHub: [@atrearyanuf](https://github.com/atrearyanuf)
- Issues: [GitHub Issues](https://github.com/atrearyanuf/Churn-Prediction-and-Recommendation-System-with-Uplift-Modeling/issues)
