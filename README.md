#  Churn Prediction & Recommendation System with Uplift Modeling

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)

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

All numbers below are printed by `Churn_Uplift_COMPLETE.ipynb`; `tests/test_readme.py` fails if they drift apart.

- **Churn prediction:** XGBoost reaches test AUC 0.814, PR-AUC 0.191 (random ranking: 0.035) and F1 0.250 on a 28:1 class imbalance. After calibration its mean predicted churn on test is 0.0359, against an actual 0.0345.
- **Uplift modeling:** in repeated cross-validation, every learner's mean out-of-fold IPW Qini is negative, and none beats random targeting in more than 40% of splits. The selected S-Learner scores test IPW Qini +10.14 (95% CI -4.05 to +26.42), so it is not distinguishable from random. Without the propensity correction, its test Qini would be +12.10 (95% CI +1.28 to +23.90), which looks significant. Its estimated effects (at most 0.017) are also below the 0.025 break-even uplift.
- **Recommendation:** with an assumed $1,600 customer value and $40 offer cost, the pipeline recommends **no offers**. That is the correct decision when the measured effect is this small.
- **Why:** the offer lowers churn by only 0.25 percentage points on average (3.64% control vs 3.39% treated in validation), the training set has 244 churners (63 in the control arm), and treatment probability depends on customer features (propensity AUC 0.631).

---

## How It Works

```
 Dataset (Orange Belgium, 11,896 customers, 178 anonymised features)
    │  label-encode categoricals, 60/20/20 split stratified on (churn × treatment)
    ▼
 ┌──────────────────────────┐        ┌──────────────────────────────────┐
 │ Stage 1: churn models    │        │ Stage 2: uplift models           │
 │  Random Forest, XGBoost, │        │  T / S / X-Learner (XGBoost),    │
 │  LightGBM → P(churn)     │        │  Uplift Random Forest            │
 │  threshold tuned for F1, │        │  → churn effect τ(x)             │
 │  Platt-calibrated        │        │                                  │
 └────────────┬─────────────┘        └────────────────┬─────────────────┘
              │                                       │ retention uplift = −τ(x)
              ▼                                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Evaluation: tie-aware Qini and IPW Qini on retention, 95% bootstrap CI │
 │ Selection:  repeated cross-validation (5 folds × 2 repeats)            │
 │ Targeting:  expected profit = value × retention uplift − offer cost    │
 │             offer only if expected profit > 0                          │
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
- **Model selection by repeated cross-validation.** A single validation split cannot separate the models (CI widths around ±12). The pipeline selects the model with the highest mean out-of-fold IPW Qini over 5 folds × 2 repeats of train + validation. The copy trained on the training set is then evaluated once on test.
- **Calibrated churn probabilities.** Class weighting inflates predicted churn. Platt scaling, fitted on validation, maps the predictions back to churn rates. It is monotone, so AUC and threshold decisions are unchanged. The recommendations and the dashboard show calibrated probabilities.

---

## Results

### Churn prediction

The decision threshold is chosen for F1 on the validation set.

| Model | Split | AUC | PR-AUC | F1 | Precision | Recall |
|-------|-------|-----|--------|----|-----------|--------|
| Random Forest | validation | 0.822 | 0.196 | 0.250 | 0.197 | 0.341 |
| LightGBM | validation | 0.825 | 0.176 | 0.294 | 0.344 | 0.256 |
| **XGBoost** (selected by AUC) | validation | 0.828 | 0.164 | 0.276 | 0.222 | 0.366 |
| **XGBoost** | **test** | **0.814** | **0.191** | **0.250** | **0.190** | **0.366** |

PR-AUC of a random ranking equals the churn rate (0.035). Random Forest has the best validation PR-AUC, but the model is selected by AUC.

| Calibration on test | Raw | Calibrated | Actual |
|---------------------|-----|------------|--------|
| Mean predicted churn | 0.0494 | 0.0359 | 0.0345 |
| Brier score | 0.0324 | 0.0311 | |

### Uplift modeling

Qini is reported on retention, in extra customers retained compared with random targeting, with a 95% bootstrap CI.

**Repeated cross-validation** (out-of-fold, 10 splits of train + validation), used for model selection:

| Model | Mean Qini | Mean IPW Qini | Std IPW Qini | Splits with IPW Qini > 0 |
|-------|-----------|---------------|--------------|--------------------------|
| S-Learner | -0.53 | -0.72 | 2.22 | 20% |
| X-Learner | -3.44 | -1.72 | 7.49 | 40% |
| T-Learner | -3.99 | -1.98 | 7.83 | 30% |

**Validation and test:**

| Model | Split | Qini | 95% CI | IPW Qini | 95% CI |
|-------|-------|------|--------|----------|--------|
| T-Learner | validation | -6.97 | -18.94 to +5.99 | -9.55 | -30.49 to +8.83 |
| S-Learner | validation | +5.87 | -3.30 to +16.16 | +1.36 | -10.30 to +13.09 |
| X-Learner | validation | -4.60 | -15.93 to +6.82 | +0.38 | -13.46 to +15.59 |
| UpliftRandomForest | validation | -4.10 | -14.30 to +6.74 | -10.91 | -31.16 to +7.59 |
| **S-Learner** (selected by cross-validation) | **test** | **+12.10** | **+1.28 to +23.90** | **+10.14** | **-4.05 to +26.42** |

No learner beats random targeting on average; the S-Learner is selected as the least bad. Its effects are tiny (std 0.001) because its single model barely uses the treatment column. The X-Learner labels 37.8% of customers persuadable, but its cross-validated IPW Qini is negative, so those labels are not reliable.

### Business decision

| Assumption | Value |
|------------|-------|
| Offer cost | $40 per contacted customer |
| Value of a retained customer | $1,600 |
| Break-even retention uplift | 0.025 |

On the 2,379 validation customers, the largest estimated retention uplift is 0.017, so expected profit is negative for everyone and the recommendation is **No_Action** for all of them. These dollar values are planning assumptions, not data from the campaign. Change `OFFER_COST` and `CUSTOMER_VALUE` in section 8 of the notebook to use your own.

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

The first cell downloads the dataset to `data/churn_uplift_anonymized.csv`. The full run takes under a minute on a laptop. It writes:

- `outputs/customer_recommendations.csv`: one row per validation customer with churn probability, retention uplift, expected profit, segment and recommendation
- `outputs/uplift_comparison_results.csv`: Qini with confidence interval for each uplift model
- plots in `outputs/` (ROC, confusion matrix, Qini curves, model comparison, SHAP)

### 2. Launch the dashboard

From the repository root, after running the notebook:

```bash
streamlit run streamlit_dashboard.py
```

It shows the KPIs, the churn-probability distribution, the recommendation counts, the uplift model comparison and a filterable recommendation table. If the notebook outputs are missing, it says so instead of crashing.

### 3. Go chart server (optional)

```bash
go run .        # needs Go 1.23+; serves http://localhost:8081
```

It shows a bar chart of the 20 customers with the highest retention uplift, read from `outputs/customer_recommendations.csv` on every request.

### 4. Use the targeting functions directly

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

The Python suite (`tests/`) checks:

- **Qini implementation:** it matches `causalml.metrics.get_qini` and `qini_score(normalize=False)`, ignores row order for tied scores, gives exactly 0 for a constant score and rejects invalid input.
- **Bootstrap intervals:** they are reproducible, contain 0 for a useless model and exclude 0 for a real signal.
- **IPW Qini:** unit weights reproduce the standard Qini. On a simulated confounded campaign, the weighted Qini is unbiased, while the unweighted one is off by about half.
- **Cross-validation:** every learner is scored on every split, and only on out-of-fold customers (a learner that memorises its training data scores exactly 0). A real learner ranks above noise.
- **Targeting:** the retention sign convention holds, offers go out only when profitable, and sleeping dogs are never targeted, even at high churn risk.
- **End to end:** on a synthetic campaign with a known effect, a T-Learner trained without oversampling targets true persuadables more than 90% of the time.
- **Dataset-paper helpers:** cross-validation splits cover every customer, EasyEnsemble folds are balanced and independent (including in parallel), and the wrappers handle 0/1 treatment flags.
- **Dashboard:** it renders from notebook outputs and explains missing files.
- **README:** every number above matches the notebook output, and every file path it mentions exists.

The Go tests check the CSV parsing (columns found by name, bad input rejected) and the HTTP handler.

---

## Project Structure

```
├── Churn_Uplift_COMPLETE.ipynb   # End-to-end pipeline (data → models → evaluation → recommendations → SHAP)
├── streamlit_dashboard.py        # Dashboard over the notebook's outputs
├── functions/
│   ├── eval_measures.py          # Qini and IPW Qini + bootstrap CI (used); profit curves (from the dataset paper)
│   ├── targeting.py              # Retention-uplift convention, segments, expected-profit recommendations
│   ├── uplift_cv.py              # Repeated cross-validation of uplift learners (out-of-fold Qini, IPW Qini)
│   └── benchmark.py, dataset.py, easy_ensemble.py, wrappers.py
│                                 # Benchmark helpers from the dataset paper's code (bugs fixed); used by uplift_cv.py
├── tests/                        # pytest suite
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
- **IPW is only as good as the propensity model.** If features that drive both treatment and churn are missing from it, the correction is incomplete.
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
- [ ] Continuous integration
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

The helper modules in `functions/` other than `functions/targeting.py`, `functions/uplift_cv.py` and the Qini functions in `functions/eval_measures.py` come from the dataset paper's code repository (see Acknowledgments).

---

##  Acknowledgments

### Dataset

- T. Verhelst, D. Mercier, J. Shrestha, G. Bontempi. *A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling.* ECML PKDD 2023 Workshop on Uplift Modeling and Causal Machine Learning for Operational Decision Making. [arXiv:2312.07206](https://arxiv.org/abs/2312.07206). Data provided by Orange Belgium; code at [TheoVerhelst/Churn-Uplift-Dataset-Paper](https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper).

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
