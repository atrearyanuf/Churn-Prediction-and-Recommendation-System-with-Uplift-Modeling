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

- **Churn prediction:** XGBoost reaches test AUC 0.814 and F1 0.250 on a 28:1 class imbalance.
- **Uplift modeling:** no model beats random targeting on the validation set; every 95% confidence interval contains 0. The S-Learner, selected on validation, scores test Qini +12.10 (95% CI +1.28 to +23.90). That is weak evidence, and its estimated effects (at most 0.017) are below the 0.025 break-even uplift.
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
 │  threshold tuned for F1  │        │  → churn effect τ(x)             │
 └────────────┬─────────────┘        └────────────────┬─────────────────┘
              │                                       │ retention uplift = −τ(x)
              ▼                                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Evaluation: tie-aware Qini on retention, 95% bootstrap CI              │
 │ Targeting: expected profit = value × retention uplift − offer cost     │
 │            offer only if expected profit > 0                           │
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

---

## Results

### Churn prediction

The decision threshold is chosen for F1 on the validation set.

| Model | Split | AUC | F1 | Precision | Recall |
|-------|-------|-----|----|-----------|--------|
| Random Forest | validation | 0.822 | 0.250 | 0.197 | 0.341 |
| LightGBM | validation | 0.825 | 0.294 | 0.344 | 0.256 |
| **XGBoost** (selected by AUC) | validation | 0.828 | 0.276 | 0.222 | 0.366 |
| **XGBoost** | **test** | **0.814** | **0.250** | **0.190** | **0.366** |

### Uplift modeling

Qini is reported on retention, in extra customers retained compared with random targeting, with a 95% bootstrap CI.

| Model | Split | Qini | 95% CI |
|-------|-------|------|--------|
| T-Learner | validation | -6.97 | -18.94 to +5.99 |
| S-Learner | validation | +5.87 | -3.30 to +16.16 |
| X-Learner | validation | -4.60 | -15.93 to +6.82 |
| UpliftRandomForest | validation | -4.10 | -14.30 to +6.74 |
| **S-Learner** (selected on validation) | **test** | **+12.10** | **+1.28 to +23.90** |

The S-Learner's effects are tiny (std 0.001): its single model barely uses the treatment column. The X-Learner labels 37.8% of customers persuadable, but its Qini is negative, so those labels are not reliable.

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

### 3. Use the targeting functions directly

```python
from functions.eval_measures import bootstrap_qini
from functions.targeting import recommend, retention_uplift

uplift = retention_uplift(learner.predict(X))           # CausalML churn effect -> retention uplift
qini, low, high = bootstrap_qini(1 - y, t, uplift)       # evaluate on retention (1 - churn)
recs = recommend(churn_probability, uplift, customer_value=1600, offer_cost=40)
```

---

## Testing

```bash
pytest
```

The suite (`tests/`) checks:

- **Qini implementation:** it matches `causalml.metrics.get_qini` and `qini_score(normalize=False)`, ignores row order for tied scores, gives exactly 0 for a constant score and rejects invalid input.
- **Bootstrap intervals:** they are reproducible, contain 0 for a useless model and exclude 0 for a real signal.
- **Targeting:** the retention sign convention holds, offers go out only when profitable, and sleeping dogs are never targeted, even at high churn risk.
- **End to end:** on a synthetic campaign with a known effect, a T-Learner trained without oversampling targets true persuadables more than 90% of the time.
- **Dashboard:** it renders from notebook outputs and explains missing files.
- **README:** every number above matches the notebook output, and every file path it mentions exists.

---

## Project Structure

```
├── Churn_Uplift_COMPLETE.ipynb   # End-to-end pipeline (data → models → evaluation → recommendations → SHAP)
├── streamlit_dashboard.py        # Dashboard over the notebook's outputs
├── functions/
│   ├── eval_measures.py          # Qini curve/coefficient + bootstrap CI (used); profit curves (from the dataset paper)
│   ├── targeting.py              # Retention-uplift convention, segments, expected-profit recommendations
│   └── benchmark.py, dataset.py, easy_ensemble.py, wrappers.py
│                                 # Helpers from the dataset paper's code; not used by the notebook
├── tests/                        # pytest suite
├── outputs/                      # Plots (tracked) and CSVs (generated, git-ignored)
├── end_user.go, go.mod, go.sum   # Experimental Go chart server (not maintained)
├── requirements.txt              # Pinned runtime dependencies
├── requirements-dev.txt          # + pytest
├── pytest.ini
└── *.pdf, *.pptx, *.html         # Course reports, slides and architecture diagrams
```

---

## Limitations

- **Treatment is only partly random.** According to the dataset paper, the control group was drawn at random from customers already selected as high churn risk. Within this data, treatment probability still varies with customer features (propensity AUC 0.631), so uplift estimates rely on there being no unmeasured confounding.
- **Small, rare signal.** There are 244 churners in the training set (63 in the control arm), and the average effect is about 0.25 percentage points. Confidence intervals are wide.
- **One validation split.** Model selection and threshold tuning use a single validation set; repeated cross-validation would give more stable comparisons.
- **Churn probabilities are not calibrated.** Class weighting inflates them (test mean 0.049 vs a 0.035 churn rate). They are shown for context, and the decision uses uplift only.
- **Features are anonymised.** SHAP explanations refer to principal components, which carry no business meaning.

## Roadmap

- [x] Two-stage churn + uplift pipeline (T/S/X-Learner, Uplift Random Forest)
- [x] Tie-aware Qini with bootstrap confidence intervals
- [x] Expected-profit targeting with an explicit sign convention
- [x] SHAP explanations for the churn model
- [x] Streamlit dashboard
- [x] pytest suite and pinned, reproducible environment
- [ ] Repeated cross-validation for model selection
- [ ] Propensity-aware evaluation (inverse-propensity-weighted Qini)
- [ ] Probability calibration and PR-AUC for the churn model
- [ ] Continuous integration
- [ ] Remove or update the unused helper modules and the Go chart server

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

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Aryan Atre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The helper modules in `functions/` other than `functions/targeting.py` and the Qini functions in `functions/eval_measures.py` come from the dataset paper's code repository (see Acknowledgments).

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
