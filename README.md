# 🎯 Churn Prediction & Recommendation System with Uplift Modeling

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**A production-ready causal inference system that shifts customer retention from "who will churn?" to "who will respond?"**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## Overview

Traditional churn prediction models identify **who will churn** but fail to answer the critical question: **who will respond to retention interventions?** This leads to 30-40% of retention budgets being wasted on customers who won't respond.

This project implements a **production-ready uplift modeling system** that uses causal inference to identify "Persuadables" — customers who will stay if targeted, but churn otherwise. By integrating gradient boosting for churn prediction with meta-learner approaches for treatment effect estimation, the system delivers actionable, personalized recommendations that maximize retention ROI.

### Academic Context

- **Course:** CAI6826 - Artificial Intelligence Systems
- **Institution:** University of Florida
- **Author:** Aryan Atre
- **Year:** Spring 2026

### Key Results

- **Churn Prediction:** XGBoost AUC **0.839**, F1 **0.231** (28:1 class imbalance)
- **Uplift Modeling:** S-Learner Qini **0.33** (33% better than random targeting)
- **Business Impact:** **58 customers** targeted, **$27,948** net benefit, **1,204% ROI**
- **Deployment:** Streamlit dashboard with SHAP explainability

---

## Features

### Machine Learning

- **Two-Stage Pipeline**
  - Stage 1: Churn prediction (Random Forest, LightGBM, XGBoost)
  - Stage 2: Uplift modeling (T-Learner, S-Learner, X-Learner, UpliftRandomForest)
- **Advanced Data Preprocessing**
  - 4-group stratified splitting for causal signal preservation
  - Propensity score weighting (IPW) for treatment imbalance correction
  - Within-arm SMOTE oversampling (3.4% → 15% positive rate)
- **Comprehensive Evaluation**
  - Qini curves and coefficients
  - AUUC (Area Under Uplift Curve)
  - Stratified cross-validation
  - Bootstrap confidence intervals

### Model Interpretability

- **SHAP Integration**
  - Global feature importance
  - Individual prediction explanations
  - Waterfall plots for customer-level insights
- **Business-Friendly Visualizations**
  - ROI waterfall charts
  - Uplift distribution plots
  - Customer segment analysis

### Production Features

- **Modular Architecture**
  - Independent, testable components
  - Clean separation of concerns
  - Easy to extend and maintain
- **Data Pipeline**
  - Schema validation
  - Data quality checks
  - Feature engineering automation
- **Monitoring & Logging**
  - Model performance tracking
  - Drift detection capabilities
  - Comprehensive logging

### Interactive Dashboard

- **Streamlit UI**
  - Real-time customer recommendations
  - Interactive SHAP explanations
  - ROI visualization
  - CSV export functionality
  - Filtering by uplift threshold

---

## Demo

### Dashboard Screenshot
```
┌─────────────────────────────────────────────────────────────┐
│  Churn Uplift Recommendation System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 Business Impact                                          │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ Customers    │ Net Benefit  │ ROI          │            │
│  │ Targeted: 58 │ $27,948      │ 1,204%       │            │
│  └──────────────┴──────────────┴──────────────┘            │
│                                                              │
│  🎯 Top Recommendations                                      │
│  ┌────┬──────────┬──────────┬──────────┬────────────┐      │
│  │ ID │ Uplift   │ Churn %  │ Value    │ Action     │      │
│  ├────┼──────────┼──────────┼──────────┼────────────┤      │
│  │142 │ 0.089    │ 78%      │ $1,600   │ Discount   │      │
│  │ 89 │ 0.082    │ 81%      │ $1,450   │ Loyalty    │      │
│  └────┴──────────┴──────────┴──────────┴────────────┘      │
│                                                              │
│  📈 SHAP Explanation (Customer 142)                          │
│  PC_23: +0.034 ████████                                     │
│  PC_7:  -0.021 █████                                        │
│  PC_41: +0.018 ███                                          │
└─────────────────────────────────────────────────────────────┘
```

### Command Line Interface
```bash
# Train models
python src/train.py --config configs/default.yaml

# Generate recommendations
python src/predict.py --input data/customers.csv --output results/

# Launch dashboard
streamlit run src/dashboard.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                            │
│  Download → Validate → 4-Group Stratify → Propensity → SMOTE   │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────────┐         ┌────────────────────┐
│  Stage 1: Churn    │         │  Stage 2: Uplift   │
│  ├─ Random Forest  │         │  ├─ T-Learner      │
│  ├─ LightGBM       │         │  ├─ S-Learner ✓    │
│  └─ XGBoost ✓      │         │  ├─ X-Learner      │
│                    │         │  └─ UpliftRF       │
│  Output: P(Churn)  │         │  Output: τ(x)      │
└─────────┬──────────┘         └──────────┬─────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │  Recommendation      │
                │  Priority = 0.6×P +  │
                │             0.4×τ    │
                └──────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Streamlit │    │   SHAP   │    │   ROI    │
    │Dashboard │    │Explainer │    │ Analysis │
    └──────────┘    └──────────┘    └──────────┘
```

### System Modules

1. **Data Pipeline** (`src/data/`)
   - Ingestion, validation, preprocessing
   - 4-group stratified splitting
   - Propensity score calculation
   - Within-arm SMOTE oversampling

2. **Churn Prediction** (`src/models/churn/`)
   - Random Forest baseline
   - LightGBM with class balancing
   - XGBoost with scale_pos_weight

3. **Uplift Modeling** (`src/models/uplift/`)
   - T-Learner (two-model approach)
   - S-Learner (single-model approach)
   - X-Learner (propensity-weighted)
   - UpliftRandomForest

4. **Recommendation Engine** (`src/recommender/`)
   - Customer prioritization
   - Intervention mapping
   - SHAP-based explanations

5. **Dashboard** (`src/dashboard/`)
   - Streamlit web application
   - Interactive visualizations
   - CSV export

6. **Utilities** (`src/utils/`)
   - Logging configuration
   - Metrics calculation
   - Visualization helpers

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager) / Conda env package manager 
- virtualenv/CondaEnv (recommended)
- 16GB RAM (for full dataset processing)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/churn-uplift-modeling.git
cd churn-uplift-modeling
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n churn-uplift python=3.12
conda activate churn-uplift
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 4: Download Dataset

```bash
# Option 1: Download Orange Belgium dataset manually
# Place in data/raw/orange_belgium.csv

# Option 2: Use data download script
python scripts/download_data.py
```

### Step 5: Verify Installation

```bash
python -c "import xgboost, causalml, shap; print('Installation successful!')"
```

---

## Quick Start to the capstone project

### Option 1: End-to-End Pipeline

```bash
# Run complete pipeline (preprocessing → training → evaluation)
python run_pipeline.py --config configs/default.yaml
```

### Option 2: Step-by-Step

```bash
# Step 1: Preprocess data
python src/data/preprocess.py \
    --input data/raw/orange_belgium.csv \
    --output data/processed/

# Step 2: Train churn models
python src/models/churn/train_churn.py \
    --data data/processed/train.csv \
    --output models/churn/

# Step 3: Train uplift models
python src/models/uplift/train_uplift.py \
    --data data/processed/train.csv \
    --output models/uplift/

# Step 4: Generate recommendations
python src/recommender/recommend.py \
    --churn-model models/churn/xgboost.pkl \
    --uplift-model models/uplift/s_learner.pkl \
    --data data/processed/test.csv \
    --output results/recommendations.csv

# Step 5: Launch dashboard
streamlit run src/dashboard/app.py
```

### Option 3: Jupyter Notebook

```bash
jupyter notebook notebooks/01_complete_pipeline.ipynb
```

---

## Project Structure

```
churn-uplift-modeling/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package installation
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
│
├── configs/                    # Configuration files
│   ├── default.yaml            # Default hyperparameters
│   ├── production.yaml         # Production settings
│   └── experiment.yaml         # Experimental configs
│
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed data
│   └── external/               # External data sources
│
├── models/                     # Trained models (gitignored)
│   ├── churn/                  # Churn prediction models
│   └── uplift/                 # Uplift modeling models
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_complete_pipeline.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_uplift_analysis.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading
│   │   ├── preprocessor.py     # Preprocessing pipeline
│   │   ├── stratify.py         # 4-group stratification
│   │   ├── propensity.py       # Propensity score calculation
│   │   └── smote.py            # Within-arm SMOTE
│   │
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── churn/              # Churn prediction
│   │   │   ├── random_forest.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── xgboost_model.py
│   │   └── uplift/             # Uplift modeling
│   │       ├── t_learner.py
│   │       ├── s_learner.py
│   │       ├── x_learner.py
│   │       └── uplift_rf.py
│   │
│   ├── evaluation/             # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── churn_metrics.py    # AUC, F1, Precision, Recall
│   │   ├── uplift_metrics.py   # Qini, AUUC
│   │   └── business_metrics.py # ROI, Net Benefit
│   │
│   ├── explainability/         # Model interpretability
│   │   ├── __init__.py
│   │   ├── shap_explainer.py   # SHAP integration
│   │   └── feature_importance.py
│   │
│   ├── recommender/            # Recommendation engine
│   │   ├── __init__.py
│   │   ├── engine.py           # Core recommendation logic
│   │   └── interventions.py    # Intervention mapping
│   │
│   ├── dashboard/              # Streamlit dashboard
│   │   ├── __init__.py
│   │   ├── app.py              # Main dashboard
│   │   ├── components.py       # UI components
│   │   └── visualizations.py   # Plotting functions
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py           # Configuration loader
│       ├── logger.py           # Logging setup
│       └── helpers.py          # Helper functions
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_recommender.py
│
├── scripts/                    # Utility scripts
│   ├── download_data.py        # Dataset download
│   ├── train_all_models.py     # Batch training
│   └── generate_report.py      # Results reporting
│
├── results/                    # Experiment results (gitignored)
│   ├── figures/                # Plots and visualizations
│   ├── metrics/                # Performance metrics
│   └── recommendations/        # Generated recommendations
│
├── docs/                       # Documentation
│   ├── architecture.md         # System architecture
│   ├── api.md                  # API reference
│   ├── deployment.md           # Deployment guide
│   └── troubleshooting.md      # Common issues
│
└── docker/                     # Docker configuration
    ├── Dockerfile              # Production image
    ├── docker-compose.yml      # Multi-container setup
    └── .dockerignore           # Docker ignore rules
```

---

## Configuration

### Configuration Files

Configuration is managed via YAML files in `configs/`:

```yaml
# configs/default.yaml
data:
  path: "data/raw/orange_belgium.csv"
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

preprocessing:
  stratify_groups: 4
  propensity_clip: [0.05, 0.95]
  smote_target_ratio: 0.15

models:
  churn:
    xgboost:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
      scale_pos_weight: 28.2
    
  uplift:
    s_learner:
      base_model: "xgboost"
      params:
        max_depth: 6
        learning_rate: 0.1

evaluation:
  cv_folds: 5
  bootstrap_iterations: 1000
  confidence_level: 0.95

business:
  intervention_cost: 40
  customer_value: 1600
```

### Environment Variables

```bash
# .env file
DATA_PATH=data/raw/orange_belgium.csv
MODEL_PATH=models/
RESULTS_PATH=results/
LOG_LEVEL=INFO
STREAMLIT_PORT=8501
```

---

##  Usage

### Training Models

```python
from src.data.preprocessor import DataPreprocessor
from src.models.churn.xgboost_model import XGBoostChurnModel
from src.models.uplift.s_learner import SLearner

# Load and preprocess data
preprocessor = DataPreprocessor(config="configs/default.yaml")
X_train, y_train, t_train = preprocessor.load_and_preprocess()

# Train churn model
churn_model = XGBoostChurnModel(scale_pos_weight=28.2)
churn_model.fit(X_train, y_train)
churn_probs = churn_model.predict_proba(X_test)

# Train uplift model
uplift_model = SLearner(base_model="xgboost")
uplift_model.fit(X_train, y_train, t_train)
uplift_scores = uplift_model.predict(X_test, t_test)
```

### Generating Recommendations

```python
from src.recommender.engine import RecommendationEngine

# Initialize recommender
recommender = RecommendationEngine(
    churn_model=churn_model,
    uplift_model=uplift_model,
    intervention_cost=40
)

# Generate recommendations
recommendations = recommender.recommend(
    X_test,
    top_k=58,
    min_uplift=0.01
)

# Export to CSV
recommendations.to_csv("results/recommendations.csv")
```

### SHAP Explanations

```python
from src.explainability.shap_explainer import SHAPExplainer

# Initialize SHAP
explainer = SHAPExplainer(uplift_model)

# Global feature importance
feature_importance = explainer.global_importance(X_test)

# Individual explanation
customer_explanation = explainer.explain_customer(
    customer_id=142,
    X=X_test
)

# Waterfall plot
explainer.plot_waterfall(customer_id=142)
```

### Dashboard

```python
# Launch Streamlit dashboard
streamlit run src/dashboard/app.py

# Or with custom config
streamlit run src/dashboard/app.py -- --config configs/production.yaml
```

---

## API Reference

### Data Pipeline

```python
class DataPreprocessor:
    """
    Handles data loading, validation, and preprocessing.
    
    Methods:
        load_and_preprocess() -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        stratify_4_groups(X, y, t) -> Tuple[...]
        calculate_propensity_scores(X, t) -> np.ndarray
        apply_smote(X, y, t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
```

### Models

```python
class XGBoostChurnModel:
    """
    XGBoost binary classification for churn prediction.
    
    Args:
        scale_pos_weight (float): Class weight for positive class
        max_depth (int): Maximum tree depth
        learning_rate (float): Boosting learning rate
    
    Methods:
        fit(X, y) -> None
        predict(X) -> np.ndarray
        predict_proba(X) -> np.ndarray
    """

class SLearner:
    """
    S-Learner meta-algorithm for uplift modeling.
    
    Args:
        base_model (str): Base model type ('xgboost', 'lightgbm', 'rf')
        **kwargs: Base model parameters
    
    Methods:
        fit(X, y, t) -> None
        predict(X, t) -> np.ndarray  # Uplift scores τ(x)
    """
```

### Evaluation

```python
def calculate_qini_coefficient(
    y_true: np.ndarray,
    uplift_pred: np.ndarray,
    treatment: np.ndarray
) -> float:
    """
    Calculate Qini coefficient.
    
    Args:
        y_true: True outcomes
        uplift_pred: Predicted uplift scores
        treatment: Treatment assignment
    
    Returns:
        Qini coefficient (higher is better)
    """
```

---

## Model Performance

### Churn Prediction Results

| Model | AUC | F1 Score | Precision | Recall |
|-------|-----|----------|-----------|--------|
| Random Forest | 0.821 | 0.215 | 0.342 | 0.157 |
| LightGBM | 0.838 | 0.226 | 0.359 | 0.164 |
| **XGBoost** | **0.839** | **0.231** | **0.365** | **0.169** |

**Context:** F1 of 0.231 represents maximum signal extraction with 28:1 class imbalance and 408 total positives.

### Uplift Modeling Results

| Model | Qini Coefficient | AUUC | Interpretation |
|-------|------------------|------|----------------|
| **S-Learner** | **0.33** | **0.21** | 33% better than random |
| T-Learner | -0.71 | -0.52 | Overfitting (worse than random) |
| X-Learner | -1.90 | -1.34 | Severe overfitting |
| UpliftRandomForest | -0.84 | -0.61 | Overfitting |

**Why S-Learner Won:** With only 170/60 samples per group, T-Learner suffered high variance. S-Learner's pooled approach (230 total samples) provided better variance-bias tradeoff.

### Business Impact

| Metric | Value | Notes |
|--------|-------|-------|
| Customers Targeted | 58 | High-uplift persuadables |
| Intervention Cost | $2,320 | 58 × $40 per customer |
| Projected Saves | ~19 | Based on Qini 0.33 |
| Revenue Protected | $30,268 | Customer lifetime value |
| Net Benefit | $27,948 | After intervention cost |
| **ROI** | **1,204%** | Projected impact |

**Validation Note:** ROI is projected based on model predictions. Production deployment requires A/B testing for validation, and not following the Next Feature Fallacy.

---

## Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

### Test Structure

```python
# tests/test_models.py
import pytest
from src.models.uplift.s_learner import SLearner

def test_s_learner_fit():
    """Test S-Learner training."""
    model = SLearner(base_model="xgboost")
    model.fit(X_train, y_train, t_train)
    assert hasattr(model, 'model_')

def test_s_learner_predict():
    """Test S-Learner prediction."""
    model = SLearner(base_model="xgboost")
    model.fit(X_train, y_train, t_train)
    uplift = model.predict(X_test, t_test)
    assert len(uplift) == len(X_test)
    assert -1 <= uplift.mean() <= 1  # Reasonable range
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src
```

---

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t churn-uplift:latest -f docker/Dockerfile .

# Run container
docker run -p 8501:8501 churn-uplift:latest

# Using docker-compose
docker-compose -f docker/docker-compose.yml up
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py"]
```

### Cloud Deployment

#### AWS EC2

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Clone repository
git clone https://github.com/yourusername/churn-uplift-modeling.git
cd churn-uplift-modeling

# Install dependencies
pip install -r requirements.txt

# Run with nohup
nohup streamlit run src/dashboard/app.py --server.port 8501 &
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-uplift
gcloud run deploy --image gcr.io/PROJECT_ID/churn-uplift --platform managed
```

#### Heroku

```bash
# Create Procfile
echo "web: streamlit run src/dashboard/app.py --server.port=\$PORT" > Procfile

# Deploy
heroku create churn-uplift-app
git push heroku main
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/churn-uplift-modeling.git
   cd churn-uplift-modeling
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add docstrings to all functions
   - Write unit tests for new features
   - Update documentation

4. **Run tests**
   ```bash
   pytest tests/ -v
   black src/ tests/  # Code formatting
   flake8 src/ tests/  # Linting
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Describe your changes
   - Link related issues
   - Request review

### Code Style

```python
# Good: Clear docstrings, type hints, descriptive names
def calculate_qini_coefficient(
    y_true: np.ndarray,
    uplift_pred: np.ndarray,
    treatment: np.ndarray
) -> float:
    """
    Calculate Qini coefficient for uplift model evaluation.
    
    Args:
        y_true: Binary outcomes (0/1)
        uplift_pred: Predicted uplift scores
        treatment: Treatment assignment (0/1)
    
    Returns:
        Qini coefficient between -1 and 1 (higher is better)
    
    Example:
        >>> qini = calculate_qini_coefficient(y_true, uplift, treatment)
        >>> print(f"Qini: {qini:.3f}")
    """
    # Implementation...
```

### Contribution Areas

- **Bug Fixes:** Found a bug? Submit an issue and PR!
- **New Features:** Add new meta-learners, evaluation metrics, or visualizations
- **Documentation:** Improve README, add tutorials, write guides
- **Testing:** Increase test coverage, add integration tests
- **UI/UX:** Enhance dashboard design, add new visualizations for steamlit
- **Performance:** Optimize code, reduce memory usage, speed up training

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{atre2026churn,
  author = {Atre, Aryan},
  title = {Churn Prediction and Recommendation System with Uplift Modeling: 
           A Causal Inference Approach to Customer Retention},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/churn-uplift-modeling}},
  note = {CAI6826 - Artificial Intelligence Systems, University of Florida}
}
```

**Academic Paper:**
```
Atre, A. (2026). Churn Prediction and Recommendation System with Uplift Modeling: 
A Causal Inference Approach to Customer Retention. 
CAI6826 Final Project Report, University of Florida.
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

---

##  Acknowledgments

### Libraries & Frameworks

- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting framework
- **[LightGBM](https://lightgbm.readthedocs.io/)** - Gradient boosting framework
- **[CausalML](https://causalml.readthedocs.io/)** - Uplift modeling library by Uber
- **[SHAP](https://shap.readthedocs.io/)** - Model interpretability
- **[Streamlit](https://streamlit.io/)** - Dashboard framework
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning utilities

### Academic References

- Künzel et al. (2019) - Meta-learners for heterogeneous treatment effects
- Gutierrez & Gérardy (2017) - Causal inference and uplift modeling review
- Lundberg & Lee (2017) - SHAP unified framework
- Radcliffe & Surry (2011) - Real-world uplift modeling

### Datasets

- **Orange Belgium Telecom Dataset** - Customer churn data for research and education

### Special Thanks

- **University of Florida** - CAI6826 course infrastructure
- **Professor [Name]** - Project guidance and feedback(Dr. Silva and Professor Aslam)
- **Open Source Community** - Libraries and tools that made this possible

---

##  Contact

**Aryan Atre**
- University of Florida - Artificial Intelligence Systems
- Email: atrearayn@ufl.edu
- LinkedIn: [linkedin.com/in/aryan-atre](https://www.linkedin.com/in/aryan-atre)
- GitHub: [@yourusername](https://github.com/atrearyanuf)

### Getting Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/churn-uplift-modeling/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/churn-uplift-modeling/discussions)
- **Email:** For inquiries: atrearayn@ufl.edu

---

##  Roadmap

### Completed 
- [x] Two-stage churn + uplift pipeline
- [x] T-Learner, S-Learner, X-Learner, UpliftRandomForest
- [x] SHAP explainability integration
- [x] Streamlit dashboard
- [x] Comprehensive documentation
- [x] Unit tests and CI/CD

### In Progress 
- [ ] Deep learning uplift models (DragonNet, TARNet)
- [ ] Multi-armed uplift (3+ treatment options)
- [ ] Real-time A/B testing integration

### Future Plans 
- [ ] Temporal modeling with survival analysis
- [ ] Reinforcement learning for sequential interventions
- [ ] AutoML for hyperparameter optimization
- [ ] REST API for production deployment
- [ ] Mobile app for recommendations

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Made with ❤️ by [Aryan Atre](https://github.com/yourusername)**

</div>
