# Churn Prediction and Recommendation System with Uplift Modeling

**Author:** Aryan Atre  
**Email:** atrearayn@ufl.edu  
**Course:** CAP6307 - Artificial Intelligence Systems Capstone  
**Institution:** University of Florida  
**Semester:** Spring 2026

---

## Project Description

This project develops a machine learning system that combines churn prediction with uplift modeling to generate personalized customer retention recommendations. The system predicts which customers are likely to churn, estimates the effectiveness of different retention interventions using causal inference techniques, and recommends specific actions for each customer.

**Key Components:**
- Churn prediction using XGBoost and LightGBM
- Uplift modeling using Uber's CausalML (T-learner and S-learner)
- Personalized recommendation system
- SHAP-based model explainability
- Production-ready data pipelines and monitoring

**Goals:**
- Achieve AUC ≥ 0.85 for churn prediction
- Demonstrate positive uplift scores (Qini > 0.1)
- Generate actionable recommendations for at-risk customers

---

## Repository Structure

```
churn-uplift-recommendation-system/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── data/                        # Data files
│   ├── raw/                     # Original datasets
│   ├── processed/               # Cleaned data
│   └── README.md                # Data documentation
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_churn_prediction.ipynb
│   ├── 04_uplift_modeling.ipynb
│   ├── 05_recommendations.ipynb
│   └── complete_pipeline.ipynb
│
├── src/                         # Source code
│   ├── data/                    # Data processing
│   ├── models/                  # Model implementations
│   ├── evaluation/              # Metrics and evaluation
│   └── utils/                   # Helper functions
│
├── scripts/                     # Executable scripts
│   ├── train_churn_model.py
│   ├── train_uplift_model.py
│   └── generate_recommendations.py
│
├── tests/                       # Unit tests
│   ├── test_data_processing.py
│   └── test_models.py
│
├── models/                      # Saved models
│   ├── churn/
│   └── uplift/
│
├── outputs/                     # Results
│   ├── figures/                 # Plots and visualizations
│   ├── reports/                 # Analysis reports
│   └── recommendations/         # Generated recommendations
│
├── docs/                        # Documentation
│   ├── methodology.md
│   └── data_dictionary.md
│
└── configs/                     # Configuration files
    └── model_config.yaml
```

---

## Folder Descriptions

### `data/`
Contains all datasets used in the project.
- `raw/` - Original, unmodified data files
- `processed/` - Cleaned and preprocessed data
- `README.md` - Information about data sources and structure

### `notebooks/`
Jupyter notebooks for analysis and experimentation. Files are numbered in execution order (01, 02, 03...).

### `src/`
Python source code organized by functionality:
- `data/` - Data loading and preprocessing
- `models/` - Churn prediction and uplift modeling
- `evaluation/` - Performance metrics and visualization
- `utils/` - Helper functions

### `scripts/`
Standalone Python scripts for specific tasks (model training, evaluation, etc.)

### `tests/`
Unit tests for code validation

### `models/`
Saved trained models and preprocessing artifacts

### `outputs/`
Generated results:
- `figures/` - Charts and plots
- `reports/` - Analysis documents
- `recommendations/` - Customer recommendation files

### `docs/`
Project documentation and methodology

### `configs/`
Configuration files for models and pipelines

---

## How to Navigate This Repository

**For first-time users:**
1. Read this README
2. Check `requirements.txt` for dependencies
3. Read `data/README.md` for dataset information
4. Run notebooks in order: `01_data_exploration.ipynb` → `02_feature_engineering.ipynb` → etc.
5. View results in `outputs/`

**To run the project:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/run_pipeline.py

# Or use Jupyter notebooks
jupyter notebook notebooks/complete_pipeline.ipynb
```

**Key files to review:**
- `notebooks/complete_pipeline.ipynb` - Complete implementation
- `outputs/reports/` - Results and analysis
- `docs/methodology.md` - Technical details

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/AryanAtre/churn-uplift-recommendation-system.git
cd churn-uplift-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
# Follow instructions in data/README.md
```

4. Run the analysis:
```bash
jupyter notebook notebooks/complete_pipeline.ipynb
```

---

## Technical Stack

- **Language:** Python 3.8+
- **ML Libraries:** XGBoost, LightGBM, scikit-learn
- **Uplift Modeling:** CausalML (Uber), scikit-uplift
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn, plotly

---

## Collaborators

- Aryan Atre - Project Lead (atrearayn@ufl.edu)
- Dr. Aslam Usman Gadha - Instructor (gadhaunaslam@ufl.edu)
- [Advisor Name] - Project Advisor

---

## License

This project is for academic purposes as part of CAP6307 coursework at the University of Florida.