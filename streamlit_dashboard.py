from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Written by Churn_Uplift_COMPLETE.ipynb (paths relative to the repo root)
DATA_FILES = {
    "raw": "data/churn_uplift_anonymized.csv",
    "recs": "outputs/customer_recommendations.csv",
    "models": "outputs/uplift_comparison_results.csv",
}

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("Churn Prediction & Uplift Modeling")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_FILES["raw"])
    recs = pd.read_csv(DATA_FILES["recs"])
    models = pd.read_csv(DATA_FILES["models"])
    return raw, recs, models

missing = [f for f in DATA_FILES.values() if not Path(f).exists()]
if missing:
    st.error("Missing " + ", ".join(missing) + ". Run Churn_Uplift_COMPLETE.ipynb "
             "first, then start the dashboard from the repository root.")
    st.stop()

raw, recs, models = load_data()

# ── KPI metrics ────────────────────────────────────────────────────────────────
total = len(raw)
churn_rate = raw["y"].mean() * 100
treated = raw["t"].mean() * 100
avg_churn_prob = recs["churn_probability"].mean() * 100
avg_uplift = recs["retention_uplift"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churn Rate", f"{churn_rate:.1f}%")
c3.metric("Treatment Rate", f"{treated:.1f}%")
c4.metric("Avg Churn Probability", f"{avg_churn_prob:.1f}%")
c5.metric("Avg Retention Uplift", f"{avg_uplift:+.3f}")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(recs, x="churn_probability", nbins=30, color_discrete_sequence=["#3b82f6"])
    fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="Churn Probability", yaxis_title="Count")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Recommended Interventions")
    counts = recs["recommended_intervention"].value_counts().reset_index()
    counts.columns = ["Intervention", "Count"]
    fig2 = px.bar(counts, x="Intervention", y="Count", color_discrete_sequence=["#10b981"])
    fig2.update_layout(margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig2, width="stretch")

# ── Model comparison ───────────────────────────────────────────────────────────
st.subheader("Uplift Model Comparison")
st.dataframe(models, width="stretch")

# ── Customer recommendations table ─────────────────────────────────────────────
st.subheader("Customer Recommendations")

col_a, col_b = st.columns(2)
threshold = col_a.slider("Min Churn Probability", 0.0, 1.0, 0.3, 0.05)
interventions = recs["recommended_intervention"].unique().tolist()
selected = col_b.multiselect("Intervention Type", interventions, default=interventions)

filtered = recs[
    (recs["churn_probability"] >= threshold) &
    (recs["recommended_intervention"].isin(selected))
].sort_values("expected_profit", ascending=False)

st.dataframe(filtered.reset_index(drop=True), width="stretch")
st.caption(f"Showing {len(filtered):,} of {len(recs):,} customers")
