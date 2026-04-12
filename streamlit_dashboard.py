import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("Churn Prediction & Uplift Modeling")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv("data/churn_uplift_anonymized.csv")
    recs = pd.read_csv("outputs/customer_recommendations.csv")
    models = pd.read_csv("outputs/uplift_comparison_results.csv")
    return raw, recs, models

raw, recs, models = load_data()

# ── KPI metrics ────────────────────────────────────────────────────────────────
total = len(raw)
churn_rate = raw["y"].mean() * 100
treated = raw["t"].mean() * 100
avg_churn_prob = recs["churn_probability"].mean() * 100
avg_uplift = recs["uplift_score"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churn Rate", f"{churn_rate:.1f}%")
c3.metric("Treatment Rate", f"{treated:.1f}%")
c4.metric("Avg Churn Probability", f"{avg_churn_prob:.1f}%")
c5.metric("Avg Uplift Score", f"{avg_uplift:.3f}")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(recs, x="churn_probability", nbins=30, color_discrete_sequence=["#3b82f6"])
    fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="Churn Probability", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Recommended Interventions")
    counts = recs["recommended_intervention"].value_counts().reset_index()
    counts.columns = ["Intervention", "Count"]
    fig2 = px.bar(counts, x="Intervention", y="Count", color_discrete_sequence=["#10b981"])
    fig2.update_layout(margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ── Model comparison ───────────────────────────────────────────────────────────
st.subheader("Uplift Model Comparison")
st.dataframe(models, use_container_width=True)

# ── Customer recommendations table ─────────────────────────────────────────────
st.subheader("Customer Recommendations")

col_a, col_b = st.columns(2)
threshold = col_a.slider("Min Churn Probability", 0.0, 1.0, 0.3, 0.05)
interventions = recs["recommended_intervention"].unique().tolist()
selected = col_b.multiselect("Intervention Type", interventions, default=interventions)

filtered = recs[
    (recs["churn_probability"] >= threshold) &
    (recs["recommended_intervention"].isin(selected))
].sort_values("priority_score", ascending=False)

st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
st.caption(f"Showing {len(filtered):,} of {len(recs):,} customers")
