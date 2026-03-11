import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Churn Prevention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #028090;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #028090 0%, #00A896 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .impact-positive {
        color: #02C39A;
        font-weight: bold;
    }
    .impact-negative {
        color: #E63946;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Business assumptions (configurable in sidebar)
st.sidebar.header("⚙️ Business Parameters")
avg_customer_value = st.sidebar.number_input(
    "Average Customer Lifetime Value ($)",
    min_value=100,
    max_value=10000,
    value=1200,
    step=100,
    help="Average revenue per customer over their lifetime"
)

intervention_costs = st.sidebar.expander("💰 Intervention Costs", expanded=True)
with intervention_costs:
    cost_premium = st.number_input("Premium Support ($)", value=150, step=10)
    cost_discount = st.number_input("20% Discount ($)", value=100, step=10)
    cost_outreach = st.number_input("Personalized Outreach ($)", value=50, step=5)
    cost_feature = st.number_input("Feature Recommendation ($)", value=30, step=5)
    cost_no_action = st.number_input("No Action ($)", value=0, step=0)

intervention_success = st.sidebar.expander("📈 Success Rates", expanded=False)
with intervention_success:
    st.info("Estimated retention improvement per intervention")
    success_premium = st.slider("Premium Support", 0, 100, 40, 5) / 100
    success_discount = st.slider("20% Discount", 0, 100, 35, 5) / 100
    success_outreach = st.slider("Personalized Outreach", 0, 100, 25, 5) / 100
    success_feature = st.slider("Feature Recommendation", 0, 100, 15, 5) / 100

# Load data
@st.cache_data
def load_data():
    """Load the recommendations CSV"""
    try:
        df = pd.read_csv('customer_recommendations.csv')
        return df
    except FileNotFoundError:
        # Generate sample data for demo
        np.random.seed(42)
        n_customers = 2049
        
        df = pd.DataFrame({
            'customer_index': range(n_customers),
            'churn_probability': np.random.beta(2, 8, n_customers),
            'uplift_score': np.random.normal(0.03, 0.02, n_customers),
            'actual_treatment': np.random.binomial(1, 0.5, n_customers),
            'actual_outcome': np.random.binomial(1, 0.045, n_customers),
            'priority_score': np.random.beta(3, 7, n_customers)
        })
        
        # Assign interventions
        def assign_intervention(row):
            if row['churn_probability'] > 0.7 and row['uplift_score'] > 0.05:
                return 'Premium_Support'
            elif row['churn_probability'] > 0.5 and row['uplift_score'] > 0.03:
                return 'Discount_20pct'
            elif row['uplift_score'] > 0.03:
                return 'Personalized_Outreach'
            elif row['churn_probability'] > 0.5:
                return 'Feature_Recommendation'
            else:
                return 'No_Action'
        
        df['recommended_intervention'] = df.apply(assign_intervention, axis=1)
        return df

df = load_data()

# Main title
st.markdown('<h1 class="main-header">🎯 Churn Prevention & ROI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Calculate metrics
intervention_cost_map = {
    'Premium_Support': cost_premium,
    'Discount_20pct': cost_discount,
    'Personalized_Outreach': cost_outreach,
    'Feature_Recommendation': cost_feature,
    'No_Action': cost_no_action
}

success_rate_map = {
    'Premium_Support': success_premium,
    'Discount_20pct': success_discount,
    'Personalized_Outreach': success_outreach,
    'Feature_Recommendation': success_feature,
    'No_Action': 0.0
}

df['intervention_cost'] = df['recommended_intervention'].map(intervention_cost_map)
df['expected_success_rate'] = df['recommended_intervention'].map(success_rate_map)
df['expected_retention_lift'] = df['expected_success_rate'] * df['churn_probability']

# BEFORE (No intervention)
customers_at_risk_before = (df['churn_probability'] > 0.5).sum()
expected_churn_before = df['churn_probability'].sum()
revenue_loss_before = expected_churn_before * avg_customer_value

# AFTER (With targeted interventions)
targeted_customers = (df['recommended_intervention'] != 'No_Action').sum()
total_intervention_cost = df['intervention_cost'].sum()
expected_customers_saved = df['expected_retention_lift'].sum()
revenue_saved = expected_customers_saved * avg_customer_value
net_roi = revenue_saved - total_intervention_cost
roi_percentage = (net_roi / total_intervention_cost * 100) if total_intervention_cost > 0 else 0

# Expected churn after intervention
expected_churn_after = expected_churn_before - expected_customers_saved
revenue_loss_after = expected_churn_after * avg_customer_value

# === TOP METRICS ROW ===
st.subheader("📊 Executive KPI Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Customers Targeted",
        value=f"{targeted_customers:,}",
        delta=f"{targeted_customers/len(df)*100:.1f}% of total"
    )

with col2:
    st.metric(
        label="💰 Total Investment",
        value=f"${total_intervention_cost:,.0f}",
        delta=f"${total_intervention_cost/targeted_customers:.0f} per customer" if targeted_customers > 0 else "N/A"
    )

with col3:
    st.metric(
        label="💵 Revenue Protected",
        value=f"${revenue_saved:,.0f}",
        delta=f"{expected_customers_saved:.0f} customers saved"
    )

with col4:
    roi_color = "normal" if roi_percentage > 0 else "inverse"
    st.metric(
        label="📈 Net ROI",
        value=f"${net_roi:,.0f}",
        delta=f"{roi_percentage:.1f}% return",
        delta_color=roi_color
    )

st.markdown("---")

# === BEFORE vs AFTER COMPARISON ===
st.subheader("🔄 Before vs After: Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📉 **BEFORE** (No Intervention)")
    
    before_metrics = pd.DataFrame({
        'Metric': [
            'At-Risk Customers',
            'Expected Churn',
            'Revenue at Risk',
            'Intervention Cost',
            'Net Impact'
        ],
        'Value': [
            f"{customers_at_risk_before:,}",
            f"{expected_churn_before:.0f}",
            f"${revenue_loss_before:,.0f}",
            "$0",
            f"-${revenue_loss_before:,.0f}"
        ]
    })
    
    st.dataframe(before_metrics, use_container_width=True, hide_index=True)
    
    # Pie chart - Before
    fig_before = go.Figure(data=[go.Pie(
        labels=['Expected Churn', 'Expected Retain'],
        values=[expected_churn_before, len(df) - expected_churn_before],
        hole=0.4,
        marker_colors=['#E63946', '#02C39A']
    )])
    fig_before.update_layout(
        title="Customer Retention Forecast",
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig_before, use_container_width=True)

with col2:
    st.markdown("### 📈 **AFTER** (With Targeted Interventions)")
    
    after_metrics = pd.DataFrame({
        'Metric': [
            'Customers Targeted',
            'Expected Churn',
            'Revenue at Risk',
            'Intervention Cost',
            'Net Impact'
        ],
        'Value': [
            f"{targeted_customers:,}",
            f"{expected_churn_after:.0f}",
            f"${revenue_loss_after:,.0f}",
            f"${total_intervention_cost:,.0f}",
            f"+${net_roi:,.0f}"
        ]
    })
    
    st.dataframe(after_metrics, use_container_width=True, hide_index=True)
    
    # Pie chart - After
    fig_after = go.Figure(data=[go.Pie(
        labels=['Expected Churn', 'Expected Retain'],
        values=[expected_churn_after, len(df) - expected_churn_after],
        hole=0.4,
        marker_colors=['#FCA311', '#02C39A']
    )])
    fig_after.update_layout(
        title="Customer Retention Forecast",
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig_after, use_container_width=True)

# Delta summary
improvement = expected_churn_before - expected_churn_after
st.success(f"✅ **Impact**: Prevented {improvement:.0f} churns | Saved ${revenue_saved:,.0f} in revenue | ROI of {roi_percentage:.1f}%")

st.markdown("---")

# === INTERVENTION BREAKDOWN ===
st.subheader("🎯 Intervention Strategy Breakdown")

intervention_summary = df.groupby('recommended_intervention').agg({
    'customer_index': 'count',
    'churn_probability': 'mean',
    'uplift_score': 'mean',
    'intervention_cost': 'sum',
    'expected_retention_lift': 'sum'
}).reset_index()

intervention_summary.columns = [
    'Intervention', 'Customers', 'Avg Churn Prob',
    'Avg Uplift', 'Total Cost', 'Expected Saved'
]

intervention_summary['Revenue Protected'] = intervention_summary['Expected Saved'] * avg_customer_value
intervention_summary['Net Benefit'] = intervention_summary['Revenue Protected'] - intervention_summary['Total Cost']
intervention_summary['ROI %'] = (intervention_summary['Net Benefit'] / intervention_summary['Total Cost'] * 100).fillna(0)

# Format for display
display_summary = intervention_summary.copy()
display_summary['Avg Churn Prob'] = display_summary['Avg Churn Prob'].apply(lambda x: f"{x:.2%}")
display_summary['Avg Uplift'] = display_summary['Avg Uplift'].apply(lambda x: f"{x:.4f}")
display_summary['Total Cost'] = display_summary['Total Cost'].apply(lambda x: f"${x:,.0f}")
display_summary['Expected Saved'] = display_summary['Expected Saved'].apply(lambda x: f"{x:.1f}")
display_summary['Revenue Protected'] = display_summary['Revenue Protected'].apply(lambda x: f"${x:,.0f}")
display_summary['Net Benefit'] = display_summary['Net Benefit'].apply(lambda x: f"${x:,.0f}")
display_summary['ROI %'] = display_summary['ROI %'].apply(lambda x: f"{x:.1f}%")

st.dataframe(display_summary, use_container_width=True, hide_index=True)

# Visualization
col1, col2 = st.columns(2)

with col1:
    fig_dist = px.bar(
        intervention_summary,
        x='Intervention',
        y='Customers',
        title='Customer Distribution by Intervention',
        color='Intervention',
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig_dist.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    fig_roi = px.bar(
        intervention_summary[intervention_summary['Intervention'] != 'No_Action'],
        x='Intervention',
        y='ROI %',
        title='ROI by Intervention Type',
        color='ROI %',
        color_continuous_scale='RdYlGn'
    )
    fig_roi.update_layout(height=350)
    st.plotly_chart(fig_roi, use_container_width=True)

st.markdown("---")

# === PRIORITY CUSTOMER LIST ===
st.subheader("👥 Top Priority Customers for Immediate Action")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    intervention_filter = st.multiselect(
        "Filter by Intervention",
        options=df['recommended_intervention'].unique(),
        default=df['recommended_intervention'].unique()
    )
with col2:
    min_churn = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.1)
with col3:
    top_n = st.slider("Show Top N Customers", 10, 100, 20, 10)

# Filter and display
filtered_df = df[
    (df['recommended_intervention'].isin(intervention_filter)) &
    (df['churn_probability'] >= min_churn)
].sort_values('priority_score', ascending=False).head(top_n)

display_df = filtered_df[[
    'customer_index', 'churn_probability', 'uplift_score',
    'priority_score', 'recommended_intervention'
]].copy()

display_df.columns = ['Customer ID', 'Churn Prob', 'Uplift Score', 'Priority', 'Action']
display_df['Churn Prob'] = display_df['Churn Prob'].apply(lambda x: f"{x:.1%}")
display_df['Uplift Score'] = display_df['Uplift Score'].apply(lambda x: f"{x:.4f}")
display_df['Priority'] = display_df['Priority'].apply(lambda x: f"{x:.4f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Priority List (CSV)",
    data=csv,
    file_name="priority_customers.csv",
    mime="text/csv"
)

st.markdown("---")

# === ADVANCED ANALYTICS ===
with st.expander("📊 Advanced Analytics & Insights", expanded=False):
    
    tab1, tab2, tab3 = st.tabs(["Risk Segmentation", "Uplift Distribution", "Cost-Benefit Analysis"])
    
    with tab1:
        st.markdown("#### Customer Risk Segmentation")
        
        # Create risk segments
        df['risk_segment'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        segment_counts = df['risk_segment'].value_counts().reset_index()
        segment_counts.columns = ['Risk Segment', 'Count']
        
        fig_segments = px.pie(
            segment_counts,
            values='Count',
            names='Risk Segment',
            title='Customer Distribution by Risk Level',
            color_discrete_sequence=['#02C39A', '#FCA311', '#F77F00', '#E63946']
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with tab2:
        st.markdown("#### Uplift Score Distribution")
        
        fig_uplift = px.histogram(
            df,
            x='uplift_score',
            nbins=50,
            title='Distribution of Uplift Scores',
            labels={'uplift_score': 'Uplift Score'},
            color_discrete_sequence=['#028090']
        )
        fig_uplift.add_vline(x=df['uplift_score'].mean(), line_dash="dash", line_color="red",
                            annotation_text=f"Mean: {df['uplift_score'].mean():.4f}")
        st.plotly_chart(fig_uplift, use_container_width=True)
    
    with tab3:
        st.markdown("#### Cost-Benefit Analysis by Risk Segment")
        
        segment_analysis = df.groupby('risk_segment').agg({
            'customer_index': 'count',
            'intervention_cost': 'sum',
            'expected_retention_lift': 'sum'
        }).reset_index()
        
        segment_analysis['Revenue Saved'] = segment_analysis['expected_retention_lift'] * avg_customer_value
        segment_analysis['Net Benefit'] = segment_analysis['Revenue Saved'] - segment_analysis['intervention_cost']
        
        fig_segment_roi = px.bar(
            segment_analysis,
            x='risk_segment',
            y=['intervention_cost', 'Revenue Saved'],
            title='Investment vs Return by Risk Segment',
            barmode='group',
            labels={'value': 'Amount ($)', 'risk_segment': 'Risk Segment'}
        )
        st.plotly_chart(fig_segment_roi, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.9rem;'>
    <p><strong>Churn Prediction & Uplift Modeling System</strong></p>
    <p>Powered by XGBoost, CausalML (T-Learner), and SHAP</p>
    <p>University of Florida | CAI 6826 | Spring 2026</p>
</div>
""", unsafe_allow_html=True)