import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page setup & Professional Branding
st.set_page_config(
    page_title="Credit Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Owner credit & disclaimer (All Rights Reserved)
st.markdown("""
<div style="background-color:#0F172A; padding:20px; border-radius:12px; margin-bottom:25px; border-left: 6px solid #3B82F6;">
    <h1 style="color:white; margin:0; font-size:24px;">Credit Risk & Customer Churn Intelligence Portal</h1>
    <p style="color:#94A3B8; margin:5px 0 0 0; font-size:14px;">
        Designed & Built by <strong>Lead Financial Analyst Raunak Sharma</strong> | Pinnacle National Bank • Predictive Division
    </p>
    <span style="color:#3B82F6; font-size:11px; font-weight:bold; text-transform:uppercase; letter-spacing:1px;">
        © 2026 Raunak Sharma. All Rights Reserved. Private Financial Intellectual Property.
    </span>
</div>
""", unsafe_allow_html=True)

# Generate or load realistic baseline dataset for training
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Simulating standard columns
    credit_scores = np.random.randint(400, 850, n_samples)
    ages = np.random.randint(18, 80, n_samples)
    tenure = np.random.randint(0, 10, n_samples)
    balances = np.random.uniform(0, 250000, n_samples)
    num_products = np.random.randint(1, 5, n_samples)
    has_card = np.random.randint(0, 2, n_samples)
    active_member = np.random.randint(0, 2, n_samples)
    salaries = np.random.uniform(10000, 200000, n_samples)
    
    # LogReg logic coefficients for ground truth exited labels
    # Simulating weights that match real-world indicators: Age (+), Balance (+), Low Credit Score (-), Low Products (-)
    z = (0.05 * (ages - 40) + 
         0.000004 * (balances - 50000) - 
         0.003 * (credit_scores - 600) - 
         0.5 * active_member - 
         0.2 * tenure +
         np.random.normal(0, 0.8, n_samples))
    
    probs = 1 / (1 + np.exp(-z))
    exited = (probs > 0.5).astype(int)
    
    df = pd.DataFrame({
        'CreditScore': credit_scores,
        'Age': ages,
        'Tenure': tenure,
        'Balance': balances,
        'NumOfProducts': num_products,
        'HasCrCard': has_card,
        'IsActiveMember': active_member,
        'EstimatedSalary': salaries,
        'Exited': exited
    })
    
    # Feature Engineering derived features
    df['BalanceToSalaryRatio'] = np.where(df['EstimatedSalary'] > 0, df['Balance'] / df['EstimatedSalary'], 0)
    df['ProductDensity'] = np.where(df['Tenure'] > 0, df['NumOfProducts'] / df['Tenure'], df['NumOfProducts'])
    df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']
    df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
    
    return df

try:
    df = load_sample_data()
except Exception as e:
    st.error(f"Error initializing modeling data: {e}")

# Features list
feature_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'EstimatedSalary', 'BalanceToSalaryRatio', 'ProductDensity', 
    'EngagementProductInteraction', 'AgeTenureInteraction', 'IsActiveMember'
]

# Standardize and train model
X = df[feature_cols]
y = df['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(C=1.0, max_iter=500)
model.fit(X_scaled, y)

# Sidebar Simulator Controls
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:15px;">
    <h3 style="margin:0; color:#1E293B;">Parameters</h3>
    <small style="color:#64748B;">Raunak Sharma's Model Input Suite</small>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Customer Financial Profile")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age (Years)", 18, 90, 42)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance (€)", min_value=0.0, value=75000.0)
num_products = st.sidebar.selectbox("Number of Active Products", [1, 2, 3, 4], index=1)
estimated_salary = st.sidebar.number_input("Estimated Annual Salary (€)", min_value=0.0, value=110000.0)

st.sidebar.header("Engagement & Relationship")
has_card = st.sidebar.checkbox("Has Credit Card", value=True)
is_active = st.sidebar.checkbox("Is Active Member", value=True)

# Process simulation input
balance_to_salary = balance / estimated_salary if estimated_salary > 0 else 0
prod_density = num_products / tenure if tenure > 0 else num_products
engagement_interaction = (1 if is_active else 0) * num_products
age_tenure_interaction = age * tenure

# Pack simulation vector
sim_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'EstimatedSalary': estimated_salary,
    'BalanceToSalaryRatio': balance_to_salary,
    'ProductDensity': prod_density,
    'EngagementProductInteraction': engagement_interaction,
    'AgeTenureInteraction': age_tenure_interaction,
    'IsActiveMember': 1 if is_active else 0
}])

# Align prediction columns
sim_data = sim_data[feature_cols]
sim_scaled = scaler.transform(sim_data)

# Run Inference
prob = model.predict_proba(sim_scaled)[0][1]
is_churn = prob > 0.5

# Visual Result Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Interactive Predictive Inference")
    
    # Meter visual
    gauge_color = "red" if is_churn else "green"
    st.markdown(f"""
    <div style="background-color:#F8FAFC; border: 1px solid #E2E8F0; padding:25px; border-radius:12px; text-align:center;">
        <h4 style="margin:0 0 10px 0; color:#475569;">Churn Churn Probability</h4>
        <h1 style="color:{gauge_color}; font-size:64px; margin:0; font-family:monospace;">{prob*100:.1f}%</h1>
        <p style="margin:10px 0 0 0; font-size:18px; font-weight:bold; color:#1E293B;">
            Recommendation: <span style="color:{gauge_color};">{"CRITICAL CHURN RISK (ACT NOW)" if is_churn else "RETAINED (LOW RISK)"}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("#### Dynamic Feature Diagnostics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Balance-to-Salary Ratio", f"{balance_to_salary:.2f}")
    m_col2.metric("Product Density", f"{prod_density:.2f}")
    m_col3.metric("Engagement Score", f"{engagement_interaction}")

with col2:
    st.subheader("Model Insights & Coefficients")
    
    # Feature Importance plotting
    coefs = model.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Impact Coefficient': coefs,
        'Risk Influence': ['Increases Risk' if c > 0 else 'Protects Retention' for c in coefs]
    }).sort_values(by='ImpactCoefficient', key=abs, ascending=False)
    
    fig = px.bar(
        importance_df, 
        x='Impact Coefficient', 
        y='Feature', 
        color='Risk Influence',
        orientation='h',
        color_discrete_map={'Increases Risk': '#EF4444', 'Protects Retention': '#10B981'},
        title="Standardized LogReg Feature Weights (Risk Factors)"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Sample Dataset preview
st.subheader("Baseline Reference Data Sample")
st.dataframe(df.head(10), use_container_width=True)
