import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Load All Models
# ------------------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Risk Analyzer")
st.markdown("Multi-Model Comparison System")

st.divider()

# ------------------------------
# Model Selection
# ------------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["SVM (Recommended)", "Random Forest", "XGBoost", "LightGBM"]
)

# Map selection
if model_choice == "SVM (Recommended)":
    model = svm_model
elif model_choice == "Random Forest":
    model = rf_model
elif model_choice == "XGBoost":
    model = xgb_model
else:
    model = lgb_model

# ------------------------------
# Inputs (Same as before)
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

    total_charges = tenure * monthly_charges
    st.text_input("Total Charges (Auto Calculated)",
                  value=round(total_charges, 2),
                  disabled=True)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

st.divider()

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Analyze Churn Risk"):

    features = np.zeros(30)

    features[0] = tenure
    features[1] = monthly_charges
    features[2] = total_charges

    if internet_service == "Fiber optic":
        features[10] = 1
    elif internet_service == "No":
        features[11] = 1

    if tech_support == "No internet service":
        features[18] = 1
    elif tech_support == "Yes":
        features[19] = 1

    if contract == "One year":
        features[24] = 1
    elif contract == "Two year":
        features[25] = 1

    if payment_method == "Credit card (automatic)":
        features[27] = 1
    elif payment_method == "Electronic check":
        features[28] = 1
    elif payment_method == "Mailed check":
        features[29] = 1

    features = features.reshape(1, -1)

    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Use proper probability
    risk_score = model.predict_proba(features_pca)[0][1]

    st.subheader("📈 Churn Probability")
    st.progress(float(risk_score))
    st.markdown(f"### Risk Score: `{round(risk_score * 100, 2)}%`")

    if risk_score > 0.70:
        st.error("🚨 Very High Risk of Churn")
    elif risk_score > 0.50:
        st.error("⚠ High Risk of Churn")
    elif risk_score > 0.30:
        st.warning("⚡ Moderate Risk")
    else:
        st.success("✅ Low Risk – Customer Likely to Stay")

st.divider()

with st.expander("ℹ Model Performance Summary"):
    st.markdown("""
    | Model | Accuracy |
    |--------|----------|
    | Decision Tree | 73% |
    | Random Forest | 79% |
    | XGBoost | 80% |
    | LightGBM | 80% |
    | **SVM (Selected)** | **80.6%** |
    """)

st.markdown("---")
st.markdown("Developed by: **Ashu Gupta** | Final Semester Project – B.Tech AI")
