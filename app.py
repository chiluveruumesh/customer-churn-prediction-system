import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("churn_pipeline.pkl")

# ---------------- HEADER ---------------- #
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer will churn using Machine Learning")

st.markdown("---")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.header("Customer Details")

SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 100, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ---------------- BUILD INPUT ---------------- #
input_df = pd.DataFrame([{
    "SeniorCitizen": int(SeniorCitizen),
    "gender": str(gender),
    "Partner": str(Partner),
    "Dependents": str(Dependents),
    "tenure": float(tenure),
    "PhoneService": str(PhoneService),
    "MultipleLines": str(MultipleLines),
    "InternetService": str(InternetService),
    "OnlineSecurity": str(OnlineSecurity),
    "OnlineBackup": str(OnlineBackup),
    "DeviceProtection": str(DeviceProtection),
    "TechSupport": str(TechSupport),
    "StreamingTV": str(StreamingTV),
    "StreamingMovies": str(StreamingMovies),
    "Contract": str(Contract),
    "PaperlessBilling": str(PaperlessBilling),
    "PaymentMethod": str(PaymentMethod),
    "MonthlyCharges": float(MonthlyCharges),
    "TotalCharges": float(TotalCharges)
}])

# ---------------- PREDICTION ---------------- #
if st.button("🔮 Predict Churn"):

    prob = model.predict_proba(input_df)[0][1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        risk = "High 🔴" if prob > 0.7 else "Medium 🟠" if prob > 0.4 else "Low 🟢"
        st.metric("Risk Level", risk)

    with col3:
        st.metric("Customer Status", "At Risk" if prob > 0.5 else "Safe")

    st.markdown("---")

    st.subheader("📊 Risk Analysis")

    st.progress(float(prob))

    if prob > 0.7:
        st.error("High Risk Customer 🔴")
        st.write("Recommendation: Offer discount or retention call.")
    elif prob > 0.4:
        st.warning("Medium Risk Customer 🟠")
        st.write("Recommendation: Engagement campaign.")
    else:
        st.success("Low Risk Customer 🟢")
        st.write("Recommendation: Maintain service quality.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Customer Churn Prediction System | Machine Learning Project")
