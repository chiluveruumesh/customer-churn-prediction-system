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
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

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
    "tenure": float(tenure),
    "MonthlyCharges": float(MonthlyCharges),
    "TotalCharges": float(TotalCharges),

    # gender
    "gender_Male": 1 if gender == "Male" else 0,

    # Partner / Dependents
    "Partner_Yes": 1 if Partner == "Yes" else 0,
    "Dependents_Yes": 1 if Dependents == "Yes" else 0,

    # Phone
    "PhoneService_Yes": 1 if PhoneService == "Yes" else 0,

    # MultipleLines
    "MultipleLines_No phone service": 1 if MultipleLines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if MultipleLines == "Yes" else 0,

    # InternetService
    "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,
    "InternetService_No": 1 if InternetService == "No" else 0,

    # OnlineSecurity
    "OnlineSecurity_No internet service": 1 if OnlineSecurity == "No internet service" else 0,
    "OnlineSecurity_Yes": 1 if OnlineSecurity == "Yes" else 0,

    # OnlineBackup
    "OnlineBackup_No internet service": 1 if OnlineBackup == "No internet service" else 0,
    "OnlineBackup_Yes": 1 if OnlineBackup == "Yes" else 0,

    # DeviceProtection
    "DeviceProtection_No internet service": 1 if DeviceProtection == "No internet service" else 0,
    "DeviceProtection_Yes": 1 if DeviceProtection == "Yes" else 0,

    # TechSupport
    "TechSupport_No internet service": 1 if TechSupport == "No internet service" else 0,
    "TechSupport_Yes": 1 if TechSupport == "Yes" else 0,

    # StreamingTV
    "StreamingTV_No internet service": 1 if StreamingTV == "No internet service" else 0,
    "StreamingTV_Yes": 1 if StreamingTV == "Yes" else 0,

    # StreamingMovies
    "StreamingMovies_No internet service": 1 if StreamingMovies == "No internet service" else 0,
    "StreamingMovies_Yes": 1 if StreamingMovies == "Yes" else 0,

    # Contract
    "Contract_One year": 1 if Contract == "One year" else 0,
    "Contract_Two year": 1 if Contract == "Two year" else 0,

    # Billing
    "PaperlessBilling_Yes": 1 if PaperlessBilling == "Yes" else 0,

    # PaymentMethod
    "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0

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

    st.markdown("## 📊 Risk Level Breakdown")

    if prob > 0.7:
        st.error("🔴 HIGH RISK")
        st.write("Action: Immediate retention call + discount offer")
    elif prob > 0.4:
        st.warning("🟠 MEDIUM RISK")
        st.write("Action: Engagement campaign")
    else:
        st.success("🟢 LOW RISK")
        st.write("Action: Maintain service quality")
        st.markdown("## 🧠 Why is this customer at risk?")
    reasons = []

    if tenure < 12:
        reasons.append("⚠️ Low tenure (new customer)")

    if Contract == "Month-to-month":
        reasons.append("⚠️ Month-to-month contract")

    if InternetService == "Fiber optic":
        reasons.append("⚠️ Fiber optic service (higher churn tendency)")

    if MonthlyCharges > 80:
        reasons.append("⚠️ High monthly charges")

    if OnlineSecurity == "No":
       reasons.append("⚠️ No online security")

    if TechSupport == "No":
        reasons.append("⚠️ No tech support")

    if len(reasons) == 0:
        reasons.append("✅ Customer profile is stable")

    for r in reasons:
        st.write(r)
    st.markdown("## 📉 Churn Probability Gauge")

    st.progress(float(prob))
    st.write(f"Probability Score: {prob:.2%}")
    st.markdown("## 💼 Business Insight")

    if prob > 0.7:
        st.write("🚨 This customer is highly likely to leave. Losing them could impact revenue.")
    elif prob > 0.4:
        st.write("⚠️ This customer may churn. Early intervention recommended.")
    else:
        st.write("✅ This customer is stable and likely to continue service.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Customer Churn Prediction System | Machine Learning Project")
