import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Churn Prediction ",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HEADER ---------------- #
st.title("📊 Customer Churn Prediction and Retention System")
st.markdown("AI-powered Customer Churn & Retention Intelligence System")

st.markdown("---")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Customer Input")

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
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ---------------- BUILD INPUT (FIXED FEATURE ALIGNMENT) ---------------- #
input_df = pd.DataFrame(columns=scaler.feature_names_in_)
input_df.loc[0] = 0

# numeric
input_df["SeniorCitizen"] = SeniorCitizen
input_df["tenure"] = tenure
input_df["MonthlyCharges"] = MonthlyCharges
input_df["TotalCharges"] = TotalCharges

# categorical encoding
input_df["gender_Male"] = 1 if gender == "Male" else 0
input_df["Partner_Yes"] = 1 if Partner == "Yes" else 0
input_df["Dependents_Yes"] = 1 if Dependents == "Yes" else 0
input_df["PhoneService_Yes"] = 1 if PhoneService == "Yes" else 0

input_df["MultipleLines_Yes"] = 1 if MultipleLines == "Yes" else 0
input_df["MultipleLines_No phone service"] = 1 if MultipleLines == "No phone service" else 0

input_df["InternetService_Fiber optic"] = 1 if InternetService == "Fiber optic" else 0
input_df["InternetService_No"] = 1 if InternetService == "No" else 0

input_df["OnlineSecurity_Yes"] = 1 if OnlineSecurity == "Yes" else 0
input_df["OnlineSecurity_No internet service"] = 1 if OnlineSecurity == "No internet service" else 0

input_df["OnlineBackup_Yes"] = 1 if OnlineBackup == "Yes" else 0
input_df["OnlineBackup_No internet service"] = 1 if OnlineBackup == "No internet service" else 0

input_df["DeviceProtection_Yes"] = 1 if DeviceProtection == "Yes" else 0
input_df["DeviceProtection_No internet service"] = 1 if DeviceProtection == "No internet service" else 0

input_df["TechSupport_Yes"] = 1 if TechSupport == "Yes" else 0
input_df["TechSupport_No internet service"] = 1 if TechSupport == "No internet service" else 0

input_df["StreamingTV_Yes"] = 1 if StreamingTV == "Yes" else 0
input_df["StreamingTV_No internet service"] = 1 if StreamingTV == "No internet service" else 0

input_df["StreamingMovies_Yes"] = 1 if StreamingMovies == "Yes" else 0
input_df["StreamingMovies_No internet service"] = 1 if StreamingMovies == "No internet service" else 0

input_df["Contract_One year"] = 1 if Contract == "One year" else 0
input_df["Contract_Two year"] = 1 if Contract == "Two year" else 0

input_df["PaperlessBilling_Yes"] = 1 if PaperlessBilling == "Yes" else 0

input_df["PaymentMethod_Electronic check"] = 1 if PaymentMethod == "Electronic check" else 0
input_df["PaymentMethod_Mailed check"] = 1 if PaymentMethod == "Mailed check" else 0
input_df["PaymentMethod_Credit card (automatic)"] = 1 if PaymentMethod == "Credit card (automatic)" else 0

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Analyze Customer"):

    X = scaler.transform(input_df)
    prob = model.predict_proba(X)[0][1]

    # ---------------- METRICS ---------------- #
    col1, col2 = st.columns(2)

    col1.metric("Churn Probability", f"{prob:.2%}")

    risk = "High 🔴" if prob > 0.7 else "Medium 🟠" if prob > 0.4 else "Low 🟢"
    col2.metric("Risk Level", risk)

    st.markdown("---")

    # ---------------- SPEEDOMETER GAUGE ---------------- #
    st.markdown("## 📊 Risk Speedometer")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Churn Risk Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, 30], "color": "#00C853"},   # green
                {"range": [30, 60], "color": "#FFD600"},  # yellow
                {"range": [60, 80], "color": "#FF6D00"},  # orange
                {"range": [80, 100], "color": "#D50000"}   # red
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RETENTION ENGINE ---------------- #
    st.markdown("## 🎯 Retention Strategy")

    if prob > 0.7:
        st.error("🚨 HIGH RISK CUSTOMER")
        action = "Call immediately + discount offer + priority support"
    elif prob > 0.4:
        st.warning("⚠️ MEDIUM RISK CUSTOMER")
        action = "Engagement campaign + retention email + upgrade offer"
    else:
        st.success("✅ LOW RISK CUSTOMER")
        action = "Maintain service + upsell opportunity"

    st.info(f"💼 Recommended Action: {action}")

    # ---------------- TOP 3 REASONS ---------------- #
    st.markdown("## 🧠 Top 3 Risk Reasons")

    reasons = []

    if tenure < 12:
        reasons.append("Low tenure (new customer)")
    if Contract == "Month-to-month":
        reasons.append("Month-to-month contract")
    if MonthlyCharges > 80:
        reasons.append("High monthly charges")
    if InternetService == "Fiber optic":
        reasons.append("Fiber optic service sensitivity")
    if OnlineSecurity == "No":
        reasons.append("No online security")
    if TechSupport == "No":
        reasons.append("No tech support")

    if len(reasons) == 0:
        st.success("Customer is stable 👍")
    else:
        for r in reasons[:3]:
            st.markdown(f"✔️ {r}")

    # ---------------- BUSINESS INSIGHT ---------------- #
    st.markdown("## 💰 Business Insight")

    if prob > 0.7:
        st.write("⚠️ High revenue loss risk if customer churns.")
    elif prob > 0.4:
        st.write("📊 Moderate risk — early intervention can save revenue.")
    else:
        st.write("✅ Stable customer — long-term value likely.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Churn Prediction and Retention System | Machine Learning")
