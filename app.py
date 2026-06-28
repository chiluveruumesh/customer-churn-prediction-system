import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HEADER ---------------- #
st.title("📊 Customer Churn Intelligence System")
st.markdown("AI-powered retention and churn prediction dashboard")

st.markdown("---")

# ---------------- INPUT SECTION ---------------- #
st.markdown("## 📝 Customer Information")

col1, col2, col3 = st.columns(3)

# -------- COLUMN 1 -------- #
with col1:
    st.subheader("👤 Profile")

    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 100, 12)

# -------- COLUMN 2 -------- #
with col2:
    st.subheader("🌐 Services")

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# -------- COLUMN 3 -------- #
with col3:
    st.subheader("💳 Billing")

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

st.markdown("---")

# ---------------- PREDICT BUTTON ---------------- #
predict = st.button("🚀 Analyze Customer", use_container_width=True)

# ---------------- LOGIC ---------------- #
if predict:

    input_df = pd.DataFrame(columns=scaler.feature_names_in_)
    input_df.loc[0] = 0

    # numeric
    input_df["SeniorCitizen"] = SeniorCitizen
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = MonthlyCharges
    input_df["TotalCharges"] = TotalCharges

    # encoding
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
    input_df["StreamingMovies_Yes"] = 1 if StreamingMovies == "Yes" else 0

    input_df["Contract_One year"] = 1 if Contract == "One year" else 0
    input_df["Contract_Two year"] = 1 if Contract == "Two year" else 0

    input_df["PaperlessBilling_Yes"] = 1 if PaperlessBilling == "Yes" else 0

    input_df["PaymentMethod_Electronic check"] = 1 if PaymentMethod == "Electronic check" else 0
    input_df["PaymentMethod_Mailed check"] = 1 if PaymentMethod == "Mailed check" else 0
    input_df["PaymentMethod_Credit card (automatic)"] = 1 if PaymentMethod == "Credit card (automatic)" else 0

    # prediction
    X = scaler.transform(input_df)
    prob = model.predict_proba(X)[0][1]

    # ---------------- METRICS ---------------- #
    colA, colB, colC = st.columns(3)

    colA.metric("Churn Probability", f"{prob:.2%}")
    risk = "High 🔴" if prob > 0.7 else "Medium 🟠" if prob > 0.4 else "Low 🟢"
    colB.metric("Risk Level", risk)
    colC.metric("Customer Status", "At Risk" if prob > 0.5 else "Safe")

    st.markdown("---")

    # ---------------- GAUGE ---------------- #
    st.subheader("📊 Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#111"},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f1c40f"},
                {"range": [60, 80], "color": "#e67e22"},
                {"range": [80, 100], "color": "#e74c3c"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RETENTION ---------------- #
    st.subheader("🎯 Retention Strategy")

    if prob > 0.7:
        action = "Immediate call + discount + priority support"
    elif prob > 0.4:
        action = "Engagement email + upgrade offer"
    else:
        action = "Maintain service + upsell opportunity"

    st.info(action)

    # ---------------- REASONS ---------------- #
    st.subheader("🧠 Top Risk Reasons")

    reasons = []

    if tenure < 12:
        reasons.append(("Low tenure", 3))
    if Contract == "Month-to-month":
        reasons.append(("Short contract", 3))
    if MonthlyCharges > 80:
        reasons.append(("High charges", 2))
    if InternetService == "Fiber optic":
        reasons.append(("Fiber optic sensitivity", 2))
    if TechSupport == "No":
        reasons.append(("No tech support", 2))
    if OnlineSecurity == "No":
        reasons.append(("No security", 2))

    if len(reasons) == 0:
        st.success("Customer is stable 👍")
    else:
        for r, _ in sorted(reasons, key=lambda x: -x[1])[:3]:
            st.write("✔️", r)

    # ---------------- BUSINESS IMPACT ---------------- #
    st.subheader("💰 Business Insight")

    if prob > 0.7:
        st.write("High revenue loss risk ⚠️")
    elif prob > 0.4:
        st.write("Moderate risk — intervention recommended")
    else:
        st.write("Stable customer — long-term value")
