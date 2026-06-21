import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Professional ML Dashboard for Customer Retention Analysis")

# ---------------- LEFT COLUMN ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Info")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", 0, 100)

with col2:
    st.subheader("Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

with col3:
    st.subheader("Billing Info")
    DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

# ---------------- FINANCIAL INPUTS ---------------- #
st.subheader("💰 Billing Details")

col4, col5 = st.columns(2)

with col4:
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)

with col5:
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

# ---------------- ENCODING ---------------- #

def yes_no(x):
    return 1 if x == "Yes" else 0

input_data = np.array([[
    SeniorCitizen,
    tenure,
    MonthlyCharges,
    TotalCharges,
    1 if gender == "Male" else 0,
    yes_no(Partner),
    yes_no(Dependents),
    yes_no(PhoneService),
    1 if MultipleLines == "No phone service" else 0,
    1 if MultipleLines == "Yes" else 0,
    1 if InternetService == "Fiber optic" else 0,
    1 if InternetService == "No" else 0,
    1 if OnlineSecurity == "No internet service" else 0,
    yes_no(OnlineSecurity),
    1 if OnlineBackup == "No internet service" else 0,
    yes_no(OnlineBackup),
    1 if DeviceProtection == "No internet service" else 0,
    yes_no(DeviceProtection),
    1 if TechSupport == "No internet service" else 0,
    yes_no(TechSupport),
    1 if StreamingTV == "No internet service" else 0,
    yes_no(StreamingTV),
    1 if StreamingMovies == "No internet service" else 0,
    yes_no(StreamingMovies),
    1 if Contract == "One year" else 0,
    1 if Contract == "Two year" else 0,
    yes_no(PaperlessBilling),
    1 if PaymentMethod == "Credit card (automatic)" else 0,
    1 if PaymentMethod == "Electronic check" else 0,
    1 if PaymentMethod == "Mailed check" else 0
]])

# ---------------- PREDICTION ---------------- #

if st.button("🔮 Predict Churn"):
    input_data = scaler.transform(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Result")

    col6, col7 = st.columns(2)

    with col6:
        st.metric("Churn Probability", f"{prob:.2f}")

    with col7:
        if prob > 0.7:
            st.error("🔴 High Risk Customer")
            st.write("Recommendation: Offer discount / retention call")
        elif prob > 0.4:
            st.warning("🟠 Medium Risk Customer")
            st.write("Recommendation: Engagement campaign")
        else:
            st.success("🟢 Low Risk Customer")
            st.write("Recommendation: Maintain service quality")