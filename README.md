#Project Overview

This project is an end-to-end Machine Learning system that predicts whether a telecom customer will leave the service or not and provides retention strategies based on risk level.

It uses customer behavior data such as tenure, contract type, internet service, and billing details to make predictions.

A Streamlit web application is built to provide an interactive interface where users can input customer details and get churn probability along with risk classification.

**Features**
-Predicts customer churn using ML model
-Outputs churn probability
-Classifies customers into:
-Low Risk
-Medium Risk
-High Risk
-Provides retention strategies
-Interactive Streamlit web app

**Machine Learning Workflow**
-Data preprocessing
-Feature engineering
-Encoding categorical variables
-Feature scaling using StandardScaler
-Model training
-Model saving (model.pkl)
-Scaler saving (scaler.pkl)

**Tech Stack**
-Python
-Pandas
-NumPy
-Scikit-learn
-Streamlit
-Joblib

**Input Features**

The model uses features like:
-Customer demographics (Gender, SeniorCitizen, Partner, Dependents)
-Services (InternetService, OnlineSecurity, TechSupport, etc.)
-Account info (Contract, Payment Method)
-Billing (MonthlyCharges, TotalCharges)
-Tenure
**How to Run Locally**
```bash
pip install streamlit
streamlit run app.py
```

**Requirements**
-streamlit
-numpy
-pandas
-scikit-learn
-joblib

**Output Example**
-Churn Probability: 0.78
-Risk Level: High Risk
-Recommendation: Offer discount or retention call

**Business Impact**
-Reduces customer churn
-Improves customer retention
-Helps data-driven decision making
-Increases revenue stability

**App Link**
https://customer-churn-prediction-system-hqwtf6znbjrrjhgiwuymsp.streamlit.app/
**Author**
Chiluveru Umesh Chandra
Github:https://github.com/chiluveruumesh/customer-churn-prediction-system
Linkdin:www.linkedin.com/in/umesh-chiluveru

**Future Improvements**
-Deploy on cloud (Streamlit Cloud / AWS)
-Add explainable AI (SHAP)
-Add dashboard analytics
-Add database integration
