import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from auth import authenticate

st.set_page_config(page_title="FraudShield AI", layout="wide")

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 FraudShield AI Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(user, pwd):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOAD MODELS ----------------
xgb = joblib.load("models/xgb_model.pkl")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Manual Prediction", "Batch Prediction", "Model Performance"]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("💳 Fraud Detection Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", "284,807")
    col2.metric("Fraud Cases", "492")
    col3.metric("Model AUC", "0.978")

# ---------------- MANUAL ----------------
elif page == "Manual Prediction":
    st.title("Manual Fraud Prediction")

    features = []
    for i in range(30):
        features.append(st.number_input(f"Feature {i+1}", value=0.0))

    if st.button("Predict"):
        sample = np.array(features).reshape(1, -1)
        prob = xgb.predict_proba(sample)[0][1]
        st.success(f"Fraud Probability: {prob:.4f}")

# ---------------- BATCH ----------------
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        probs = xgb.predict_proba(df)[:,1]
        df["Fraud Probability"] = probs
        st.dataframe(df.head())

        fig = px.histogram(df, x="Fraud Probability")
        st.plotly_chart(fig)

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.image("models/roc_curve.png")
    st.image("models/confusion_matrix.png")