import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ------------------------------
# AUTO TRAIN IF MODEL NOT EXISTS
# ------------------------------

@st.cache_resource
def train_model():

    # Download dataset automatically from Kaggle public link
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    xgb.fit(X_res, y_res)

    y_prob = xgb.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)

    return xgb, auc


st.set_page_config(page_title="FraudShield AI", layout="wide")
st.title("💳 FraudShield AI - Live Fraud Detection System")

st.info("Model will auto-train on first run. Please wait 1-2 minutes.")

model, auc = train_model()

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Manual Prediction"]
)

# ------------------------------
# DASHBOARD
# ------------------------------

if page == "Dashboard":

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    col1.metric("Dataset Size", "284,807")
    col2.metric("Model AUC", f"{auc:.3f}")

    st.success("Model successfully trained and deployed!")

# ------------------------------
# MANUAL PREDICTION
# ------------------------------

elif page == "Manual Prediction":

    st.subheader("Manual Fraud Prediction")

    features = []
    for i in range(30):
        features.append(st.number_input(f"Feature {i+1}", value=0.0))

    if st.button("Predict"):
        sample = np.array(features).reshape(1, -1)
        prob = model.predict_proba(sample)[0][1]

        if prob > 0.8:
            st.error(f"🚨 HIGH RISK ({prob:.3f})")
        elif prob > 0.4:
            st.warning(f"⚠ MEDIUM RISK ({prob:.3f})")
        else:
            st.success(f"✅ LOW RISK ({prob:.3f})")