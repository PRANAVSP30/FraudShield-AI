import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from auth import authenticate

st.set_page_config(page_title="FraudShield AI", layout="wide")

# ============================
# CUSTOM FINTECH CSS
# ============================

st.markdown("""
<style>
body {
    background-color: #F8FAFC;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}
.login-box {
    background-color: white;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ============================
# AUTHENTICATION
# ============================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>💳 FraudShield AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Enterprise Fraud Monitoring System</h4>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ============================
# AUTO TRAIN MODEL
# ============================

@st.cache_resource
def train_model():
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
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    xgb.fit(X_res, y_res)

    y_prob = xgb.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)

    return xgb, auc, df

model, auc, df = train_model()

# ============================
# SIDEBAR
# ============================

st.sidebar.title("FraudShield AI")
page = st.sidebar.radio(
    "",
    ["📊 Dashboard", "🔍 Manual Prediction"]
)

st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# ============================
# DASHBOARD
# ============================

if page == "📊 Dashboard":

    st.title("📊 Fraud Monitoring Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
    col3.metric("Model AUC", f"{auc:.3f}")

    st.divider()

    # Fraud distribution chart
    fig = px.pie(df, names="Class", title="Fraud vs Normal Transactions")
    st.plotly_chart(fig, use_container_width=True)

# ============================
# MANUAL PREDICTION
# ============================

elif page == "🔍 Manual Prediction":

    st.title("🔍 Real-Time Fraud Scoring")

    cols = st.columns(3)
    features = []

    for i in range(30):
        features.append(cols[i % 3].number_input(f"Feature {i+1}", value=0.0))

    if st.button("Analyze Transaction"):
        sample = np.array(features).reshape(1, -1)
        prob = model.predict_proba(sample)[0][1]

        if prob > 0.8:
            st.error(f"🚨 HIGH RISK - Probability: {prob:.3f}")
        elif prob > 0.4:
            st.warning(f"⚠ MEDIUM RISK - Probability: {prob:.3f}")
        else:
            st.success(f"✅ LOW RISK - Probability: {prob:.3f}")