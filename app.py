import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(page_title="FraudShield AI", layout="wide")

# ============================
# AUTHENTICATION
# ============================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>💳 FraudShield AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Enterprise Fraud Monitoring System</h4>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

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

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X_res, y_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    return model, auc, df

with st.spinner("Training model... Please wait (first run only)"):
    model, auc, df = train_model()

# ============================
# SIDEBAR
# ============================

st.sidebar.title("FraudShield AI")
page = st.sidebar.radio("", ["📊 Dashboard", "🔍 Fraud Simulation"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

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

    fig = px.pie(df, names="Class", title="Fraud vs Normal Transactions")
    st.plotly_chart(fig, use_container_width=True)

# ============================
# FRAUD SIMULATION
# ============================

elif page == "🔍 Fraud Simulation":

    st.title("🔍 Transaction Fraud Simulation")

    mode = st.radio(
        "Select Mode",
        [
            "Realistic Manual Entry",
            "Random Transaction Generator",
            "Select Real Dataset Transaction"
        ]
    )

    # =====================================================
    # 1️⃣ REALISTIC MANUAL ENTRY (BASED ON REAL DATA ROW)
    # =====================================================

    if mode == "Realistic Manual Entry":

        st.subheader("Modify a Real Transaction")

        # Select base row
        base_row = df.sample(1).iloc[0]

        st.write("Base Transaction Pattern Loaded (PCA structure preserved)")

        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        time_val = st.number_input("Transaction Time (seconds)", min_value=0.0, value=50000.0)

        if st.button("Analyze Modified Transaction"):

            modified_row = base_row.copy()

            modified_row["Amount"] = amount
            modified_row["Time"] = time_val

            features = modified_row.drop("Class").values.reshape(1, -1)

            prob = model.predict_proba(features)[0][1]

            if prob > 0.8:
                st.error(f"🚨 HIGH RISK - Probability: {prob:.3f}")
            elif prob > 0.4:
                st.warning(f"⚠ MEDIUM RISK - Probability: {prob:.3f}")
            else:
                st.success(f"✅ LOW RISK - Probability: {prob:.3f}")

    # =====================================================
    # 2️⃣ RANDOM TRANSACTION GENERATOR
    # =====================================================

    elif mode == "Random Transaction Generator":

        if st.button("Generate Random Transaction"):

            random_row = df.sample(1).iloc[0]
            features = random_row.drop("Class").values.reshape(1, -1)

            prob = model.predict_proba(features)[0][1]

            st.write("Generated Transaction:")
            st.dataframe(random_row)

            if prob > 0.8:
                st.error(f"🚨 HIGH RISK - Probability: {prob:.3f}")
            elif prob > 0.4:
                st.warning(f"⚠ MEDIUM RISK - Probability: {prob:.3f}")
            else:
                st.success(f"✅ LOW RISK - Probability: {prob:.3f}")

    # =====================================================
    # 3️⃣ SELECT REAL DATASET TRANSACTION
    # =====================================================

    elif mode == "Select Real Dataset Transaction":

        sample_df = df.sample(100).reset_index(drop=True)

        selected_index = st.selectbox("Choose Transaction", sample_df.index)
        selected_row = sample_df.loc[selected_index]

        st.write("Selected Transaction:")
        st.dataframe(selected_row)

        if st.button("Analyze Selected Transaction"):

            features = selected_row.drop("Class").values.reshape(1, -1)
            prob = model.predict_proba(features)[0][1]
            actual = selected_row["Class"]

            st.info(f"Actual Label: {'Fraud' if actual == 1 else 'Normal'}")

            if prob > 0.8:
                st.error(f"🚨 Predicted HIGH RISK - Probability: {prob:.3f}")
            elif prob > 0.4:
                st.warning(f"⚠ Predicted MEDIUM RISK - Probability: {prob:.3f}")
            else:
                st.success(f"✅ Predicted LOW RISK - Probability: {prob:.3f}")