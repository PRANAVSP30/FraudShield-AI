import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import bcrypt

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
# FINTECH UI STYLING
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

def check_password(username, password):
    hashed = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
    if username == "admin":
        return bcrypt.checkpw(password.encode(), hashed)
    return False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>💳 FraudShield AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Enterprise Fraud Monitoring</h4>", unsafe_allow_html=True)

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

    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)

    return model, auc, df

with st.spinner("Training fraud detection model... (first run only)"):
    model, auc, df = train_model()

# ============================
# SIDEBAR
# ============================

st.sidebar.title("FraudShield AI")

page = st.sidebar.radio(
    "",
    ["📊 Dashboard", "🔍 Fraud Simulation"]
)

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
# FRAUD SIMULATION MODULE
# ============================

elif page == "🔍 Fraud Simulation":

    st.title("🔍 Transaction Fraud Simulation")

    mode = st.radio(
        "Choose Simulation Mode",
        [
            "Manual Entry (Clean View)",
            "Random Transaction Generator",
            "Select Real Dataset Transaction"
        ]
    )

    # -----------------------------
    # CLEAN MANUAL ENTRY
    # -----------------------------
    if mode == "Manual Entry (Clean View)":

        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        time_val = st.number_input("Transaction Time (seconds)", min_value=0.0, value=50000.0)

        pca_features = np.zeros(28)

        if st.button("Analyze Transaction"):

            full_features = np.concatenate(([time_val], pca_features, [amount]))
            full_features = full_features.reshape(1, -1)

            prob = model.predict_proba(full_features)[0][1]

            if prob > 0.8:
                st.error(f"🚨 HIGH RISK - Probability: {prob:.3f}")
            elif prob > 0.4:
                st.warning(f"⚠ MEDIUM RISK - Probability: {prob:.3f}")
            else:
                st.success(f"✅ LOW RISK - Probability: {prob:.3f}")

    # -----------------------------
    # RANDOM GENERATOR
    # -----------------------------
    elif mode == "Random Transaction Generator":

        if st.button("Generate & Analyze"):

            random_features = np.random.normal(0, 1, 30).reshape(1, -1)
            prob = model.predict_proba(random_features)[0][1]

            st.write("Generated Transaction Features:")
            st.dataframe(pd.DataFrame(random_features))

            if prob > 0.8:
                st.error(f"🚨 HIGH RISK - Probability: {prob:.3f}")
            elif prob > 0.4:
                st.warning(f"⚠ MEDIUM RISK - Probability: {prob:.3f}")
            else:
                st.success(f"✅ LOW RISK - Probability: {prob:.3f}")

    # -----------------------------
    # REAL DATASET SELECTOR
    # -----------------------------
    elif mode == "Select Real Dataset Transaction":

        sample_df = df.sample(100).reset_index(drop=True)

        selected_index = st.selectbox("Choose Transaction Row", sample_df.index)
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