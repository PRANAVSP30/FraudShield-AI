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

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Scale
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Before SMOTE:")
print(y_train.value_counts())

# Handle imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_res.value_counts())

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss"
)

xgb.fit(X_res, y_res)

y_prob = xgb.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print("XGBoost AUC:", auc)

# Confusion Matrix
cm = confusion_matrix(y_test, xgb.predict(X_test))
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig("models/confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.savefig("models/roc_curve.png")

# Isolation Forest
iso = IsolationForest(contamination=0.001)
iso.fit(X_train)

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True)
lof.fit(X_train)

# Save models
joblib.dump(xgb, "models/xgb_model.pkl")
joblib.dump(iso, "models/iso_model.pkl")
joblib.dump(lof, "models/lof_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("All models saved successfully.")