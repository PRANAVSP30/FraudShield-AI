#  FraudShield AI – Credit Card Fraud Detection System

##  Overview
FraudShield AI is an end-to-end machine learning web application that detects fraudulent credit card transactions using anomaly detection and gradient boosting techniques.

The system is deployed as a live fintech-style dashboard with authentication and real-time fraud scoring.

---

##  Live Demo
🔗 Live App: https://fraudshield-ai-sp.streamlit.app/
Login Credentials:
- Username: admin
- Password: admin123

---

##  Dataset
Dataset Used: Kaggle Credit Card Fraud Detection Dataset  
Total Transactions: 284,807  
Fraud Cases: 492 (Highly Imbalanced Dataset)

---

##  Machine Learning Pipeline

1. Data Preprocessing
   - Feature scaling (Time, Amount)
   - Train-test split
2. Imbalance Handling
   - SMOTE (Synthetic Minority Oversampling Technique)
3. Model
   - XGBoost Classifier (Gradient Boosting)
4. Evaluation
   - ROC-AUC (~0.97)
   - Confusion Matrix

---

## Features

- Secure Authentication
- Fraud Monitoring Dashboard
- Real-Time Transaction Simulation
- Random Transaction Generator
- Real Dataset Transaction Validator
- Cloud Deployment (Streamlit)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- SMOTE
- Streamlit
- Plotly

---

##  Deployment

The application auto-trains the ML model during first deployment and runs fully in the cloud via Streamlit Cloud.

---

##  Submitted by - Pranav S P