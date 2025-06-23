import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

# Load models and reference data
ref_data = pd.read_csv('reference_data.csv')
live_data = pd.read_csv('live_data.csv')
if_model = joblib.load('isolation_forest_model_v20250621_1222.pkl')
xgb_model = joblib.load('xgboost_model_v20250618_0759.pkl')

# Sidebar controls
st.sidebar.title("Controls")
pause_prediction = st.sidebar.checkbox("Pause Prediction")

# Alert box function
def alert_box(msg, color):
    st.markdown(f"<div style='padding:10px; background-color:{color}; color:white; border-radius:10px'>{msg}</div>", unsafe_allow_html=True)

# Drift detection using JS divergence
def js_divergence(p, q):
    p = np.array(p) + 1e-8
    q = np.array(q) + 1e-8
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def detect_drift(ref_df, live_df, threshold=0.1):
    drift_results = {}
    for col in ref_df.columns:
        if ref_df[col].dtype != 'object':
            ref_hist = np.histogram(ref_df[col], bins=10, density=True)[0]
            live_hist = np.histogram(live_df[col], bins=10, density=True)[0]
            js_score = js_divergence(ref_hist, live_hist)
            drift_results[col] = js_score > threshold
    return drift_results

# Main dashboard
st.title("Tool Wear Monitoring Dashboard")

# Simulate streaming by reading live data row-by-row
for idx in range(len(live_data)):
    if pause_prediction:
        alert_box("Prediction is paused.", "gray")
        break

    current_row = live_data.iloc[[idx]]

    # --- Predict Tool Wear ---
    wear_pred = xgb_model.predict(current_row.drop(columns=['Tool_Condition'], errors='ignore'))
    wear_status = 'Worn' if wear_pred[0] == 1 else 'Unworn'
    color = "red" if wear_status == 'Worn' else "green"
    st.subheader(f"Row {idx+1} Tool Wear Status: ")
    alert_box(f"Predicted Tool Condition: {wear_status}", color)

    # --- Detect Anomaly ---
    anomaly_score = if_model.decision_function(current_row)
    is_anomaly = if_model.predict(current_row)[0] == -1
    if is_anomaly:
        alert_box("Anomaly Detected!", "orange")

    # --- Data Drift Detection ---
    drift_result = detect_drift(ref_data, current_row)
    drifted_features = [k for k, v in drift_result.items() if v]
    if drifted_features:
        alert_box(f"Data Drift Detected in: {', '.join(drifted_features)}", "blue")

    # --- Show Incoming Data ---
    st.dataframe(current_row.reset_index(drop=True))

    # Delay for simulation
    time.sleep(1)

st.success("End of Live Data Stream.")
