import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from evidently.metrics import DataDriftPreset
from evidently.report import Report

# --- Load models and data ---
ref_data = pd.read_csv("reference_data.csv")
live_data = pd.read_csv("live_data.csv")

if_model = joblib.load("models/isolation_forest_model_v20250621_1222.pkl")
xgb_model = joblib.load("models/xgboost_model_v20250618_0759.pkl")

# --- Sidebar Controls ---
st.sidebar.title("Controls")
pause_prediction = st.sidebar.checkbox("Pause Prediction", value=False)

# --- Helper: Alert Box ---
def alert_box(message, color="gray"):
    st.markdown(
        f"<div style='padding:10px; background-color:{color}; color:white; border-radius:10px'>{message}</div>",
        unsafe_allow_html=True,
    )

# --- Drift Detection with Evidently (PSI based) ---
def check_data_drift(reference, current_row):
    report = Report(metrics=[DataDriftPreset()])
    try:
        report.run(reference_data=reference, current_data=current_row)
        drift_result = report.as_dict()
        drifted = drift_result["metrics"][0]["result"]["dataset_drift"]
        drifted_cols = [
            f["column_name"]
            for f in drift_result["metrics"][0]["result"]["drift_by_columns"].values()
            if f["drift_detected"]
        ]
        return drifted, drifted_cols
    except Exception as e:
        return False, []

# --- Streamlit App Header ---
st.title("🛠️ Tool Wear Monitoring Dashboard")
st.markdown("This dashboard streams live data, predicts tool wear, detects anomalies and monitors for data drift.")

# --- Process Each Row in Live Data ---
for idx in range(len(live_data)):
    if pause_prediction:
        alert_box("⏸️ Prediction paused.", "gray")
        break

    row = live_data.iloc[[idx]].copy()
    input_features = row.drop(columns=["Tool_Condition"], errors="ignore")

    # --- Predict Tool Wear ---
    wear_prediction = xgb_model.predict(input_features)[0]
    wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
    wear_color = "red" if wear_prediction == 1 else "green"
    alert_box(f"Tool Wear Prediction: {wear_label}", wear_color)

    # --- Anomaly Detection ---
    anomaly_result = if_model.predict(input_features)[0]
    if anomaly_result == -1:
        alert_box("⚠️ Anomaly Detected!", "orange")

    # --- Drift Detection ---
    drift_detected, drift_columns = check_data_drift(ref_data, row)
    if drift_detected:
        drift_col_str = ', '.join(drift_columns) if drift_columns else 'Unknown columns'
        alert_box(f"📊 Data Drift Detected in: {drift_col_str}", "blue")

    # --- Display Incoming Row ---
    with st.expander(f"📄 Incoming Data Row {idx+1}"):
        st.dataframe(row.reset_index(drop=True))

    # --- Simulate Real-time Delay ---
    time.sleep(1)

st.success("✅ Live data stream completed.")
