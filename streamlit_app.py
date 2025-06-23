import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- Load models and data ---
live_data = pd.read_csv("live_data.csv")

if_model = joblib.load(os.path.join("models", "isolation_forest_model_v20250621_1222.pkl"))
xgb_model = joblib.load(os.path.join("models", "xgboost_model_v20250618_0759.pkl"))

# --- Define expected features ---
EXPECTED_FEATURES = [
    "Y1_OutputCurrent",
    "X1_CommandPosition",
    "X1_ActualPosition",
    "clamp_pressure",
    "Y1_CommandPosition",
    "Y1_ActualPosition",
    "X1_OutputCurrent",
    "X1_DCBusVoltage",
    "X1_OutputVoltage",
    "Z1_CommandPosition",
    "Z1_ActualPosition",
    "M1_CURRENT_FEEDRATE",
    "X1_OutputPower",
    "Y1_OutputVoltage",
    "S1_OutputCurrent",
    "Y1_DCBusVoltage",
    "feedrate",
    "Y1_OutputPower",
    "S1_CurrentFeedback",
    "S1_ActualVelocity"
]

# --- Sidebar Controls ---
st.sidebar.title("Controls")
pause_prediction = st.sidebar.checkbox("Pause Prediction", value=False)

# --- Helper: Alert Box ---
def alert_box(message, color="gray"):
    st.markdown(
        f"<div style='padding:10px; background-color:{color}; color:white; border-radius:10px'>{message}</div>",
        unsafe_allow_html=True,
    )

# --- Streamlit App Header ---
st.title("🛠️ Tool Wear Monitoring Dashboard")
st.markdown("This dashboard streams live data, predicts tool wear, and detects anomalies.")

# --- Process Each Row in Live Data ---
for idx in range(len(live_data)):
    if pause_prediction:
        alert_box("⏸️ Prediction paused.", "gray")
        break

    row = live_data.iloc[[idx]].copy()
    try:
        input_features = row[EXPECTED_FEATURES]
    except KeyError as e:
        st.error(f"Missing expected feature(s): {e}")
        break

    # --- Predict Tool Wear ---
    wear_prediction = xgb_model.predict(input_features)[0]
    wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
    wear_color = "red" if wear_prediction == 1 else "green"
    alert_box(f"Tool Wear Prediction: {wear_label}", wear_color)

    # --- Anomaly Detection ---
    anomaly_result = if_model.predict(input_features)[0]
    if anomaly_result == -1:
        alert_box("⚠️ Anomaly Detected!", "orange")

    # --- Display Incoming Row ---
    with st.expander(f"📄 Incoming Data Row {idx+1}"):
        st.dataframe(row.reset_index(drop=True))

    # --- Simulate Real-time Delay ---
    time.sleep(1)

st.success("✅ Live data stream completed.")
