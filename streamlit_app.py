import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt

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

# --- Streamlit App Header ---
st.set_page_config(layout="wide")
st.title("🛠️ Tool Wear Monitoring Dashboard")

# --- Initialize session state ---
if "observed_count" not in st.session_state:
    st.session_state.observed_count = 0
if "anomaly_data" not in st.session_state:
    st.session_state.anomaly_data = pd.DataFrame(columns=live_data.columns)
if "current_wear_label" not in st.session_state:
    st.session_state.current_wear_label = "Unknown"
if "feature_series" not in st.session_state:
    st.session_state.feature_series = []

# --- Dashboard Metrics ---
st.subheader("🔍 Summary")
st.markdown(f"**Number of Data Observed:** {st.session_state.observed_count}")
st.markdown(f"**Total Number of Features:** {len(EXPECTED_FEATURES)}")
st.markdown(f"**Current Tool Wear Condition:** {st.session_state.current_wear_label}")

# --- Anomaly Table ---
st.subheader("⚠️ Anomaly Detected Rows")
st.dataframe(st.session_state.anomaly_data.reset_index(drop=True), use_container_width=True)

# --- Feature Visualization ---
st.subheader("📈 Feature Visualization - X1_OutputCurrent")
st.session_state.feature_series.append(live_data.loc[st.session_state.observed_count, "X1_OutputCurrent"])
fig, ax = plt.subplots()
ax.plot(st.session_state.feature_series, marker='o')
ax.set_xlabel("Observation")
ax.set_ylabel("X1_OutputCurrent")
ax.set_title("Live Update of X1_OutputCurrent")
st.pyplot(fig)

# --- Process New Row ---
if st.session_state.observed_count < len(live_data):
    row = live_data.iloc[[st.session_state.observed_count]].copy()
    try:
        input_features = row[EXPECTED_FEATURES]
    except KeyError as e:
        st.error(f"Missing expected feature(s): {e}")
    else:
        # Predict Tool Wear
        wear_prediction = xgb_model.predict(input_features)[0]
        wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
        st.session_state.current_wear_label = wear_label

        # Anomaly Detection
        anomaly_result = if_model.predict(input_features)[0]
        if anomaly_result == -1:
            st.session_state.anomaly_data = pd.concat([st.session_state.anomaly_data, row], ignore_index=True)

        st.session_state.observed_count += 1

st.button("🔁 Next Observation")
