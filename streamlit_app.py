import streamlit as st
import pandas as pd
import time
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import entropy

# --- SETTINGS ---
MODEL_PATH = "models/xgboost_model_v20250618_0759.pkl"
ANOMALY_MODEL_PATH = "models/isolation_forest_model_v20250621_1222.pkl"
REFERENCE_PATH = "reference_data.csv"
DATA_PATH = "live_data.csv"
PREDICTION_INTERVAL = 5  # seconds
PSI_THRESHOLD = 0.25  # for drift detection

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tool Condition Monitor", layout="wide")
st.title("🔧 Real-Time Tool Condition Monitoring Dashboard")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    clf_model = joblib.load(MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    return clf_model, anomaly_model

model, anomaly_detector = load_models()

# --- LOAD REFERENCE DATA FOR DRIFT CHECK ---
@st.cache_data
def load_reference():
    df = pd.read_csv(REFERENCE_PATH)
    return df.drop(columns=["tool_condition"])

reference_data = load_reference()

# --- PSI DRIFT CHECK ---
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / np.sum(expected_counts)
    actual_percents = actual_counts / np.sum(actual_counts)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def is_data_drifted(ref_df, incoming_df):
    drift_scores = {
        col: calculate_psi(ref_df[col], incoming_df[col])
        for col in ref_df.columns
    }
    drift_detected = any(score > PSI_THRESHOLD for score in drift_scores.values())
    return drift_detected, drift_scores

# --- SESSION STATE ---
if "paused" not in st.session_state:
    st.session_state.paused = False
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "prediction"])
if "row_index" not in st.session_state:
    st.session_state.row_index = 0
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = time.time()
if "table_data" not in st.session_state:
    st.session_state.table_data = pd.DataFrame()

# --- LOAD LIVE DATA ---
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_data()

if df is None or len(df) == 0:
    st.warning("⏳ Waiting for live data...")
    time.sleep(1)
    st.rerun()

# --- SIDEBAR CONTROL ---
st.sidebar.header("⚙️ Controls")
pause_toggle = st.sidebar.checkbox("Pause Prediction", value=st.session_state.paused)
if pause_toggle != st.session_state.paused:
    st.session_state.paused = pause_toggle

# --- ALERT STATE ---
drift_alert = False
anomaly_alert = False

# --- PREDICTION LOGIC ---
elapsed = time.time() - st.session_state.last_prediction_time
if not st.session_state.paused and elapsed >= PREDICTION_INTERVAL and st.session_state.row_index < len(df):
    current_row = df.iloc[st.session_state.row_index:st.session_state.row_index+1].copy()
    try:
        features = current_row.drop(columns=["tool_condition"])
    except KeyError:
        st.error("❌ 'tool_condition' column missing.")
        st.stop()

    # --- Drift Check on Batch (1 row in this case) ---
    drift_alert, drift_scores = is_data_drifted(reference_data, features)

    # --- Anomaly Detection ---
    anomaly = anomaly_detector.predict(features)[0]
    anomaly_alert = anomaly == -1

    if drift_alert or anomaly_alert:
        st.session_state.paused = True

    # --- Predict if OK ---
    if not st.session_state.paused:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        prediction = model.predict(features_scaled)[0]

        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        st.session_state.history.loc[len(st.session_state.history)] = [timestamp, prediction]

        display_row = current_row.copy()
        display_row["Predicted Condition"] = "Worn" if prediction == 1 else "Unworn"
        display_row["Timestamp"] = timestamp
        st.session_state.table_data = pd.concat([st.session_state.table_data, display_row], ignore_index=True)

        st.session_state.row_index += 1
        st.session_state.last_prediction_time = time.time()

# --- ALERT MESSAGES ---
st.sidebar.header("🚨 System Alerts")
if drift_alert:
    st.sidebar.error("⚠️ Data Drift Detected — Prediction Paused")
if anomaly_alert:
    st.sidebar.error("⚠️ Anomaly Detected — Prediction Paused")
if not drift_alert and not anomaly_alert and not st.session_state.paused:
    st.sidebar.success("✅ System Healthy")

# --- DISPLAY TABLE ---
if not st.session_state.table_data.empty:
    st.subheader("📋 Live Processed Data Table")
    st.dataframe(st.session_state.table_data, use_container_width=True)

# --- DISPLAY CURRENT PREDICTION ---
if len(st.session_state.history) > 0:
    latest_pred = st.session_state.history.iloc[-1]
    pred_label = "Worn" if latest_pred["prediction"] == 1 else "Unworn"
    st.subheader("🔍 Current Prediction")
    st.metric(label="Tool Condition", value=pred_label)

# --- VISUALIZATION ---
if len(st.session_state.history) > 0:
    st.subheader("📈 Prediction History (Smoothed)")
    history_df = st.session_state.history.copy()
    history_df["idx"] = range(len(history_df))

    lowess = sm.nonparametric.lowess
    smoothed = lowess(history_df["prediction"], history_df["idx"], frac=0.4)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history_df["idx"], history_df["prediction"], 'o', alpha=0.5, label="Raw Prediction")
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label="Smoothed Curve")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unworn", "Worn"])
    ax.set_xlabel("Time (Prediction Steps)")
    ax.set_title("Tool Condition Over Time")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- AUTO REFRESH ---
time.sleep(1)
st.rerun()
