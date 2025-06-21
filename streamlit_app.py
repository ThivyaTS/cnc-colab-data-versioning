import streamlit as st
import pandas as pd
import time
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# --- SETTINGS ---
MODEL_PATH = "models/xgboost_model_v20250618_0759.pkl"
DRIFT_REFERENCE_PATH = "reference_data.csv"
ANOMALY_MODEL_PATH = "models/isolation_forest_model_v20250621_1222.pkl"
DATA_PATH = "live_data.csv"
PREDICTION_INTERVAL = 5  # seconds
BATCH_SIZE = 5  # Check drift every 5 rows

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tool Condition Monitor", layout="wide")
st.title("🔧 Real-Time Tool Condition Monitoring Dashboard")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_PATH)
    iso = joblib.load(ANOMALY_MODEL_PATH)
    return clf, iso

model, iso_model = load_models()

# --- LOAD REFERENCE DATA FOR DRIFT ---
@st.cache_data
def load_reference_data():
    df_ref = pd.read_csv(DRIFT_REFERENCE_PATH)
    return df_ref.describe()

ref_stats = load_reference_data()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "prediction"])
if "row_index" not in st.session_state:
    st.session_state.row_index = 0
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = time.time()
if "table_data" not in st.session_state:
    st.session_state.table_data = pd.DataFrame()
if "paused" not in st.session_state:
    st.session_state.paused = False
if "drift_detected" not in st.session_state:
    st.session_state.drift_detected = False
if "anomaly_detected" not in st.session_state:
    st.session_state.anomaly_detected = False

# --- SIDEBAR ALERTS & PAUSE CONTROL ---
st.sidebar.header("🚨 System Status")
st.sidebar.toggle("⏸️ Manually Pause Prediction", key="paused")
if st.session_state.drift_detected:
    st.sidebar.error("⚠️ Data Drift Detected! Prediction paused.")
if st.session_state.anomaly_detected:
    st.sidebar.error("🚨 Anomaly Detected! Prediction paused.")
if st.session_state.paused and not (st.session_state.drift_detected or st.session_state.anomaly_detected):
    st.sidebar.warning("⏸️ Manually paused.")
if st.session_state.paused:
    if st.sidebar.button("▶️ Resume Prediction"):
        st.session_state.paused = False
        st.session_state.drift_detected = False
        st.session_state.anomaly_detected = False

# --- LOAD DATA ---
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_data()
if df is None or len(df) == 0:
    st.warning("⏳ Waiting for live data...")
    time.sleep(1)
    st.rerun()

# --- PAUSE CHECK ---
elapsed = time.time() - st.session_state.last_prediction_time
if not st.session_state.paused and elapsed >= PREDICTION_INTERVAL and st.session_state.row_index < len(df):
    current_row = df.iloc[st.session_state.row_index:st.session_state.row_index+1].copy()

    try:
        features = current_row.drop(columns=["tool_condition"])
    except KeyError:
        st.error("❌ 'tool_condition' column missing.")
        st.stop()

    # --- DRIFT CHECK ---
    if st.session_state.row_index % BATCH_SIZE == 0:
        recent_batch = df.iloc[max(0, st.session_state.row_index - BATCH_SIZE + 1):st.session_state.row_index + 1]
        if not recent_batch.empty:
            for col in features.columns:
                if col in ref_stats.index:
                    ref_mean = ref_stats.at["mean", col]
                    batch_mean = recent_batch[col].mean()
                    drift_ratio = abs(batch_mean - ref_mean) / (ref_mean + 1e-6)
                    if drift_ratio > 0.2:
                        st.session_state.paused = True
                        st.session_state.drift_detected = True

    # --- ANOMALY DETECTION ---
    anomaly_result = iso_model.predict(features)
    if anomaly_result[0] == -1:
        st.session_state.paused = True
        st.session_state.anomaly_detected = True

    # --- CONTINUE ONLY IF STILL UNPAUSED ---
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

# --- DISPLAY TABLE ---
if not st.session_state.table_data.empty:
    st.subheader("📋 Live Processed Data Table")
    st.dataframe(st.session_state.table_data, use_container_width=True)

# --- DISPLAY LATEST PREDICTION ---
if len(st.session_state.history) > 0:
    latest_pred = st.session_state.history.iloc[-1]
    pred_label = "Worn" if latest_pred["prediction"] == 1 else "Unworn"
    st.subheader("🔍 Current Prediction")
    st.metric(label="Tool Condition", value=pred_label)

# --- PREDICTION GRAPH ---
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
