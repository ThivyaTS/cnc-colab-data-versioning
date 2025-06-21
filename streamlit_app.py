import streamlit as st
import pandas as pd
import time
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- SETTINGS ---
MODEL_PATH = "models/xgboost_model_v20250618_0759.pkl"
DATA_PATH = "live_data.csv"
PREDICTION_INTERVAL = 5  # seconds

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tool Condition Monitor", layout="wide")
st.title("🔧 Real-Time Tool Condition Monitoring Dashboard")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- SESSION STATE INIT ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "prediction"])
if "row_index" not in st.session_state:
    st.session_state.row_index = 0
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = time.time()
if "processed_data" not in st.session_state:
    st.session_state.processed_data = pd.DataFrame()

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

# --- RUN PREDICTION EVERY 5 SECONDS ---
elapsed = time.time() - st.session_state.last_prediction_time
if elapsed >= PREDICTION_INTERVAL and st.session_state.row_index < len(df):
    current_row = df.iloc[st.session_state.row_index:st.session_state.row_index+1].copy()

    try:
        features = current_row.drop(columns=["tool_condition"])
    except KeyError:
        st.error("❌ 'tool_condition' column missing.")
        st.stop()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    prediction = model.predict(features_scaled)[0]

    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
    st.session_state.history.loc[len(st.session_state.history)] = [timestamp, prediction]

    # Add prediction result to current row for table display
    current_row["Prediction"] = "Worn" if prediction == 1 else "Unworn"
    current_row["Timestamp"] = timestamp
    st.session_state.processed_data = pd.concat([st.session_state.processed_data, current_row], ignore_index=True)

    st.session_state.row_index += 1
    st.session_state.last_prediction_time = time.time()

# --- DISPLAY CURRENT PREDICTION ---
if len(st.session_state.history) > 0:
    latest_pred = st.session_state.history.iloc[-1]
    pred_label = "Worn" if latest_pred["prediction"] == 1 else "Unworn"
    st.subheader("🔍 Current Prediction")
    st.metric(label="Tool Condition", value=pred_label)

# --- DISPLAY PROCESSED TABLE ---
if len(st.session_state.processed_data) > 0:
    st.subheader("📋 Live Updated Data Table")
    st.dataframe(st.session_state.processed_data.tail(10), use_container_width=True)

# --- SMOOTHED CURVE CHART ---
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
