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
REFRESH_INTERVAL = 5  # seconds

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tool Condition Monitor", layout="wide")
st.title("🔧 Real-Time Tool Condition Monitoring Dashboard")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "prediction"])
if "last_seen_index" not in st.session_state:
    st.session_state.last_seen_index = -1

# --- LOAD DATA FUNCTION ---
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

# --- LOAD & CHECK DATA ---
df = load_data()
if df is None or len(df) == 0:
    st.warning("⏳ Waiting for live data...")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# --- NEW ROWS SINCE LAST SEEN ---
new_rows = df.iloc[st.session_state.last_seen_index + 1:]
if new_rows.empty:
    st.info("✅ No new data yet...")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# --- TABLE OF NEW DATA ---
st.subheader("📋 New Incoming Data")
st.dataframe(new_rows, use_container_width=True)

# --- PROCESS EACH NEW ROW ---
for idx, row in new_rows.iterrows():
    latest_row = row.to_frame().T
    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')

    # Drop label to predict
    try:
        features = latest_row.drop(columns=["tool_condition"])
    except KeyError:
        st.error("❌ 'tool_condition' column missing.")
        st.stop()

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Save to history
    st.session_state.history.loc[len(st.session_state.history)] = [timestamp, prediction]
    st.session_state.last_seen_index = idx  # update last seen

# --- SHOW CURRENT PREDICTION ---
latest_pred = st.session_state.history.iloc[-1]
pred_label = "Worn" if latest_pred["prediction"] == 1 else "Unworn"
st.subheader("🔍 Current Prediction")
st.metric(label="Tool Condition", value=pred_label)

# --- HISTORY VISUALIZATION WITH SMOOTH CURVE ---
st.subheader("📈 Prediction History (Fitted Curve)")
history_df = st.session_state.history.copy()
history_df["idx"] = range(len(history_df))

# Smooth line with LOWESS
lowess = sm.nonparametric.lowess
smoothed = lowess(history_df["prediction"], history_df["idx"], frac=0.3)

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
time.sleep(REFRESH_INTERVAL)
st.rerun()
