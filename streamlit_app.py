import streamlit as st
import pandas as pd
import time
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- SETTINGS ---
MODEL_PATH = "models/xgboost_model_v20250618_0759.pkl"
DATA_PATH = "live_data.csv"  # This should be updated with new rows over time
REFRESH_INTERVAL = 1  # seconds

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tool Condition Monitor", layout="wide")
st.title("🔧 Real-Time Tool Condition Monitoring Dashboard")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- SESSION STATE FOR PERSISTENT LIVE VIEW ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "prediction"])

# --- LOAD AND PREPROCESS DATA ---
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_data()

# --- WAIT IF DATA NOT AVAILABLE ---
if df is None or len(df) == 0:
    st.warning("⏳ Waiting for `live_data.csv` file to appear or be populated...")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# --- SHOW LIVE DATA TABLE ---
st.subheader("📋 Incoming Live Data")
st.dataframe(df.tail(10), use_container_width=True)  # Show only the last 10 rows for clarity

# --- SELECT LATEST ROW ---
latest_row = df.iloc[-1:].copy()

# --- SCALE FEATURES ---
try:
    features = latest_row.drop(columns=["tool_condition"])
except KeyError:
    st.error("❌ 'tool_condition' column not found in data.")
    st.stop()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- MAKE PREDICTION ---
prediction = model.predict(features_scaled)[0]
pred_label = "Worn" if prediction == 1 else "Unworn"

# --- SAVE TO HISTORY ---
timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
st.session_state.history.loc[len(st.session_state.history)] = [timestamp, prediction]

# --- DISPLAY CURRENT PREDICTION ---
st.subheader("🔍 Current Prediction")
st.metric(label="Tool Condition", value=pred_label)

# --- DISPLAY PREDICTION HISTORY ---
st.subheader("📈 Prediction Over Time")
history_df = st.session_state.history

fig, ax = plt.subplots()
ax.plot(history_df["timestamp"], history_df["prediction"], marker='o', linestyle='-')
ax.set_ylabel("Prediction (0=Unworn, 1=Worn)")
ax.set_xlabel("Time")
ax.set_ylim(-0.2, 1.2)
ax.grid(True)
st.pyplot(fig)

# --- AUTO-REFRESH ---
time.sleep(REFRESH_INTERVAL)
st.rerun()
