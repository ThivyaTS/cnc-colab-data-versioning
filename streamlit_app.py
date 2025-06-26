import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Load models and data ---
live_data = pd.read_csv("live_data.csv")

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

header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("🛠️ Tool Wear Monitoring Dashboard")
with header_col2:
    st.button("🔁 Next Observation")

# --- Initialize session state ---
if "observed_count" not in st.session_state:
    st.session_state.observed_count = 0
if "current_wear_label" not in st.session_state:
    st.session_state.current_wear_label = "Unknown"
if "feature_series" not in st.session_state:
    st.session_state.feature_series = []
if "cmd_pos_series" not in st.session_state:
    st.session_state.cmd_pos_series = []
if "actual_pos_series" not in st.session_state:
    st.session_state.actual_pos_series = []
if "y1_cmd_series" not in st.session_state:
    st.session_state.y1_cmd_series = []

# --- Dashboard Metrics ---
st.subheader("🔍 Summary")
st.markdown(f"**Number of Data Observed:** {st.session_state.observed_count}")
st.markdown(f"**Total Number of Features:** {len(EXPECTED_FEATURES)}")
st.markdown(f"**Current Tool Wear Condition:** {st.session_state.current_wear_label}")

# --- Feature Visualization - X1_OutputCurrent ---
st.subheader("📈 Feature Visualization - X1_OutputCurrent")
if st.session_state.observed_count < len(live_data):
    row = live_data.loc[st.session_state.observed_count]
    st.session_state.feature_series.append(row["X1_OutputCurrent"])
    st.session_state.cmd_pos_series.append(row["X1_CommandPosition"])
    st.session_state.actual_pos_series.append(row["X1_ActualPosition"])
    st.session_state.y1_cmd_series.append(row["Y1_CommandPosition"])

fig, ax = plt.subplots()
ax.plot(st.session_state.feature_series, marker='o', color='steelblue')
ax.set_xlabel("Observation")
ax.set_ylabel("X1_OutputCurrent")
ax.set_title("Live Update of X1_OutputCurrent")
st.pyplot(fig)

# --- Additional Visualizations ---
st.subheader("📊 Position Comparisons")

graph_col1, graph_col2, graph_col3 = st.columns(3)

with graph_col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(st.session_state.cmd_pos_series, color='darkgreen', marker='x')
    ax1.set_title("X1 Command Position")
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("X1_CommandPosition")
    fig1.patch.set_edgecolor('black')
    fig1.patch.set_linewidth(1.5)
    st.pyplot(fig1)

with graph_col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(st.session_state.actual_pos_series, color='crimson', marker='s')
    ax2.set_title("X1 Actual Position")
    ax2.set_xlabel("Observation")
    ax2.set_ylabel("X1_ActualPosition")
    fig2.patch.set_edgecolor('black')
    fig2.patch.set_linewidth(1.5)
    st.pyplot(fig2)

with graph_col3:
    fig3, ax3 = plt.subplots()
    ax3.plot(st.session_state.y1_cmd_series, color='orange', marker='^')
    ax3.set_title("Y1 Command Position")
    ax3.set_xlabel("Observation")
    ax3.set_ylabel("Y1_CommandPosition")
    fig3.patch.set_edgecolor('black')
    fig3.patch.set_linewidth(1.5)
    st.pyplot(fig3)

# --- Process New Row ---
if st.session_state.observed_count < len(live_data):
    row = live_data.iloc[[st.session_state.observed_count]].copy()
    try:
        input_features = row[EXPECTED_FEATURES]
    except KeyError as e:
        st.error(f"Missing expected feature(s): {e}")
    else:
        wear_prediction = xgb_model.predict(input_features)[0]
        wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
        st.session_state.current_wear_label = wear_label
        st.session_state.observed_count += 1
