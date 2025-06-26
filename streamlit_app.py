import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
import base64
from data_drift_detector.detectors import PSIDriftDetector

# --- Streamlit Config ---
st.set_page_config(layout="wide")

# --- Background Image with Blur and Overlay ---
def set_background(image_path, blur_px=6, overlay_opacity=0.7):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, {overlay_opacity}),
            rgba(0, 0, 0, {overlay_opacity})
        ), url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        backdrop-filter: blur({blur_px}px);
        -webkit-backdrop-filter: blur({blur_px}px);
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

set_background("image1.png", blur_px=6, overlay_opacity=0.7)

# --- Load Data and Model ---
live_data = pd.read_csv("live_data.csv")
reference_data = pd.read_csv("reference_data.csv")
xgb_model = joblib.load(os.path.join("models", "xgboost_model_v20250618_0759.pkl"))

# --- Expected Features ---
EXPECTED_FEATURES = [
    "Y1_OutputCurrent", "X1_CommandPosition", "X1_ActualPosition", "clamp_pressure",
    "Y1_CommandPosition", "Y1_ActualPosition", "X1_OutputCurrent", "X1_DCBusVoltage",
    "X1_OutputVoltage", "Z1_CommandPosition", "Z1_ActualPosition", "M1_CURRENT_FEEDRATE",
    "X1_OutputPower", "Y1_OutputVoltage", "S1_OutputCurrent", "Y1_DCBusVoltage",
    "feedrate", "Y1_OutputPower", "S1_CurrentFeedback", "S1_ActualVelocity"
]

# --- Initialize Session State ---
if "observed_count" not in st.session_state:
    st.session_state.observed_count = 0
if "current_wear_label" not in st.session_state:
    st.session_state.current_wear_label = "Unknown"
if "last_wear_prediction" not in st.session_state:
    st.session_state.last_wear_prediction = 0
if "psi_score" not in st.session_state:
    st.session_state.psi_score = 0.0
if "feature_series" not in st.session_state:
    st.session_state.feature_series = []
if "cmd_pos_series" not in st.session_state:
    st.session_state.cmd_pos_series = []
if "actual_pos_series" not in st.session_state:
    st.session_state.actual_pos_series = []
if "y1_cmd_series" not in st.session_state:
    st.session_state.y1_cmd_series = []

# --- Email Alert Function ---
def send_email_alert(subject="Tool Wear Alert", message="Tool condition has changed to WORN. Immediate maintenance is recommended."):
    EMAIL_ADDRESS = "m032410022@student.utem.edu.my"
    EMAIL_PASSWORD = "Thanilparsad12???"  # Replace with your app password
    TO_EMAIL = "m032410022@student.utem.edu.my"

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(message)

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("Email sent successfully")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# --- PSI Drift Function using data-drift-detector ---
def calculate_psi_drift(current_row_df):
    try:
        detector = PSIDriftDetector()
        detector.fit(reference_data[EXPECTED_FEATURES])
        psi_scores = detector.detect(current_row_df[EXPECTED_FEATURES])
        return psi_scores["overall"]["psi"]
    except Exception as e:
        st.error(f"Drift detection error: {e}")
        return 0.0

# --- App Header ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("Tool Wear Monitoring Dashboard")
with header_col2:
    st.button("Next Observation")

# --- Process Current Observation ---
if st.session_state.observed_count < len(live_data):
    row = live_data.iloc[st.session_state.observed_count]
    st.session_state.feature_series.append(row["X1_OutputCurrent"])
    st.session_state.cmd_pos_series.append(row["X1_CommandPosition"])
    st.session_state.actual_pos_series.append(row["X1_ActualPosition"])
    st.session_state.y1_cmd_series.append(row["Y1_CommandPosition"])

    try:
        input_features = row[EXPECTED_FEATURES].values.reshape(1, -1)
        wear_prediction = xgb_model.predict(input_features)[0]
        wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
        st.session_state.current_wear_label = wear_label

        # --- Tool Condition Changed Alert ---
        if st.session_state.last_wear_prediction == 0 and wear_prediction == 1:
            st.warning("⚠️ Tool Condition Changed. Maintenance Alert. Sent Email.")
            send_email_alert()

        st.session_state.last_wear_prediction = wear_prediction

        # --- Data Drift Detection ---
        current_row_df = row.to_frame().T
        psi_score = calculate_psi_drift(current_row_df)
        st.session_state.psi_score = psi_score

        if psi_score > 0.5:
            st.warning("⚠️ Data Drift Detected. Email Sent.")
            send_email_alert(
                subject="Data Drift Alert",
                message="⚠️ Drift detected in sensor input. PSI > 50%. Please investigate."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

    st.session_state.observed_count += 1

# --- Metrics Display ---
st.subheader("🔍 Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of Data Observed", f"{st.session_state.observed_count}")
with col2:
    st.metric("Total Number of Features", f"{len(EXPECTED_FEATURES)}")
with col3:
    st.metric("Tool Wear Condition", f"{st.session_state.current_wear_label}")
with col4:
    st.metric("Data Drift (%)", f"{round(st.session_state.psi_score * 100, 2)}%")

# --- Visualization Helper ---
def plot_series(series, title, ylabel, color, marker):
    fig, ax = plt.subplots()
    ax.plot(series, color=color, marker=marker)
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    fig.tight_layout()
    return fig

# --- Feature Visualizations ---
st.subheader("📈 Live Feature Visualizations")
r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    st.pyplot(plot_series(st.session_state.feature_series, "X1 Output Current", "X1_OutputCurrent", "steelblue", "o"))
with r1c2:
    st.pyplot(plot_series(st.session_state.cmd_pos_series, "X1 Command Position", "X1_CommandPosition", "darkgreen", "x"))
with r2c1:
    st.pyplot(plot_series(st.session_state.actual_pos_series, "X1 Actual Position", "X1_ActualPosition", "crimson", "s"))
with r2c2:
    st.pyplot(plot_series(st.session_state.y1_cmd_series, "Y1 Command Position", "Y1_CommandPosition", "orange", "^"))
