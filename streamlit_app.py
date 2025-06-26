import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

# --- Load models and data ---
live_data = pd.read_csv("live_data.csv")
xgb_model = joblib.load(os.path.join("models", "xgboost_model_v20250618_0759.pkl"))

# --- Define expected features ---
EXPECTED_FEATURES = [
    "Y1_OutputCurrent", "X1_CommandPosition", "X1_ActualPosition", "clamp_pressure",
    "Y1_CommandPosition", "Y1_ActualPosition", "X1_OutputCurrent", "X1_DCBusVoltage",
    "X1_OutputVoltage", "Z1_CommandPosition", "Z1_ActualPosition", "M1_CURRENT_FEEDRATE",
    "X1_OutputPower", "Y1_OutputVoltage", "S1_OutputCurrent", "Y1_DCBusVoltage",
    "feedrate", "Y1_OutputPower", "S1_CurrentFeedback", "S1_ActualVelocity"
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
if "last_wear_prediction" not in st.session_state:
    st.session_state.last_wear_prediction = 0  # Assume UNWORN initially

# --- Email function ---
def send_email_alert():
    EMAIL_ADDRESS = "your_outlook_email@example.com"  # Replace with your Outlook email
    EMAIL_PASSWORD = "your_password_or_app_password"  # Use app password if MFA enabled
    TO_EMAIL = "recipient@example.com"  # Replace with maintenance recipient

    msg = EmailMessage()
    msg['Subject'] = "Tool Wear Alert - Maintenance Required"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content("The tool condition has changed to WORN. Immediate maintenance is recommended.")

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("Email sent successfully")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# --- Process incoming row ---
if st.session_state.observed_count < len(live_data):
    row = live_data.loc[st.session_state.observed_count]
    st.session_state.feature_series.append(row["X1_OutputCurrent"])
    st.session_state.cmd_pos_series.append(row["X1_CommandPosition"])
    st.session_state.actual_pos_series.append(row["X1_ActualPosition"])
    st.session_state.y1_cmd_series.append(row["Y1_CommandPosition"])

    try:
        input_features = row[EXPECTED_FEATURES].values.reshape(1, -1)
        wear_prediction = xgb_model.predict(input_features)[0]
        wear_label = "🟥 WORN" if wear_prediction == 1 else "🟩 UNWORN"
        st.session_state.current_wear_label = wear_label

        # Trigger maintenance alert if tool becomes worn
        if st.session_state.last_wear_prediction == 0 and wear_prediction == 1:
            st.warning("⚠️ Tool Condition Changed. Maintenance Alert. Sent Email.")
            send_email_alert()

        st.session_state.last_wear_prediction = wear_prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    st.session_state.observed_count += 1

# --- Metrics Display ---
st.subheader("🔍 Summary")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Number of Data Observed", f"**{st.session_state.observed_count}**")
with metric_col2:
    st.metric("Total Number of Features", f"**{len(EXPECTED_FEATURES)}**")
with metric_col3:
    st.metric("Current Tool Wear Condition", f"**{st.session_state.current_wear_label}**")

# --- Plot helper ---
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
st.subheader("📊 Live Feature Visualizations")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    fig1 = plot_series(
        st.session_state.feature_series,
        "X1 Output Current",
        "X1_OutputCurrent",
        "steelblue",
        "o"
    )
    st.pyplot(fig1)

with row1_col2:
    fig2 = plot_series(
        st.session_state.cmd_pos_series,
        "X1 Command Position",
        "X1_CommandPosition",
        "darkgreen",
        "x"
    )
    st.pyplot(fig2)

with row2_col1:
    fig3 = plot_series(
        st.session_state.actual_pos_series,
        "X1 Actual Position",
        "X1_ActualPosition",
        "crimson",
        "s"
    )
    st.pyplot(fig3)

with row2_col2:
    fig4 = plot_series(
        st.session_state.y1_cmd_series,
        "Y1 Command Position",
        "Y1_CommandPosition",
        "orange",
        "^"
    )
    st.pyplot(fig4)
