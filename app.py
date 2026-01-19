import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Bike Demand Prediction",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.stButton>button {
    background-color: #4F8BF9;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5em 2em;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE (SAFE – NOT WIDGET BOUND)
# ======================================================
if "show_result" not in st.session_state:
    st.session_state.show_result = False

# ======================================================
# SIDEBAR NAVIGATION (ORIGINAL – UNCHANGED)
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🚲 Bike Demand Prediction", "📊 EDA Profile Report"]
)

# ======================================================
# 📊 EDA PROFILE REPORT PAGE (ORIGINAL)
# ======================================================
if page == "📊 EDA Profile Report":
    st.title("📊 Bike Dataset – EDA Profile Report")

    html_file = "profile.html"
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=1200, scrolling=True)
    else:
        st.error("❌ profile.html not found. Please add it to the project root.")

    st.stop()

# ======================================================
# LOAD MODEL (ORIGINAL)
# ======================================================
MODEL_PATH = "best_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = model.named_steps["preprocessor"].feature_names_in_

SEASON_MAP = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

# ======================================================
# 📈 RESULT PAGE (NEW – NO SIDEBAR CHANGES)
# ======================================================
if st.session_state.show_result:
    st.title("📈 Bike Demand Prediction Result")

    st.success(f"🚲 **Predicted Bike Demand:** {st.session_state.prediction:.0f}")

    st.subheader("🔍 Input Details")
    st.dataframe(st.session_state.input_data, use_container_width=True)

    if st.button("🔙 Back"):
        st.session_state.show_result = False
        st.rerun()

    st.stop()

# ======================================================
# 🚲 MAIN PREDICTION PAGE (ORIGINAL)
# ======================================================
st.title("🚲 Bike Demand Prediction App")
st.write("Predict bike demand by entering feature values or uploading a CSV file.")

option = st.radio("**Choose input method:**", ["Manual Input", "Upload CSV"])

# Feature grouping (original)
main_features = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "atemp",
    "windspeed", "day", "is_peak_hour", "is_weekend"
]

weather_features = [
    f for f in feature_names
    if f.startswith("weathersit_") or "weather" in f
]

other_features = [
    f for f in feature_names
    if f not in main_features + weather_features
]

# ======================================================
# MANUAL INPUT (ORIGINAL)
# ======================================================
if option == "Manual Input":
    st.sidebar.header("Input Values")
    input_dict = {}

    st.sidebar.subheader("Date & Time")
    input_dict["season"] = st.sidebar.selectbox("Season", list(SEASON_MAP.keys()))
    input_dict["yr"] = st.sidebar.selectbox("Year", [2011, 2012])
    input_dict["mnth"] = st.sidebar.slider("Month", 1, 12, 6)
    input_dict["day"] = st.sidebar.slider("Day", 1, 31, 1)
    input_dict["hr"] = st.sidebar.slider("Hour", 0, 23, 12)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Flags")
    for flag in ["holiday", "weekday", "workingday", "is_peak_hour", "is_weekend"]:
        if flag in feature_names:
            input_dict[flag] = st.sidebar.selectbox(
                flag.replace("_", " ").title(), [0, 1]
            )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Weather & Conditions")
    if "atemp" in feature_names:
        input_dict["atemp"] = st.sidebar.slider("Feels Like Temp", 0.0, 1.0, 0.5)
    if "windspeed" in feature_names:
        input_dict["windspeed"] = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

    if weather_features or other_features:
        with st.sidebar.expander("Advanced Options"):
            for name in weather_features + other_features:
                if name in input_dict:
                    continue
                input_dict[name] = st.number_input(name, value=0.0)

    # Prepare model input
    input_row = {}
    for k, v in input_dict.items():
        if k == "season":
            input_row[k] = SEASON_MAP[v]
        elif k == "yr":
            input_row[k] = 1 if v == 2012 else 0
        else:
            input_row[k] = v

    for f in feature_names:
        if f not in input_row:
            input_row[f] = 0

    input_df = pd.DataFrame([input_row])[feature_names]

    pretty_df = input_df.copy()
    pretty_df.columns = [c.replace("_", " ").title() for c in pretty_df.columns]

    st.subheader("Selected Inputs")
    st.dataframe(pretty_df, use_container_width=True, height=80)

    # ==================================================
    # PREDICT BUTTON (MODIFIED ONLY HERE)
    # ==================================================
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]

        st.session_state.prediction = prediction
        st.session_state.input_data = pretty_df
        st.session_state.show_result = True

        st.rerun()

# ======================================================
# CSV UPLOAD (ORIGINAL)
# ======================================================
else:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Input Data Preview:**", df.head())

        if st.button("Predict for CSV"):
            for f in feature_names:
                if f not in df.columns:
                    df[f] = 0

            df = df[feature_names]
            preds = model.predict(df)

            st.success("Prediction Completed")
            st.dataframe(pd.DataFrame({"Prediction": preds}))

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
---
<p style='text-align:center; color:#888;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
