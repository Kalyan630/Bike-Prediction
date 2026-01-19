import streamlit as st
import joblib
import pandas as pd
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Bike Demand Prediction",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "input"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_PATH = "best_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found")
    st.stop()

model = joblib.load(MODEL_PATH)

try:
    feature_names = model.feature_names_in_
except:
    feature_names = model.named_steps["preprocessor"].feature_names_in_

SEASON_MAP = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

# ==================================================
# 📈 RESULT PAGE
# ==================================================
if st.session_state.view == "result":
    st.title("📈 Bike Demand Prediction Result")

    st.success(f"🚲 **Predicted Bike Demand:** {st.session_state.prediction:.0f}")

    st.subheader("🔍 Input Details")
    st.dataframe(st.session_state.pretty_input, use_container_width=True)

    if st.button("🔙 Back to Prediction"):
        st.session_state.view = "input"
        st.rerun()

    st.stop()

# ==================================================
# 🚲 INPUT PAGE
# ==================================================
st.title("🚲 Bike Demand Prediction")

st.write("Enter details and click **Predict** to view results on a new page.")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season", list(SEASON_MAP.keys()))
    year = st.selectbox("Year", [2011, 2012])
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 1)
    hour = st.slider("Hour", 0, 23, 12)

with col2:
    atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed", 0.0, 1.0, 0.5)
    holiday = st.selectbox("Holiday", [0, 1])
    workingday = st.selectbox("Working Day", [0, 1])
    is_weekend = st.selectbox("Weekend", [0, 1])

# --------------------------------------------------
# BUILD INPUT ROW
# --------------------------------------------------
row = {f: 0 for f in feature_names}

row["season"] = SEASON_MAP[season]
row["yr"] = 1 if year == 2012 else 0
row["mnth"] = month
row["day"] = day
row["hr"] = hour
row["atemp"] = atemp
row["windspeed"] = windspeed
row["holiday"] = holiday
row["workingday"] = workingday
row["is_weekend"] = is_weekend

input_df = pd.DataFrame([row])[feature_names]

pretty_df = input_df.copy()
pretty_df.columns = [c.replace("_", " ").title() for c in pretty_df.columns]

st.subheader("Selected Inputs")
st.dataframe(pretty_df, use_container_width=True)

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.session_state.prediction = prediction
    st.session_state.pretty_input = pretty_df
    st.session_state.view = "result"

    st.rerun()
