import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config("Bike Demand Prediction", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "🚲 Bike Demand Prediction"

# --------------------------------------------------
# SIDEBAR NAVIGATION (FIXED)
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🚲 Bike Demand Prediction", "📊 EDA Profile Report", "📈 Prediction Result"],
    key="page"
)

# --------------------------------------------------
# EDA PAGE
# --------------------------------------------------
if page == "📊 EDA Profile Report":
    st.title("📊 Bike Dataset – EDA Profile Report")

    if os.path.exists("profile.html"):
        with open("profile.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=1200, scrolling=True)
    else:
        st.error("profile.html not found")

    st.stop()

# --------------------------------------------------
# RESULT PAGE
# --------------------------------------------------
if page == "📈 Prediction Result":
    st.title("📈 Prediction Result")

    if "prediction" not in st.session_state:
        st.warning("No prediction yet.")
    else:
        st.success(f"🚲 Predicted Bike Demand: {st.session_state.prediction:.0f}")
        st.dataframe(st.session_state.input_data)

    if st.button("🔙 Back"):
        st.session_state.page = "🚲 Bike Demand Prediction"
        st.rerun()

    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("best_model.pkl")
feature_names = model.feature_names_in_

# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------
st.title("🚲 Bike Demand Prediction")

SEASON_MAP = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

season = st.selectbox("Season", list(SEASON_MAP.keys()))
year = st.selectbox("Year", [2011, 2012])
hour = st.slider("Hour", 0, 23, 12)
atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)

row = {f: 0 for f in feature_names}
row["season"] = SEASON_MAP[season]
row["yr"] = 1 if year == 2012 else 0
row["hr"] = hour
row["atemp"] = atemp

df = pd.DataFrame([row])[feature_names]

st.dataframe(df)

if st.button("Predict"):
    pred = model.predict(df)[0]

    st.session_state.prediction = pred
    st.session_state.input_data = df
    st.session_state.page = "📈 Prediction Result"
    st.rerun()
