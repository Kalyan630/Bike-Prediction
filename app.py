import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Bike Demand Prediction",
    layout="wide"
)

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #4F8BF9; color: white; font-weight: bold; border-radius: 8px; padding: 0.5em 2em; }
    .stRadio>div>label { font-size: 1.1em; }
    .stNumberInput>div>input { border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🚲 Bike Demand Prediction", "📊 EDA Profile Report"]
)

# ======================================================
# 📊 EDA PROFILE REPORT PAGE
# ======================================================
if page == "📊 EDA Profile Report":
    st.title("📊 Bike Dataset – EDA Profile Report")

    html_file = "profile.html"

    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            source_code = f.read()

        components.html(
            source_code,
            height=1200,
            scrolling=True
        )
    else:
        st.error("❌ profile.html not found. Please add it to the project root.")

    st.markdown("---")
    st.info("This report was generated using **YData Profiling**.")
    st.stop()
    
# --- Load the trained model ---
MODEL_PATH = 'best_model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error('Model file not found! Please ensure best_model.pkl is present.')
    st.stop()
model = joblib.load(MODEL_PATH)

# --- Detect required features from model ---
try:
    feature_names = model.feature_names_in_
except AttributeError:
    try:
        feature_names = model.named_steps['preprocessor'].feature_names_in_
    except Exception:
        st.error('Could not automatically detect required feature names from the model. Please update the code with the correct feature list.')
        st.stop()

# --- UI for input method ---
st.title('🚲 Bike Demand Prediction App')
st.write('Predict bike demand by entering feature values or uploading a CSV file.')

option = st.radio('**Choose input method:**', ['Manual Input', 'Upload CSV'])

SEASON_MAP = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}

# Group features for better UI
main_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'atemp', 'windspeed', 'day', 'is_peak_hour', 'is_weekend']
weather_features = [f for f in feature_names if f.startswith('weathersit_') or 'weather' in f]
other_features = [f for f in feature_names if f not in main_features + weather_features]

if option == 'Manual Input':
    st.sidebar.header('Input Values')
    input_dict = {}
    st.sidebar.subheader('Date & Time')
    # Date & Time
    input_dict['season'] = st.sidebar.selectbox('Season', list(SEASON_MAP.keys())) if 'season' in feature_names else None
    input_dict['yr'] = st.sidebar.selectbox('Year', [2011, 2012]) if 'yr' in feature_names else None
    input_dict['mnth'] = st.sidebar.slider('Month', 1, 12, 6) if 'mnth' in feature_names else None
    input_dict['day'] = st.sidebar.slider('Day', 1, 31, 1) if 'day' in feature_names else None
    input_dict['hr'] = st.sidebar.slider('Hour', 0, 23, 12) if 'hr' in feature_names else None
    st.sidebar.markdown('---')
    st.sidebar.subheader('Flags')
    # Binary flags
    for flag in ['holiday', 'weekday', 'workingday', 'is_peak_hour', 'is_weekend']:
        if flag in feature_names:
            input_dict[flag] = st.sidebar.selectbox(flag.replace('_', ' ').title(), [0, 1])
    st.sidebar.markdown('---')
    st.sidebar.subheader('Weather & Conditions')
    # Weather
    if 'atemp' in feature_names:
        input_dict['atemp'] = st.sidebar.slider('Feels Like Temp', 0.0, 1.0, 0.5)
    if 'windspeed' in feature_names:
        input_dict['windspeed'] = st.sidebar.slider('Windspeed', 0.0, 1.0, 0.5)
    # Advanced/Other features
    if weather_features or other_features:
        with st.sidebar.expander('Advanced Options'):
            for name in weather_features + other_features:
                if name in ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'atemp', 'windspeed', 'day', 'is_peak_hour', 'is_weekend']:
                    continue
                if name.startswith('weathersit_'):
                    input_dict[name] = st.number_input(name, value=0.0, step=0.01)
                elif 'weather' in name:
                    input_dict[name] = st.number_input(name, value=0.0, step=0.01)
                else:
                    input_dict[name] = st.number_input(name, value=0.0)
    # Prepare input for model
    input_row = {}
    for k, v in input_dict.items():
        if k == 'season' and isinstance(v, str):
            input_row[k] = SEASON_MAP[v]
        elif k == 'yr' and v in [2011, 2012]:
            input_row[k] = 1 if v == 2012 else 0
        elif v is not None:
            input_row[k] = v
    # Fill missing features with 0
    for f in feature_names:
        if f not in input_row:
            input_row[f] = 0
    input_df = pd.DataFrame([input_row])[feature_names]
    # Prettify column names for display
    pretty_cols = [c.replace('_', ' ').title() for c in input_df.columns]
    pretty_input_df = input_df.copy()
    pretty_input_df.columns = pretty_cols
    st.subheader('Selected Inputs')
    st.dataframe(pretty_input_df, use_container_width=True, height=80)
    if st.button('Predict'):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f'Predicted Bike Demand: {prediction:.0f}')
        except Exception as e:
            st.error(f'Prediction error: {e}')
else:
    st.subheader('📁 Upload CSV File')
    st.info(f'Upload a CSV file with columns: {", ".join(feature_names)}')
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('**Input Data Preview:**', df.head())
        if st.button('Predict for CSV', use_container_width=True):
            try:
                if 'season' in df.columns and df['season'].dtype == object:
                    df['season'] = df['season'].map(SEASON_MAP)
                if 'yr' in df.columns and df['yr'].dtype != int:
                    df['yr'] = df['yr'].apply(lambda x: 1 if int(x) == 2012 else 0)
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                df = df[feature_names]
                preds = model.predict(df)
                st.write('**Predictions:**')
                st.dataframe(pd.DataFrame({'Prediction': preds}))
            except Exception as e:
                st.error(f'Error during prediction: {e}')

st.markdown("""
---
<p style='text-align:center; color: #888;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
