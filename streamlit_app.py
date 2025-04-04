
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model_pipeline.pkl")
scaler = joblib.load("scaler.pkl")

# Page Title
st.title("Injection Moulding Quality Predictor")
st.markdown("Use the sliders to set process parameters and predict product quality class.")

# Input function with proper column names
def user_input():
    mould_temp = st.slider("Mould Temperature (°C)", 20.0, 300.0, 150.0)
    pressure = st.slider("Injection Pressure (bar)", 10.0, 200.0, 100.0)
    cycle_time = st.slider("Cycle Time (s)", 1.0, 100.0, 30.0)
    volume = st.slider("Volume (cm³)", 0.1, 1000.0, 200.0)

    df = pd.DataFrame([{
        'mould_temperature': mould_temp,
        'injection_pressure': pressure,
        'cycle_time': cycle_time,
        'volume': volume
    }])
    return df

# Collect input
input_df = user_input()

# Show input for debugging
st.write("Input Parameters", input_df)

# Ensure same column names and order as scaler expects
input_df = input_df.reindex(columns=scaler.feature_names_in_)

# Preprocess input
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)
proba = model.predict_proba(scaled_input)

# Class mapping
class_map = {0: "Acceptable", 1: "Inefficient", 2: "Target", 3: "Waste"}
predicted_label = class_map.get(prediction[0], "Unknown")

# Show prediction
st.subheader("Prediction")
st.write(f"**Predicted Quality Class:** {predicted_label}")

# Show probability scores
st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(proba, columns=class_map.values())
st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'))
