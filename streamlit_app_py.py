# -*- coding: utf-8 -*-
"""streamlit_app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TOveT_EnJdPwFQlFDPNEEiLc7R3vhG0c
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model_pipeline.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Injection Moulding Quality Predictor")

# Inputs
def user_input():
    mould_temp = st.slider("Mould Temperature", 20.0, 300.0)
    pressure = st.slider("Injection Pressure", 10.0, 200.0)
    cycle_time = st.slider("Cycle Time", 1.0, 100.0)
    volume = st.slider("Volume", 0.1, 1000.0)
    return pd.DataFrame([{
        'mould_temperature': mould_temp,
        'injection_pressure': pressure,
        'cycle_time': cycle_time,
        'volume': volume
    }])

input_df = user_input()
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

# Class mapping
class_map = {0: "Acceptable", 1: "Inefficient", 2: "Target", 3: "Waste"}
st.subheader("Prediction")
st.write(f"Predicted Quality: **{class_map[prediction[0]]}**")