import streamlit as st
import pandas as pd
from predictive_model import train_model, predict_demand
from data_collection import integrate_data

# Streamlit app
st.title("Hyper-Local Supply Chain Forecaster")
city = st.text_input("Enter City", "Mumbai")
location = st.text_input("Enter Location (lat,lon,radius)", "19.0760,72.8777,10mi")

if st.button("Forecast Demand"):
    df = integrate_data(city, location)
    model, _, _ = train_model(df)  # Simplified for demo
    prediction = predict_demand(model, df)
    st.write(f"Predicted Demand for {city}: {prediction[0]:.2f} units")