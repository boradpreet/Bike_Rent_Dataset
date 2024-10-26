import streamlit as st
import pickle
import numpy as np


# Load your trained model once
with open('Bike.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app setup
st.title("Bike Rental Prediction App")

# Create input fields for the features
season = st.selectbox('Season', [1, 2, 3, 4])  # Example choices for season
mnth = st.slider('Month', 1, 12, 1)
holiday = st.selectbox('Holiday', [0, 1])
weekday = st.slider('Weekday', 0, 6, 0)
workingday = st.selectbox('Working Day', [0, 1])
weathersit = st.selectbox('Weather Situation', [1, 2, 3, 4])
temp = st.number_input('Temperature', min_value=0.0, max_value=1.0, value=0.5)
atemp = st.number_input('Apparent Temperature', min_value=0.0, max_value=1.0, value=0.5)
hum = st.number_input('Humidity', min_value=0.0, max_value=1.0, value=0.5)
windspeed = st.number_input('Windspeed', min_value=0.0, max_value=1.0, value=0.5)
casual = st.number_input('Casual Users', min_value=0, value=0)
registered = st.number_input('Registered Users', min_value=0, value=0)
Year = st.slider('Year', 2010, 2022, 2020)
Day = st.slider('Day', 1, 31, 1)

# When the user clicks the 'Predict' button, the prediction is made
if st.button('Predict'):
    # Prepare the feature vector for the model
    features = np.array([[season, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, Year, Day]])
    
    # Predict the output using the trained model
    prediction = model.predict(features)
    
    # Use the exact output value without rounding
    output = int(prediction[0])  # This should give you the exact prediction value
    
    # Display the prediction result
    st.success(f"Total Bike Rent: {output}")
