import streamlit as st
import pandas as pd
import requests

# Define the backend URL
API_URL = "http://127.0.0.1:5000/predict"  # Adjust if Flask runs on a different host/port

# Load the data schema to get the feature names
def load_data():
    data = pd.read_csv('data/dummy_audiometry_data.csv')
    left_columns = data.filter(like='Left').columns
    data.drop(left_columns, axis=1, inplace=True)
    X = data.drop(['Patient_ID', 'Hearing_Status', 'Hearing_Status.1'], axis=1)
    return X.columns

# Get feature names for the input form
feature_columns = load_data()

st.title("Audiometry Classification with SVM")

# Sidebar for audiometry data input
st.sidebar.title("Enter Audiometry Data")
audiometry_data = {}

for feature in feature_columns:
    audiometry_data[feature] = st.sidebar.slider(f'{feature} (dB HL)', min_value=0, max_value=120, value=50)

# Convert input data to JSON
if st.sidebar.button("Predict"):
    try:
        # Send request to Flask API
        response = requests.post(API_URL, json=audiometry_data)
        response_data = response.json()
        
        # Display the prediction result
        if response.status_code == 200:
            st.subheader("Prediction:")
            st.write(response_data["prediction"])
            st.subheader("Input Audiometry Data:")
            input_data_df = pd.DataFrame([response_data["input_data"]])
            st.table(input_data_df)
        else:
            st.error(f"Error: {response_data['error']}")
    
    except Exception as e:
        st.error(f"Error: {e}")
