import streamlit as st
import requests
from PIL import Image

# Define the Flask API URL
API_URL = "http://127.0.0.1:5000/predict-otoscope"  # Adjust if Flask runs on a different host/port

st.title("Otoscope Image Classification")

# Upload the otoscope image
uploaded_image = st.file_uploader("Choose an otoscope image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send the image to the Flask API for prediction
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            files = {"file": uploaded_image.getvalue()}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
                else:
                    st.error(f"Error: {response.json()['error']}")
            except Exception as e:
                st.error(f"Error: {e}")
