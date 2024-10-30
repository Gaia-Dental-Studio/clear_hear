from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

app = Flask(__name__)

# Load the otoscope model
otoscope_model = tf.keras.models.load_model('models/audiogram_classifier_model_multiclass.h5')
otoscope_class_labels = ['abnormal (aom)', 'abnormal (earwax)', 'normal']  # Replace with actual class labels

# Preprocess the image to fit model input requirements
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the model's input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction endpoint
@app.route('/predict-otoscope', methods=['POST'])
def predict_otoscope():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        # Load and preprocess the image
        image = Image.open(file)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = otoscope_model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        # Return JSON response
        return jsonify({
            "prediction": otoscope_class_labels[predicted_class],
            "confidence": float(confidence)
        })
    
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image format. Please upload a valid jpg, jpeg, or png image."}), 400

if __name__ == '__main__':
    app.run(debug=True)
