from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained SVM model
svm_model = joblib.load('models/SVM_model.pkl')

# Load the audiometry data
def load_data():
    return pd.read_csv('data/dummy_audiometry_data.csv')

data = load_data()
left_columns = data.filter(like='Left').columns
data.drop(left_columns, axis=1, inplace=True)

# Prepare the data
X = data.drop(['Patient_ID', 'Hearing_Status', 'Hearing_Status.1'], axis=1)  # Features (hearing thresholds)
y = data['Hearing_Status']  # Target variable (hearing status)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input format. JSON required."}), 400
    
    input_data = request.get_json()
    
    try:
        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Check that input contains all necessary features
        missing_features = [feature for feature in X.columns if feature not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        # Make predictions
        prediction = svm_model.predict(input_df)
        
        # Map prediction to human-readable text
        prediction_text = "Normal Hearing" if prediction[0] == 0 else "Hearing Loss"
        
        # Return prediction result as JSON
        return jsonify({
            "prediction": prediction_text,
            "input_data": input_data
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
