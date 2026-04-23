from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load model (adjust path if needed)
model_path = os.path.join('models', 'energy_forecast_model.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON: {"hour": 14, "day_of_week": 2, "is_weekend": 0, "lag_1": 3.2, "rolling_mean_3": 3.1}
    You can also send minimal features; model expects 5 features.
    """
    data = request.get_json()
    
    # Extract features (must match training order)
    features = np.array([[
        data['hour'],
        data['day_of_week'],
        data['is_weekend'],
        data['lag_1'],
        data['rolling_mean_3']
    ]])
    
    prediction = model.predict(features)
    return jsonify({'predicted_energy_kw': round(prediction[0], 2)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
