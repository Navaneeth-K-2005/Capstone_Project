from flask import Flask, request, jsonify
from src.model import load_model, predict
from src.logging_setup import setup_logging
import pandas as pd

app = Flask(__name__)
logger = setup_logging()
model = load_model()

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    
   
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing input features'}), 400
    
    df = pd.DataFrame([data['features']])
    prediction = predict(model, df)
    
    logger.info(f"Input: {data['features']}, Prediction: {prediction.tolist()}")
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
