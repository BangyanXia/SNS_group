from flask import Flask, jsonify, send_from_directory
import os
import json
from datetime import datetime
from exchange_rate_predictor import ExchangeRatePredictor

app = Flask(__name__)

# Configure this to match your directory structure
STATIC_FOLDER = '.'  # current directory

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(STATIC_FOLDER, 'exchange-rate-visualization.html')

@app.route('/api/exchange-rate-forecast')
def get_forecast():
    """Return forecast data from the most recent JSON file or generate a new one if needed"""
    # Find the most recent JSON file in the current directory
    json_files = [f for f in os.listdir(STATIC_FOLDER) 
                  if f.endswith('.json') and 'GBP' in f and 'CNY' in f]
    
    if json_files:
        # Sort by modification time (newest first)
        latest_file = sorted(json_files, 
                            key=lambda x: os.path.getmtime(os.path.join(STATIC_FOLDER, x)), 
                            reverse=True)[0]
        
        # Check if the file is recent (less than 1 day old)
        file_mtime = os.path.getmtime(os.path.join(STATIC_FOLDER, latest_file))
        file_age = datetime.now().timestamp() - file_mtime
        
        # If the file is recent, use it
        if file_age < 86400:  # 24 hours in seconds
            with open(os.path.join(STATIC_FOLDER, latest_file), 'r') as f:
                data = json.load(f)
                return jsonify(data)
    
    # If no recent file found, generate a new prediction
    try:
        predictor = ExchangeRatePredictor(from_symbol="GBP", to_symbol="CNY")
        output_file = predictor.predict_to_json()
        
        # Read the generated file
        with open(output_file, 'r') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
