from flask import Flask, jsonify, send_from_directory, request
import os
import json
from datetime import datetime
from exchange_rate_predictor import ExchangeRatePredictor
from stock_price_predictor import StockPricePredictor

app = Flask(__name__)

# Configure this to match your directory structure
STATIC_FOLDER = '.'  # current directory


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(STATIC_FOLDER, 'exchange-rate-visualization.html')


@app.route('/stocks')
def stocks():
    """Serve the stock visualization HTML page"""
    return send_from_directory(STATIC_FOLDER, 'stock-price-visualization.html')


@app.route('/api/exchange-rate-forecast')
def get_exchange_rate_forecast():
    """Return exchange rate forecast data from the most recent JSON file or generate a new one if needed"""
    from_symbol = request.args.get('from_symbol', 'GBP')
    to_symbol = request.args.get('to_symbol', 'CNY')

    # Find the most recent JSON file in the current directory for the specified currency pair
    json_files = [f for f in os.listdir(STATIC_FOLDER)
                  if f.endswith('.json') and from_symbol in f and to_symbol in f]

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
        predictor = ExchangeRatePredictor(from_symbol=from_symbol, to_symbol=to_symbol)
        output_file = predictor.predict_to_json()

        # Read the generated file
        with open(output_file, 'r') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stock-forecast')
def get_stock_forecast():
    """Return stock price forecast data from the most recent JSON file or generate a new one if needed"""
    ticker = request.args.get('ticker', 'AAPL')
    days = request.args.get('days', 7, type=int)

    try:
        # Find the most recent JSON file in the current directory for the specified ticker
        json_files = [f for f in os.listdir(STATIC_FOLDER)
                      if f.endswith('.json') and f'{ticker}_stock_predictions' in f]

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
                    print(f"Using existing stock data for {ticker} from {latest_file}")
                    return jsonify(data)

        # If no recent file found, generate a new prediction
        predictor = StockPricePredictor(ticker=ticker, prediction_days=days)
        output_file, predictions = predictor.predict_to_json()

        print(f"Generated new stock prediction for {ticker}, saved to {output_file}")
        return jsonify(predictions)
    except Exception as e:
        print(f"Error in stock forecast API: {str(e)}")

        # Return sample data as fallback
        sample_data = generate_sample_stock_data(ticker, days)
        return jsonify(sample_data)


def generate_sample_stock_data(ticker, days=7):
    """Generate sample stock data when API fails"""
    today = datetime.now()
    base_price = 190 if ticker in ['AAPL', 'MSFT'] else 100
    sample_data = []

    for i in range(1, days + 1):
        # Skip weekends
        date = today + timedelta(days=i)
        if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
            continue

        # Generate random price movement
        price_change = (np.random.random() - 0.45) * 5  # Slight upward bias
        open_price = base_price + (i * 2) + (np.random.random() - 0.5) * 3
        close_price = open_price + price_change
        high_price = max(open_price, close_price) + np.random.random() * 2
        low_price = min(open_price, close_price) - np.random.random() * 2

        # Add to sample data
        sample_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(open_price), 2),
            "high": round(float(high_price), 2),
            "low": round(float(low_price), 2),
            "close": round(float(close_price), 2),
            "volume": int(np.random.random() * 10000000 + 5000000)
        })

    return sample_data


if __name__ == '__main__':
    app.run(debug=True, port=5000)