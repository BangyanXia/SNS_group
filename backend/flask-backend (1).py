from flask import Flask, jsonify, send_from_directory, request
import os
import json
import sys
import random
from datetime import datetime, timedelta
import numpy as np

current_dir = os.getcwd()
sys.path.append("../..")
# 获取上级目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 ExchangeRatePredictor 和 StockPricePredictor
from exchangerate_prediction.iTransformer_exchangerate.exchange_rate_predictor import ExchangeRatePredictor
from stock_prediction.iTransformer_main.stock_price_predictor import StockPricePredictor
from weather_prediction.weather_predictor import WeatherPredictor

app = Flask(__name__)

# Configure this to match your directory structure
# STATIC_FOLDER = '.'  # current directory
STATIC_FOLDER2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
STATIC_FOLDEREXCHANGE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'exchangerate_prediction', 'iTransformer_exchangerate'))
STATIC_FOLDERSTACK = os.path.abspath(os.path.join(os.path.dirname(__file__), 'stock_prediction', 'iTransformer_main'))
STATIC_FOLDERWEATHER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'weather_prediction'))


@app.route('/')
def index():
    """Serve the chatbot interface as the main entry point"""
    return send_from_directory(STATIC_FOLDER2, 'forecast-chatbot.html')


@app.route('/exchange')
def exchange():
    """Serve the exchange rate visualization HTML page"""
    return send_from_directory(STATIC_FOLDER2, 'exchange-rate-visualization.html')


@app.route('/stocks')
def stocks():
    """Serve the stock visualization HTML page"""
    return send_from_directory(STATIC_FOLDER2, 'stock-price-visualization.html')


@app.route('/weather')
def weather():
    """Serve the weather visualization HTML page"""
    return send_from_directory(STATIC_FOLDER2, 'weather-visualization.html')


@app.route('/api/exchange-rate-forecast')
def get_exchange_rate_forecast():
    """Return exchange rate forecast data from the most recent JSON file or generate a new one if needed"""
    from_symbol = request.args.get('from_symbol', 'GBP')
    to_symbol = request.args.get('to_symbol', 'CNY')

    # Find the most recent JSON file in the current directory for the specified currency pair
    json_files = [f for f in os.listdir(STATIC_FOLDEREXCHANGE)
                  if f.endswith('.json') and from_symbol in f and to_symbol in f]

    if json_files:
        # Sort by modification time (newest first)
        latest_file = sorted(json_files,
                             key=lambda x: os.path.getmtime(os.path.join(STATIC_FOLDEREXCHANGE, x)),
                             reverse=True)[0]

        # Check if the file is recent (less than 1 day old)
        file_mtime = os.path.getmtime(os.path.join(STATIC_FOLDEREXCHANGE, latest_file))
        file_age = datetime.now().timestamp() - file_mtime

        # If the file is recent, use it
        if file_age < 86400:  # 24 hours in seconds
            with open(os.path.join(STATIC_FOLDEREXCHANGE, latest_file), 'r') as f:
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
        json_files = [f for f in os.listdir(STATIC_FOLDERSTACK)
                      if f.endswith('.json') and f'{ticker}_stock_predictions' in f]

        if json_files:
            # Sort by modification time (newest first)
            latest_file = sorted(json_files,
                                 key=lambda x: os.path.getmtime(os.path.join(STATIC_FOLDERSTACK, x)),
                                 reverse=True)[0]

            # Check if the file is recent (less than 1 day old)
            file_mtime = os.path.getmtime(os.path.join(STATIC_FOLDERSTACK, latest_file))
            file_age = datetime.now().timestamp() - file_mtime

            # If the file is recent, use it
            if file_age < 86400:  # 24 hours in seconds
                with open(os.path.join(STATIC_FOLDERSTACK, latest_file), 'r') as f:
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


@app.route('/api/weather-forecast')
def get_weather_forecast():
    """Return weather forecast data from the most recent JSON file or generate a new one if needed"""
    city = request.args.get('city', 'Beijing')
    country = request.args.get('country', 'CN')
    days = request.args.get('days', 7, type=int)
    api_key = request.args.get('api_key', 'YOUR_OPENWEATHERMAP_API_KEY')

    try:
        # Find the most recent JSON file for the specified city
        city_folder = os.path.join(STATIC_FOLDERWEATHER, city)
        if not os.path.exists(city_folder):
            os.makedirs(city_folder, exist_ok=True)

        json_files = [f for f in os.listdir(city_folder)
                      if f.endswith('.json') and 'weather_forecast' in f]

        if json_files:
            # Sort by modification time (newest first)
            latest_file = sorted(json_files,
                                 key=lambda x: os.path.getmtime(os.path.join(city_folder, x)),
                                 reverse=True)[0]

            # Check if the file is recent (less than 1 day old)
            file_mtime = os.path.getmtime(os.path.join(city_folder, latest_file))
            file_age = datetime.now().timestamp() - file_mtime

            # If the file is recent, use it
            if file_age < 86400:  # 24 hours in seconds
                with open(os.path.join(city_folder, latest_file), 'r') as f:
                    data = json.load(f)
                    print(f"Using existing weather data for {city} from {latest_file}")
                    return jsonify(data)

        # If no recent file found, generate a new prediction
        predictor = WeatherPredictor(city=city, country=country, days=days, api_key=api_key)
        output_file, predictions = predictor.predict_to_json()

        print(f"Generated new weather forecast for {city}, saved to {output_file}")
        return jsonify(predictions)
    except Exception as e:
        print(f"Error in weather forecast API: {str(e)}")

        # Generate sample data as fallback
        return jsonify(generate_sample_weather_data(city, country, days))


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
        price_change = (random.random() - 0.45) * 5  # Slight upward bias
        open_price = base_price + (i * 2) + (random.random() - 0.5) * 3
        close_price = open_price + price_change
        high_price = max(open_price, close_price) + random.random() * 2
        low_price = min(open_price, close_price) - random.random() * 2

        # Add to sample data
        sample_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(open_price), 2),
            "high": round(float(high_price), 2),
            "low": round(float(low_price), 2),
            "close": round(float(close_price), 2),
            "volume": int(random.random() * 10000000 + 5000000)
        })

    return sample_data


def generate_sample_weather_data(city, country, days=7):
    """Generate sample weather forecast when API fails"""
    forecast = []
    today = datetime.now()

    # City-based temperature ranges (approximations)
    city_temp_map = {
        "Beijing": {"summer": (25, 35), "winter": (-5, 10), "other": (10, 25)},
        "London": {"summer": (15, 25), "winter": (0, 10), "other": (5, 20)},
        "New York": {"summer": (20, 32), "winter": (-5, 5), "other": (5, 20)},
        "Tokyo": {"summer": (25, 35), "winter": (5, 15), "other": (10, 25)},
        "Sydney": {"summer": (25, 35), "winter": (10, 20), "other": (15, 25)},
        "Moscow": {"summer": (15, 30), "winter": (-15, 0), "other": (0, 15)},
        "Cairo": {"summer": (30, 40), "winter": (15, 25), "other": (20, 30)}
    }

    # Use provided city or default to Beijing's range
    city_temps = city_temp_map.get(city, city_temp_map["Beijing"])

    # Determine season (based on northern hemisphere, except for Sydney)
    month = today.month
    if city == "Sydney":  # Southern hemisphere
        is_summer = month <= 2 or month == 12
        is_winter = 6 <= month <= 8
    else:  # Northern hemisphere
        is_summer = 6 <= month <= 8
        is_winter = month <= 2 or month == 12

    # Select temperature range based on season
    if is_summer:
        temp_range = city_temps["summer"]
    elif is_winter:
        temp_range = city_temps["winter"]
    else:
        temp_range = city_temps["other"]

    # Weather conditions for sampling
    weather_conditions = [
        {"id": 800, "main": "Clear", "description": "clear sky", "icon": "01d"},
        {"id": 801, "main": "Clouds", "description": "few clouds", "icon": "02d"},
        {"id": 802, "main": "Clouds", "description": "scattered clouds", "icon": "03d"},
        {"id": 803, "main": "Clouds", "description": "broken clouds", "icon": "04d"},
        {"id": 500, "main": "Rain", "description": "light rain", "icon": "10d"},
        {"id": 501, "main": "Rain", "description": "moderate rain", "icon": "10d"},
        {"id": 600, "main": "Snow", "description": "light snow", "icon": "13d"},
        {"id": 741, "main": "Fog", "description": "fog", "icon": "50d"},
        {"id": 701, "main": "Mist", "description": "mist", "icon": "50d"},
    ]

    # Generate sample forecast for each day
    for i in range(1, days + 1):
        # Date for this forecast
        forecast_date = today + timedelta(days=i)
        date_str = forecast_date.strftime("%Y-%m-%d")

        # Base temperature on range for the city/season
        base_temp = random.uniform(temp_range[0], temp_range[1])

        # Add some continuity between days (trend)
        if i > 1:
            prev_day = forecast[-1]
            trend_factor = 0.7  # How much previous day influences the next
            base_temp = (base_temp + trend_factor * prev_day["day_temp"]) / (1 + trend_factor)

        # Day and night temperatures
        day_temp = base_temp + random.uniform(-2, 2)
        night_temp = day_temp - random.uniform(5, 10)  # Nights are cooler

        # Select weather condition appropriate for the season
        if is_winter and random.random() < 0.7:
            # More clouds, fog, snow in winter
            condition = random.choice([
                w for w in weather_conditions
                if w["id"] in [600, 801, 802, 803, 741, 701]
            ])
        elif is_summer and random.random() < 0.7:
            # More clear, few clouds, occasional rain in summer
            condition = random.choice([
                w for w in weather_conditions
                if w["id"] in [800, 801, 500]
            ])
        else:
            # Any condition
            condition = random.choice(weather_conditions)

        # Weather dependent values
        if condition["main"] in ["Rain", "Snow"]:
            humidity = random.randint(70, 95)
            precipitation_chance = random.randint(60, 100)
            precipitation_amount = random.uniform(0.5, 15) if condition["main"] == "Rain" else random.uniform(1, 8)
        elif condition["main"] in ["Clouds", "Fog", "Mist"]:
            humidity = random.randint(60, 85)
            precipitation_chance = random.randint(20, 60)
            precipitation_amount = random.uniform(0, 1)
        else:
            humidity = random.randint(30, 70)
            precipitation_chance = random.randint(0, 20)
            precipitation_amount = 0

        # Add day to forecast
        forecast.append({
            "date": date_str,
            "day_temp": round(day_temp, 1),
            "night_temp": round(night_temp, 1),
            "humidity": humidity,
            "wind_speed": round(random.uniform(0, 15), 1),
            "pressure": random.randint(990, 1030),
            "weather_id": condition["id"],
            "weather_main": condition["main"],
            "weather_description": condition["description"],
            "weather_icon": condition["icon"],
            "precipitation_chance": precipitation_chance,
            "precipitation_amount": round(precipitation_amount, 1),
            "uv_index": random.randint(0, 11)
        })

    return forecast


if __name__ == '__main__':
    app.run(debug=True, port=5000)