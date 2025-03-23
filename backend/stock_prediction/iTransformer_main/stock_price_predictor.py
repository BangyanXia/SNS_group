import argparse
import torch
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import json


class StockPricePredictor:
    def __init__(self,
                 ticker="AAPL",
                 prediction_days=7,
                 api_key="S4J5OANDXEY87T9B"):
        """
        Initialize the Stock Price Predictor with the specified ticker symbol.

        Parameters:
        -----------
        ticker : str, default="AAPL"
            The stock ticker symbol (e.g., AAPL for Apple)
        prediction_days : int, default=7
            Number of days to predict into the future
        api_key : str, default="S4J5OANDXEY87T9B"
            Alpha Vantage API key for fetching stock data
        """
        self.ticker = ticker
        self.prediction_days = prediction_days
        self.api_key = api_key
        self.data_folder = "./data/stock"
        self.model_id = f"{ticker}_forecast"

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints", exist_ok=True)

        # Create results directory if it doesn't exist
        if not os.path.exists("./results"):
            os.makedirs("./results", exist_ok=True)

    def fetch_stock_data(self):
        """Fetch stock price data from Alpha Vantage API"""

        URL = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "apikey": self.api_key,
            "outputsize": "full"
        }

        file_name = f"{self.ticker}_Daily_Stock_Price_{datetime.now().strftime('%Y-%m-%d')}.csv"
        file_path = os.path.join(self.data_folder, file_name)

        if not os.path.exists(file_path):
            print(f"Fetching stock price data for {self.ticker}...")
            response = requests.get(URL, params=params)
            data = response.json()

            if "Time Series (Daily)" not in data:
                error_msg = f"API request failure: {data.get('Note', data)}"
                print(error_msg)
                raise Exception(error_msg)

            stock_data = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            stock_data.columns = ["Open", "High", "Low", "Close", "Volume"]

            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data = stock_data.sort_index()

            # Convert volume to numeric
            stock_data["Volume"] = pd.to_numeric(stock_data["Volume"])

            # Convert price columns to numeric
            for col in ["Open", "High", "Low", "Close"]:
                stock_data[col] = pd.to_numeric(stock_data[col])

            stock_data.to_csv(file_path, index_label="date")

            print(f"Stock price data has been saved to {file_name}")
        else:
            print(f"Using existing data file: {file_name}")
            stock_data = pd.read_csv(file_path, index_col="date", parse_dates=True)

        return file_path, stock_data

    def predict(self):
        """Generate stock price predictions based on historical data"""

        # Fetch historical data
        file_path, stock_data = self.fetch_stock_data()

        # Get the latest data for prediction
        latest_data = stock_data.iloc[-60:].copy()  # Use last 60 days of data

        # Calculate simple moving averages
        latest_data['SMA_5'] = latest_data['Close'].rolling(window=5).mean()
        latest_data['SMA_20'] = latest_data['Close'].rolling(window=20).mean()

        # Calculate exponential moving averages
        latest_data['EMA_5'] = latest_data['Close'].ewm(span=5, adjust=False).mean()
        latest_data['EMA_20'] = latest_data['Close'].ewm(span=20, adjust=False).mean()

        # Calculate volatility (std dev of returns)
        latest_data['Volatility'] = latest_data['Close'].pct_change().rolling(window=20).std()

        # Calculate MACD
        latest_data['MACD'] = latest_data['EMA_5'] - latest_data['EMA_20']

        # Analyze recent trend direction and strength
        recent_trend = latest_data['Close'].iloc[-5:].pct_change().mean() * 100
        trend_strength = abs(recent_trend)

        # Generate predictions
        last_close = latest_data['Close'].iloc[-1]
        last_date = latest_data.index[-1]

        # Fill in missing values
        latest_data = latest_data.fillna(method='bfill')

        # Base volatility on recent market activity
        base_volatility = latest_data['Volatility'].iloc[-1] if not pd.isna(
            latest_data['Volatility'].iloc[-1]) else 0.01

        # Create prediction results
        result = []
        predicted_close = last_close

        # Find the last available Friday and Thursday in historical data
        last_friday_data = None
        last_thursday_data = None

        for i in range(10):  # Look back up to 10 days
            check_date = last_date - timedelta(days=i)
            weekday = check_date.weekday()

            # Check if date exists in the dataset
            if check_date in stock_data.index:
                if weekday == 4 and last_friday_data is None:  # Friday
                    last_friday_data = stock_data.loc[check_date]
                elif weekday == 3 and last_thursday_data is None:  # Thursday
                    last_thursday_data = stock_data.loc[check_date]

            # Stop if we have both Friday and Thursday data
            if last_friday_data is not None and last_thursday_data is not None:
                break

        for i in range(1, self.prediction_days + 1):
            # Calculate prediction date
            pred_date = last_date + timedelta(days=i)

            # Check if date is a weekend
            is_weekend = pred_date.weekday() >= 5  # 5=Saturday, 6=Sunday

            # If it's a weekend, use actual Friday or Thursday data
            if is_weekend:
                if last_friday_data is not None:
                    # Use last Friday's actual data
                    prediction = {
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "open": round(float(last_friday_data['Open']), 2),
                        "high": round(float(last_friday_data['High']), 2),
                        "low": round(float(last_friday_data['Low']), 2),
                        "close": round(float(last_friday_data['Close']), 2),
                        "volume": 0,
                        "note": "Weekend - Using last available Friday's actual data, with zero volume"
                    }
                elif last_thursday_data is not None:
                    # Use last Thursday's actual data
                    prediction = {
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "open": round(float(last_thursday_data['Open']), 2),
                        "high": round(float(last_thursday_data['High']), 2),
                        "low": round(float(last_thursday_data['Low']), 2),
                        "close": round(float(last_thursday_data['Close']), 2),
                        "volume": 0,
                        "note": "Weekend - Using last available Thursday's actual data, with zero volume"
                    }
                else:
                    # Fallback if neither Friday nor Thursday data are available
                    prediction = {
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "open": "Market Closed (Weekend)",
                        "high": "Market Closed (Weekend)",
                        "low": "Market Closed (Weekend)",
                        "close": "Market Closed (Weekend)",
                        "volume": 0,
                        "note": "Weekend - No weekday data available"
                    }
            else:
                # Add some randomness to simulate real price movements
                # Direction influenced by recent trend, magnitude by volatility
                daily_return = np.random.normal(recent_trend / 100, base_volatility * 1.5)

                # Calculate predicted prices
                predicted_close = predicted_close * (1 + daily_return)
                predicted_open = predicted_close * (1 + np.random.normal(0, base_volatility * 0.3))
                predicted_high = max(predicted_open, predicted_close) * (
                            1 + abs(np.random.normal(0, base_volatility * 0.5)))
                predicted_low = min(predicted_open, predicted_close) * (
                            1 - abs(np.random.normal(0, base_volatility * 0.5)))

                # Create prediction
                prediction = {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "open": round(float(predicted_open), 2),
                    "high": round(float(predicted_high), 2),
                    "low": round(float(predicted_low), 2),
                    "close": round(float(predicted_close), 2),
                    "volume": int(latest_data['Volume'].mean())
                }

            # Add day's prediction to results
            result.append(prediction)

        return result

    def predict_to_json(self, output_file=None):
        """Run prediction and save results to JSON file"""
        predictions = self.predict()

        if output_file is None:
            output_file = f"{self.ticker}_stock_predictions_{datetime.now().strftime('%Y-%m-%d')}.json"

        output_path = os.path.join("./results", output_file)

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print(f"Predictions saved to {output_path}")
        return output_path, predictions


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default="AAPL", help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    parser.add_argument('--api_key', type=str, default="S4J5OANDXEY87T9B", help='Alpha Vantage API key')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file path')

    args = parser.parse_args()

    predictor = StockPricePredictor(
        ticker=args.ticker,
        prediction_days=args.days,
        api_key=args.api_key
    )

    output_file, predictions = predictor.predict_to_json(args.output_file)

    # Print sample of results
    print("\nPrediction Results Sample:")
    for day in predictions[:3]:  # Show first 3 days
        if isinstance(day['open'], str) and "Market Closed" in day['open']:
            print(f"Date: {day['date']}, {day['open']}, Note: {day.get('note', '')}")
        else:
            print(f"Date: {day['date']}, Open: ${day['open']:.2f}, Close: ${day['close']:.2f}")
    if len(predictions) > 3:
        print("...")