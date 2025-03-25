import argparse
import torch
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler
from argparse import Namespace

# Import the iTransformer model
from model.iTransformer import Model


class StockPricePredictor:
    def __init__(self,
                 ticker="AAPL",
                 prediction_days=7,
                 api_key="S4J5OANDXEY87T9B",
                 checkpoint_path="./checkpoints/stock_iTransformer_custom_M_ft365_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/checkpoint.pth"):
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
        checkpoint_path : str
            Path to the trained iTransformer model checkpoint
        """
        self.ticker = ticker
        self.prediction_days = prediction_days
        self.api_key = api_key
        self.data_folder = "./data/stock"
        self.model_id = f"{ticker}_forecast"
        self.checkpoint_path = checkpoint_path

        # Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = torch.device("cpu")

        # Model config
        self.seq_len = 365  # Must match training sequence length

        # Create required directories
        for directory in ["./data/stock", "./checkpoints", "./results"]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

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

    def get_fed_rate_data(self):
        """
        Get Federal Funds Rate data or generate placeholder if not available
        """
        try:
            # Try to fetch Federal Funds Rate data (DFF)
            # In a real scenario, you would fetch this from FRED or another source
            print("Attempting to fetch Federal Funds Rate data...")

            # For this example, we'll just generate a placeholder
            # In a real application, you would fetch this from a reliable source
            dummy_fed_rate = 5.25  # Current approximate Fed Funds rate as of 2023
            return dummy_fed_rate

        except Exception as e:
            print(f"Error fetching Fed Rate data: {e}")
            print("Using a default placeholder value for Fed Rate")
            return 5.0  # Default placeholder

    def prepare_model_input(self, stock_data):
        """
        Prepare the input data for the iTransformer model
        """
        # Add Federal Funds Rate column if it doesn't exist
        if 'DFF' not in stock_data.columns:
            fed_rate = self.get_fed_rate_data()
            stock_data['DFF'] = fed_rate

        # Ensure we have enough historical data
        if len(stock_data) < self.seq_len:
            raise ValueError(
                f"Not enough historical data. Need at least {self.seq_len} days, but got {len(stock_data)}")

        # Select the latest data points
        latest_data = stock_data.iloc[-self.seq_len:][['Open', 'Close', 'High', 'Low', 'Volume', 'DFF']].values

        # Scale the data
        scaler = StandardScaler()
        scaler.fit(latest_data)
        latest_data_scaled = scaler.transform(latest_data)

        # Convert to tensor
        latest_data_tensor = torch.tensor(latest_data_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Generate time features for encoder
        dates = stock_data.index[-self.seq_len:]
        x_mark_enc = self.generate_time_features(dates)
        x_mark_enc_tensor = torch.tensor(x_mark_enc, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Generate time features for decoder (future dates)
        last_date = stock_data.index[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(self.prediction_days)]
        x_mark_dec = self.generate_time_features(future_dates)
        x_mark_dec_tensor = torch.tensor(x_mark_dec, dtype=torch.float32).unsqueeze(0).to(self.device)

        return latest_data_tensor, x_mark_enc_tensor, x_mark_dec_tensor, scaler, future_dates

    def generate_time_features(self, dates):
        """Generate time features: month, day, weekday"""
        return np.array([[d.month, d.day, d.weekday()] for d in dates], dtype=np.float32)

    def load_model(self):
        """
        Load the pre-trained iTransformer model
        """
        # Define model configurations
        model_configs = Namespace(
            enc_in=6, dec_in=6, c_out=6, seq_len=self.seq_len, pred_len=self.prediction_days, label_len=48,
            d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.1, factor=1,
            activation='gelu', output_attention=False, use_norm=True, embed='timeF',
            freq='h', class_strategy='projection'
        )

        # Initialize the model
        model = Model(model_configs).to(self.device)

        # Load the trained weights
        try:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            model.eval()
            print(f"Successfully loaded model from {self.checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple prediction method")
            return None

    def predict(self):
        """Generate stock price predictions using the iTransformer model"""

        # Fetch historical data
        file_path, stock_data = self.fetch_stock_data()

        try:
            # Try to use the iTransformer model
            model = self.load_model()

            if model is not None:
                # Prepare input data
                latest_data, x_mark_enc, x_mark_dec, scaler, future_dates = self.prepare_model_input(stock_data)

                # Make predictions
                with torch.no_grad():
                    x_dec = torch.zeros((1, self.prediction_days, 6)).to(self.device)  # Empty decoder input
                    predicted_seq = model(latest_data, x_mark_enc, x_dec, x_mark_dec)

                # Convert predictions back to original scale
                predictions = predicted_seq.cpu().numpy().squeeze()
                predictions_original = scaler.inverse_transform(predictions)

                # Create prediction results
                result = []
                for i in range(self.prediction_days):
                    pred_date = future_dates[i]

                    # Check if it's weekend
                    is_weekend = pred_date.weekday() >= 5  # 5=Saturday, 6=Sunday

                    if is_weekend:
                        # For weekends, indicate market closed
                        prediction = {
                            "date": pred_date.strftime("%Y-%m-%d"),
                            "open": "Market Closed (Weekend)",
                            "high": "Market Closed (Weekend)",
                            "low": "Market Closed (Weekend)",
                            "close": "Market Closed (Weekend)",
                            "volume": 0,
                            "note": "Weekend - Market Closed"
                        }
                    else:
                        # For weekdays, use model predictions
                        prediction = {
                            "date": pred_date.strftime("%Y-%m-%d"),
                            "open": round(float(predictions_original[i, 0]), 2),
                            "high": round(float(predictions_original[i, 2]), 2),
                            "low": round(float(predictions_original[i, 3]), 2),
                            "close": round(float(predictions_original[i, 1]), 2),
                            "volume": int(predictions_original[i, 4]),
                            "note": "Prediction from iTransformer model"
                        }

                    result.append(prediction)

                return result
            else:
                # Fall back to the simple prediction method if model loading fails
                return self.predict_simple(stock_data)

        except Exception as e:
            print(f"Error using iTransformer model: {e}")
            print("Falling back to simple prediction method")
            return self.predict_simple(stock_data)

    def predict_simple(self, stock_data):
        """
        Fallback simple prediction method (original implementation)
        """
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
                    "volume": int(latest_data['Volume'].mean()),
                    "note": "Prediction from simple statistical model (fallback)"
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

    parser = argparse.ArgumentParser(description='Stock Price Prediction with iTransformer')
    parser.add_argument('--ticker', type=str, default="AAPL", help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    parser.add_argument('--api_key', type=str, default="S4J5OANDXEY87T9B", help='Alpha Vantage API key')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--checkpoint', type=str,
                        default="./checkpoints/stock_iTransformer_custom_M_ft365_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/checkpoint.pth",
                        help='Path to model checkpoint file')

    args = parser.parse_args()

    predictor = StockPricePredictor(
        ticker=args.ticker,
        prediction_days=args.days,
        api_key=args.api_key,
        checkpoint_path=args.checkpoint
    )

    output_file, predictions = predictor.predict_to_json(args.output_file)

    # Print sample of results
    print("\nPrediction Results Sample:")
    for day in predictions[:3]:  # Show first 3 days
        if isinstance(day['open'], str) and "Market Closed" in day['open']:
            print(f"Date: {day['date']}, {day['open']}, Note: {day.get('note', '')}")
        else:
            print(
                f"Date: {day['date']}, Open: ${day['open']:.2f}, Close: ${day['close']:.2f}, Note: {day.get('note', '')}")
    if len(predictions) > 3:
        print("...")