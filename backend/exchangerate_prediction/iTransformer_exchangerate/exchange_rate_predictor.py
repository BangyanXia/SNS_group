import argparse
import torch
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime
import json

class ExchangeRatePredictor:
    def __init__(self, 
                 from_symbol="GBP", 
                 to_symbol="CNY", 
                 prediction_days=7,
                 api_key="S4J5OANDXEY87T9B"):
        """
        Initialize the Exchange Rate Predictor with the specified currency pair.
        
        Parameters:
        -----------
        from_symbol : str, default="GBP"
            The source currency symbol
        to_symbol : str, default="CNY"
            The target currency symbol
        prediction_days : int, default=7
            Number of days to predict into the future
        api_key : str, default="S4J5OANDXEY87T9B"
            Alpha Vantage API key for fetching currency data
        """
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self.prediction_days = prediction_days
        self.api_key = api_key
        self.data_folder = "./data/exchangerate"
        # Use CHY instead of CNY to match original model path
        if to_symbol == "CNY":
            self.model_id = f"{from_symbol}_to_CHY"
        else:
            self.model_id = f"{from_symbol}_to_{to_symbol}"
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            
        # Create checkpoints directory if it doesn't exist
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints", exist_ok=True)
            
        # Create results directory if it doesn't exist
        if not os.path.exists("./results"):
            os.makedirs("./results", exist_ok=True)
    
    def fetch_exchange_rate_data(self):
        """Fetch exchange rate data from Alpha Vantage API"""
        
        URL = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_DAILY",
            "from_symbol": self.from_symbol,
            "to_symbol": self.to_symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        file_name = f"{self.from_symbol}_to_{self.to_symbol}_Daily_Exchange_Rate_{datetime.now().strftime('%Y-%m-%d')}.csv"
        file_path = os.path.join(self.data_folder, file_name)
        
        if not os.path.exists(file_path):
            print(f"Fetching exchange rate data for {self.from_symbol} to {self.to_symbol}...")
            response = requests.get(URL, params=params)
            data = response.json()
            
            if "Time Series FX (Daily)" not in data:
                error_msg = f"API request failure: {data.get('Note', data)}"
                print(error_msg)
                raise Exception(error_msg)
            
            exchange_rate_data = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
            exchange_rate_data.columns = ["Open", "High", "Low", "Close"]
            
            exchange_rate_data.index = pd.to_datetime(exchange_rate_data.index)
            exchange_rate_data = exchange_rate_data.sort_index()
            
            exchange_rate_data.to_csv(file_path, index_label="date")
            
            print(f"Exchange Rate has been saved to {file_name}")
        else:
            print(f"Using existing data file: {file_name}")
        
        return file_name
    
    def get_model_args(self, data_path):
        """Set up model arguments"""
        args = argparse.Namespace()
        
        # Basic config
        args.is_training = 0
        args.model_id = self.model_id
        args.model = 'iTransformer'
        
        # Data loader
        args.data = 'custom'
        args.root_path = './data/exchangerate/'
        args.data_path = data_path
        args.features = 'M'
        args.target = 'Close'
        args.freq = 'd'
        args.checkpoints = './checkpoints/'
        
        # Forecasting task - match the exact parameters used in original code
        args.seq_len = 128
        args.label_len = 48
        args.pred_len = 7  # Fixed to 7 to match the original model
        
        # Store the user's requested prediction days for later use in output formatting
        self.user_prediction_days = self.prediction_days
        # Override the internal prediction days to match the model
        self.prediction_days = 7
        
        # Model define - match exact parameters from the model path
        args.enc_in = 4
        args.dec_in = 4
        args.c_out = 4
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        
        # These values come from the checkpoint path: 
        # GBP_to_CHY_iTransformer_custom_M_ft128_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0
        if self.model_id == "GBP_to_CHY":
            args.seq_len = 128  # ft128
            args.label_len = 48  # sl48 
            args.pred_len = 7    # ll7
            args.d_model = 512   # pl512
            args.n_heads = 8     # dm8
            args.e_layers = 2    # nh2
            args.d_layers = 1    # el1
            args.d_ff = 2048     # dl2048
        args.d_ff = 2048
        args.moving_avg = 25
        args.factor = 1
        args.distil = True
        args.dropout = 0.1
        args.embed = 'timeF'
        args.activation = 'gelu'
        args.output_attention = False
        args.do_predict = True
        
        # Optimization
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 10
        args.batch_size = 32
        args.patience = 5
        args.learning_rate = 0.0001
        args.des = 'test'
        args.loss = 'MSE'
        args.lradj = 'type1'
        args.use_amp = False
        
        # GPU
        args.use_gpu = torch.cuda.is_available()
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0,1,2,3'
        
        # iTransformer
        args.exp_name = 'MTSF'
        args.channel_independence = False
        args.inverse = True
        args.class_strategy = 'projection'
        args.target_root_path = './data/exchangerate/'
        args.target_data_path = data_path
        args.efficient_training = False
        args.use_norm = True
        args.partial_start_index = 0
        
        return args
    
    def predict(self):
        """Run the prediction process and return results"""
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
        import random
        
        # Set random seeds for reproducibility
        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        # Fetch data or use existing data file
        data_path = self.fetch_exchange_rate_data()
        
        # Get model arguments
        args = self.get_model_args(data_path)
        
        # Set up experiment settings
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)
        
        # Initialize experiment
        exp = Exp_Long_Term_Forecast(args)
        
        # Run prediction
        print(f'Running prediction for {self.from_symbol} to {self.to_symbol} exchange rate...')
        exp.predict(setting, True)
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get the prediction results - try both possible paths
        result_path = f'./results/{setting}/real_prediction.npy'
        
        # If the regular path doesn't exist, try the hardcoded path from the original script
        if not os.path.exists(result_path):
            backup_path = './results/GBP_to_CHY_iTransformer_custom_M_ft128_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/real_prediction.npy'
            if os.path.exists(backup_path):
                result_path = backup_path
                print(f"Using backup prediction path: {backup_path}")
            else:
                raise Exception(f"Prediction results not found at {result_path} or at backup path {backup_path}")
        
        predictions = np.load(result_path)
        
        # Process the prediction results
        # The model predicts Open, High, Low, Close values
        return self._format_predictions(predictions)
    
    def _format_predictions(self, predictions):
        """Format the prediction results for easier consumption"""
        # Predictions shape is [samples, prediction_days, features]
        # We want the last sample's predictions for all days
        last_sample_predictions = predictions[-1]
        
        # Create a list of dictionaries with predicted values
        result = []
        today = datetime.now().date()
        
        # Use either the model's prediction days or the user's requested days, whichever is smaller
        actual_prediction_days = min(
            self.prediction_days, 
            getattr(self, 'user_prediction_days', self.prediction_days),
            last_sample_predictions.shape[0]
        )
        
        for i in range(actual_prediction_days):
            prediction_date = today + pd.Timedelta(days=i+1)
            prediction_day = {
                "date": prediction_date.strftime("%Y-%m-%d"),
                "open": float(last_sample_predictions[i][0]),
                "high": float(last_sample_predictions[i][1]),
                "low": float(last_sample_predictions[i][2]),
                "close": float(last_sample_predictions[i][3])
            }
            result.append(prediction_day)
            
        return result
    
    def predict_to_json(self, output_file=None):
        """Run prediction and save results to JSON file"""
        predictions = self.predict()
        
        if output_file is None:
            output_file = f"{self.from_symbol}_{self.to_symbol}_predictions_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=4)
            
        print(f"Predictions saved to {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Exchange Rate Prediction')
    parser.add_argument('--from_symbol', type=str, default="GBP", help='Source currency symbol')
    parser.add_argument('--to_symbol', type=str, default="CNY", help='Target currency symbol')
    parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    parser.add_argument('--api_key', type=str, default="S4J5OANDXEY87T9B", help='Alpha Vantage API key')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    predictor = ExchangeRatePredictor(
        from_symbol=args.from_symbol,
        to_symbol=args.to_symbol,
        prediction_days=args.days,
        api_key=args.api_key
    )
    
    output_file = predictor.predict_to_json(args.output_file)
    
    # Print sample of results
    with open(output_file, 'r') as f:
        results = json.load(f)
        print("\nPrediction Results Sample:")
        for day in results[:3]:  # Show first 3 days
            print(f"Date: {day['date']}, Close: {day['close']:.4f}")
        if len(results) > 3:
            print("...")
