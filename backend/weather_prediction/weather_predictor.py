import requests
import pandas as pd
import numpy as np
import json
import os
import sys
import torch
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add current directory to system path to ensure modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class WeatherPredictor:
    def __init__(self,
                 city="Beijing",
                 country="CN",
                 days=7,
                 api_key="YOUR_OPENWEATHERMAP_API_KEY",
                 use_ml_model=True):
        """
        Initialize the Weather Predictor with the specified location.

        Parameters:
        -----------
        city : str, default="Beijing"
            The city name for weather forecast
        country : str, default="CN"
            The country code (ISO 3166 country codes)
        days : int, default=7
            Number of days to predict into the future
        api_key : str
            OpenWeatherMap API key for fetching weather data
        use_ml_model : bool, default=True
            Whether to use machine learning model for prediction (if available)
        """
        self.city = city
        self.country = country
        self.days = days
        self.api_key = api_key
        self.data_folder = "./data/weather"
        self.use_ml_model = use_ml_model

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)

        # Check if ML model is available for this city
        self.has_ml_model = False
        if self.use_ml_model:
            self.has_ml_model = self._check_ml_model_availability()

    def _check_ml_model_availability(self):
        """Check if ML model is available for this city"""
        if self.city == "Beijing":
            model_path = "Beijing/weather_lstm_model_B.pth"
            x_test_path = "Beijing/X_test_tensor_B.pth"
            scaler_path = "Beijing/scaler_y_B.pkl"
            return os.path.exists(model_path) and os.path.exists(x_test_path) and os.path.exists(scaler_path)
        elif self.city == "London":
            model_path = "London/weather_lstm_model_L.pth"
            x_test_path = "London/X_test_tensor_L.pth"
            scaler_path = "London/scaler_y_L.pkl"
            return os.path.exists(model_path) and os.path.exists(x_test_path) and os.path.exists(scaler_path)
        return False

    def fetch_current_weather(self):
        """Fetch current weather data from OpenWeatherMap API"""

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{self.city},{self.country}",
            "appid": self.api_key,
            "units": "metric"  # Use metric units (Celsius)
        }

        file_name = f"{self.city}_{self.country}_current_weather_{datetime.now().strftime('%Y-%m-%d')}.json"
        file_path = os.path.join(self.data_folder, file_name)

        if os.path.exists(file_path):
            print(f"Using existing current weather data for {self.city}")
            with open(file_path, 'r') as f:
                return json.load(f)

        try:
            print(f"Fetching current weather data for {self.city}...")
            response = requests.get(url, params=params)
            data = response.json()

            if response.status_code != 200:
                error_msg = f"API request failure: {data.get('message', 'Unknown error')}"
                print(error_msg)
                return None

            # Save the response to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Current weather data has been saved to {file_name}")
            return data
        except Exception as e:
            print(f"Error fetching current weather: {str(e)}")
            return None

    def predict_with_ml_model(self):
        """Use machine learning model for weather prediction"""
        try:
            print(f"Using ML model for {self.city} prediction...")

            if self.city == "Beijing":
                # Import Beijing model module
                try:
                    from Beijing.ModelTrain_Beijing import LSTMModel
                except ImportError:
                    print("Failed to import Beijing.ModelTrain_Beijing module, adjusting import path...")
                    # If direct import fails, try adjusting path
                    sys.path.append(os.path.join(current_dir, "Beijing"))
                    from Beijing.ModelTrain_Beijing import LSTMModel

                model_path = "Beijing/weather_lstm_model_B.pth"
                x_test_path = "Beijing/X_test_tensor_B.pth"
                scaler_path = "Beijing/scaler_y_B.pkl"
                suffix = "B"
            elif self.city == "London":
                # Import London model module
                try:
                    from London.ModelTrain_London import LSTMModel
                except ImportError:
                    print("Failed to import London.ModelTrain_London module, adjusting import path...")
                    # If direct import fails, try adjusting path
                    sys.path.append(os.path.join(current_dir, "London"))
                    from London.ModelTrain_London import LSTMModel

                model_path = "London/weather_lstm_model_L.pth"
                x_test_path = "London/X_test_tensor_L.pth"
                scaler_path = "London/scaler_y_L.pkl"
                suffix = "L"
            else:
                raise ValueError(f"No ML model available for {self.city}")

            # Load model
            model = LSTMModel(target_size=4)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model.eval()
            print(f"Model loaded for {self.city}")

            # Load test data
            X_test = torch.load(x_test_path)
            print(f"X_test loaded, shape: {X_test.shape}")

            # Predict next days
            X_input = X_test[-1].unsqueeze(0)  # Take the last day's data
            future_preds = []

            # Predict day by day
            for _ in range(min(7, self.days)):  # Maximum 7 days
                with torch.no_grad():
                    y_pred = model(X_input).cpu().numpy()  # Predict 1 day
                    future_preds.append(y_pred)

                # Fill in missing features
                y_pred_filled = np.zeros((1, 1, X_input.shape[2]))  # Shape (1, 1, feature_size)
                y_pred_filled[:, :, -4:] = y_pred  # Only replace the last 4 target variables

                # Convert to PyTorch Tensor
                y_pred_tensor = torch.tensor(y_pred_filled, dtype=torch.float32)

                # Concatenate dimensions
                X_input = torch.cat((X_input[:, 1:, :], y_pred_tensor), dim=1)

            # Convert predictions to numpy array
            future_preds = np.array(future_preds).reshape(len(future_preds), 4)
            print("Pre-inverse transform values:")
            print(future_preds)

            # Inverse transform
            scaler_y = joblib.load(scaler_path)
            future_preds = scaler_y.inverse_transform(future_preds)
            print("Inverse transform complete")

            print(
                f"Weather prediction for next {len(future_preds)} days (temperature, humidity, wind speed, pressure):")
            print(future_preds)

            # Convert ML model predictions to standard format
            forecast = self._convert_ml_predictions_to_forecast(future_preds)
            return forecast

        except Exception as e:
            print(f"ML model prediction failed: {str(e)}")
            print("Falling back to rule-based prediction method")
            return None

    def _convert_ml_predictions_to_forecast(self, predictions):
        """Convert ML model predictions to standard forecast format"""
        forecast = []
        today = datetime.now()

        # Weather condition mapping (simple rules based on temperature and humidity)
        def get_weather_condition(temp, humidity):
            if humidity > 80:
                if temp < 0:
                    return {"id": 600, "main": "Snow", "description": "light snow", "icon": "13d"}
                else:
                    return {"id": 500, "main": "Rain", "description": "light rain", "icon": "10d"}
            elif humidity > 60:
                return {"id": 803, "main": "Clouds", "description": "broken clouds", "icon": "04d"}
            elif humidity > 40:
                return {"id": 801, "main": "Clouds", "description": "few clouds", "icon": "02d"}
            else:
                return {"id": 800, "main": "Clear", "description": "clear sky", "icon": "01d"}

        for i in range(min(self.days, len(predictions))):
            # Extract data from predictions
            temp = predictions[i][0]  # Temperature
            humidity = predictions[i][1]  # Humidity
            wind_speed = predictions[i][2]  # Wind speed
            pressure = predictions[i][3]  # Pressure

            # Ensure values are in reasonable ranges
            humidity = max(0, min(100, humidity))
            wind_speed = max(0, wind_speed)
            pressure = max(950, min(1050, pressure))

            # Generate night temperature (fixed 6 degrees lower than day)
            night_temp = temp - 6

            # Determine weather condition
            condition = get_weather_condition(temp, humidity)

            # Calculate precipitation probability and amount based on humidity
            if condition["main"] in ["Rain", "Snow"]:
                precipitation_chance = min(100, max(60, humidity))
                precipitation_amount = 5.0 if condition["main"] == "Rain" else 2.5
            elif condition["main"] == "Clouds":
                precipitation_chance = min(60, max(20, humidity - 20))
                precipitation_amount = 1.0 if precipitation_chance > 40 else 0
            else:
                precipitation_chance = max(0, min(20, humidity - 40))
                precipitation_amount = 0

            # UV index removed as requested

            # Forecast date
            forecast_date = today + timedelta(days=i + 1)
            date_str = forecast_date.strftime("%Y-%m-%d")

            # Add to forecast
            forecast.append({
                "date": date_str,
                "day_temp": round(temp, 1),
                "night_temp": round(night_temp, 1),
                "humidity": int(round(humidity)),
                "wind_speed": round(wind_speed, 1),
                "pressure": int(round(pressure)),
                "weather_id": condition["id"],
                "weather_main": condition["main"],
                "weather_description": condition["description"],
                "weather_icon": condition["icon"],
                "precipitation_chance": int(round(precipitation_chance)),
                "precipitation_amount": round(precipitation_amount, 1)
            })

        # If ML model predicted fewer days than requested, fill the rest with rule-based predictions
        if len(predictions) < self.days:
            # Use last prediction as a base for additional predictions
            last_pred = forecast[-1]

            for i in range(len(predictions), self.days):
                forecast_date = today + timedelta(days=i + 1)
                date_str = forecast_date.strftime("%Y-%m-%d")

                # Use fixed values instead of random changes
                day_temp = last_pred["day_temp"]  # No change
                night_temp = day_temp - 6  # Fixed difference

                # Fixed values for other parameters
                humidity = last_pred["humidity"]
                wind_speed = last_pred["wind_speed"]
                pressure = last_pred["pressure"]

                # Determine weather condition based on temperature and humidity
                condition = get_weather_condition(day_temp, humidity)

                # Precipitation probability and amount
                if condition["main"] in ["Rain", "Snow"]:
                    precipitation_chance = min(100, max(60, humidity))
                    precipitation_amount = 5.0 if condition["main"] == "Rain" else 2.5
                elif condition["main"] == "Clouds":
                    precipitation_chance = min(60, max(20, humidity - 20))
                    precipitation_amount = 1.0 if precipitation_chance > 40 else 0
                else:
                    precipitation_chance = max(0, min(20, humidity - 40))
                    precipitation_amount = 0

                # UV index removed as requested

                # Add to forecast
                forecast.append({
                    "date": date_str,
                    "day_temp": round(day_temp, 1),
                    "night_temp": round(night_temp, 1),
                    "humidity": int(round(humidity)),
                    "wind_speed": round(wind_speed, 1),
                    "pressure": int(round(pressure)),
                    "weather_id": condition["id"],
                    "weather_main": condition["main"],
                    "weather_description": condition["description"],
                    "weather_icon": condition["icon"],
                    "precipitation_chance": int(round(precipitation_chance)),
                    "precipitation_amount": round(precipitation_amount, 1),
                    "prediction_method": "ML model extended prediction"
                })

        return forecast

    def predict(self):
        """Generate weather predictions based on current conditions and historical patterns"""
        # Check if ML model is available and user enabled ML prediction
        if self.has_ml_model and self.use_ml_model:
            ml_forecast = self.predict_with_ml_model()
            if ml_forecast:
                print("ML model prediction successful")
                return ml_forecast

        # If no ML model or ML prediction failed, try to get current weather and generate rule-based prediction
        current_weather = self.fetch_current_weather()

        # If API request fails, use sample data
        if current_weather is None:
            return self._generate_sample_forecast()

        # Extract current conditions
        try:
            current_temp = current_weather['main']['temp']
            current_humidity = current_weather['main']['humidity']
            current_pressure = current_weather['main']['pressure']
            current_wind_speed = current_weather['wind']['speed']
            current_weather_id = current_weather['weather'][0]['id']
            current_weather_main = current_weather['weather'][0]['main']
            current_weather_desc = current_weather['weather'][0]['description']
            lat = current_weather['coord']['lat']
            lon = current_weather['coord']['lon']
        except KeyError:
            # If we can't extract the data, use sample forecast
            return self._generate_sample_forecast()

        # Generate forecast based on current conditions
        forecast = []
        today = datetime.now()

        # Base weather codes for prediction
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

        # Current season affects the forecast
        month = today.month
        is_summer = 5 <= month <= 8
        is_winter = month <= 2 or month == 12

        # Temperature changes based on season (for northern hemisphere)
        if is_summer:
            temp_change = 0.5  # Summer has less variation
            trend_factor = 0.5  # Generally warmer
        elif is_winter:
            temp_change = 0.5  # Winter has less variation
            trend_factor = -0.5  # Generally cooler
        else:
            temp_change = 0.5  # Spring/Fall moderate variation
            trend_factor = 0  # Neutral trend

        # Generate forecast for each day
        for i in range(1, self.days + 1):
            # Date for this forecast
            forecast_date = today + timedelta(days=i)
            date_str = forecast_date.strftime("%Y-%m-%d")

            # Day in forecast (0-indexed)
            day_in_forecast = i - 1

            # Temperature forecast (with trend)
            day_temp = current_temp + (trend_factor * day_in_forecast) + (temp_change * (i % 2))  # Slight oscillation
            night_temp = day_temp - 6  # Nights are cooler by fixed amount

            # Constrain temperatures to reasonable ranges
            day_temp = max(min(day_temp, 45), -30)  # Between -30°C and 45°C
            night_temp = max(min(night_temp, 40), -35)  # Between -35°C and 40°C

            # Weather condition (deterministic pattern based on day in forecast)
            day_mod = day_in_forecast % len(weather_conditions)

            # First day more likely similar to current
            if day_in_forecast == 0:
                condition = next(
                    (w for w in weather_conditions if w["id"] // 100 == current_weather_id // 100),
                    weather_conditions[0]
                )
            # Seasonal patterns
            elif is_summer:
                # Summer: more clear/clouds, some rain, no snow
                condition_idx = day_mod % (len(weather_conditions) - 1)  # Skip snow
                condition = weather_conditions[condition_idx]
            elif is_winter:
                # Winter: more clouds, fog, snow, less clear
                if day_mod == 0:  # Replace clear with clouds
                    condition = weather_conditions[3]  # Broken clouds
                else:
                    condition = weather_conditions[day_mod]
            else:
                # Regular pattern
                condition = weather_conditions[day_mod]

            # Humidity based on weather (higher for rain/snow/fog)
            if condition["id"] in [500, 501, 600, 741, 701]:
                humidity = 80
            else:
                humidity = 60

            # Wind speed - fixed value based on current
            wind_speed = current_wind_speed

            # Chance of precipitation based on weather
            if condition["main"] in ["Rain", "Snow"]:
                precipitation_chance = 80
                precipitation_amount = 5.0 if condition["main"] == "Rain" else 3.0
            elif condition["main"] in ["Clouds", "Fog", "Mist"]:
                precipitation_chance = 40
                precipitation_amount = 1.0
            else:
                precipitation_chance = 10
                precipitation_amount = 0

            # UV index removed as requested

            # Add to forecast
            forecast.append({
                "date": date_str,
                "day_temp": round(day_temp, 1),
                "night_temp": round(night_temp, 1),
                "humidity": humidity,
                "wind_speed": round(wind_speed, 1),
                "pressure": current_pressure,
                "weather_id": condition["id"],
                "weather_main": condition["main"],
                "weather_description": condition["description"],
                "weather_icon": condition["icon"],
                "precipitation_chance": precipitation_chance,
                "precipitation_amount": round(precipitation_amount, 1)
            })

        return forecast

    def _generate_sample_forecast(self):
        """Generate sample weather forecast when API data is unavailable"""
        forecast = []
        today = datetime.now()

        # City-based temperature values (fixed values instead of ranges)
        city_temp_map = {
            "Beijing": {"summer": 30, "winter": 5, "other": 18},
            "London": {"summer": 20, "winter": 5, "other": 12},
        }

        # Use provided city or default to Beijing's values
        city_temps = city_temp_map.get(self.city, city_temp_map["Beijing"])

        # Determine season (based on northern hemisphere, except for Sydney)
        month = today.month
        if self.city == "Sydney":  # Southern hemisphere
            is_summer = month <= 2 or month == 12
            is_winter = 6 <= month <= 8
        else:  # Northern hemisphere
            is_summer = 6 <= month <= 8
            is_winter = month <= 2 or month == 12

        # Select temperature based on season
        if is_summer:
            base_temp = city_temps["summer"]
        elif is_winter:
            base_temp = city_temps["winter"]
        else:
            base_temp = city_temps["other"]

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
        for i in range(1, self.days + 1):
            # Date for this forecast
            forecast_date = today + timedelta(days=i)
            date_str = forecast_date.strftime("%Y-%m-%d")

            # Temperature with slight oscillation but no randomness
            day_temp = base_temp + (0.5 * (i % 2))
            night_temp = day_temp - 6  # Nights are cooler by fixed amount

            # Select weather condition based on deterministic pattern
            day_mod = (i - 1) % len(weather_conditions)

            # Apply seasonal adjustments to pattern
            if is_winter and day_mod == 0:  # Replace clear with snow in winter
                condition = weather_conditions[6]  # Snow
            elif is_summer and day_mod == 6:  # Replace snow with clear in summer
                condition = weather_conditions[0]  # Clear
            else:
                condition = weather_conditions[day_mod]

            # Weather dependent values (fixed)
            if condition["main"] in ["Rain", "Snow"]:
                humidity = 80
                precipitation_chance = 80
                precipitation_amount = 5 if condition["main"] == "Rain" else 3
            elif condition["main"] in ["Clouds", "Fog", "Mist"]:
                humidity = 70
                precipitation_chance = 40
                precipitation_amount = 0.5
            else:
                humidity = 50
                precipitation_chance = 10
                precipitation_amount = 0

            # Add day to forecast
            forecast.append({
                "date": date_str,
                "day_temp": round(day_temp, 1),
                "night_temp": round(night_temp, 1),
                "humidity": humidity,
                "wind_speed": 5.0,  # Fixed wind speed
                "pressure": 1010,  # Fixed pressure
                "weather_id": condition["id"],
                "weather_main": condition["main"],
                "weather_description": condition["description"],
                "weather_icon": condition["icon"],
                "precipitation_chance": precipitation_chance,
                "precipitation_amount": precipitation_amount
            })

        return forecast

    def predict_to_json(self, output_file=None):
        """Run prediction and save results to JSON file"""
        predictions = self.predict()

        if output_file is None:
            output_file = f"{self.city}_{self.country}_weather_forecast_{datetime.now().strftime('%Y-%m-%d')}.json"

        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=4)

        print(f"Weather forecast saved to {output_file}")
        return output_file, predictions

    def visualize_forecast(self, predictions=None):
        """Generate visualization of the forecast"""
        if predictions is None:
            predictions = self.predict()

        # Extract data for plotting
        dates = [day["date"] for day in predictions]
        temps_day = [day["day_temp"] for day in predictions]
        temps_night = [day["night_temp"] for day in predictions]
        humidity = [day["humidity"] for day in predictions]
        pressure = [day["pressure"] for day in predictions]
        wind_speed = [day["wind_speed"] for day in predictions]
        precip_chance = [day["precipitation_chance"] for day in predictions]

        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))

        # Temperature subplot
        axs[0].plot(dates, temps_day, 'ro-', label='Day Temperature (°C)')
        axs[0].plot(dates, temps_night, 'bo-', label='Night Temperature (°C)')
        axs[0].set_title(f'{self.city} Temperature Forecast')
        axs[0].set_ylabel('Temperature (°C)')
        axs[0].grid(True)
        axs[0].legend()

        # Humidity and Precipitation subplot
        axs[1].plot(dates, humidity, 'go-', label='Humidity (%)')
        axs[1].set_ylabel('Humidity (%)')
        axs[1].grid(True)
        ax2 = axs[1].twinx()
        ax2.plot(dates, precip_chance, 'co-', label='Precipitation Chance (%)')
        ax2.set_ylabel('Precipitation Chance (%)')
        axs[1].set_title(f'{self.city} Humidity and Precipitation Forecast')
        lines1, labels1 = axs[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Wind and Pressure subplot
        axs[2].plot(dates, wind_speed, 'mo-', label='Wind Speed (m/s)')
        axs[2].set_ylabel('Wind Speed (m/s)')
        axs[2].grid(True)
        ax3 = axs[2].twinx()
        ax3.plot(dates, pressure, 'yo-', label='Pressure (hPa)')
        ax3.set_ylabel('Pressure (hPa)')
        axs[2].set_title(f'{self.city} Wind and Pressure Forecast')
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Format x-axis for all subplots
        for ax in axs:
            ax.set_xlabel('Date')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        return fig


if __name__ == "__main__":
    # 预测北京天气
    print("Generating forecast for Beijing...")
    beijing_predictor = WeatherPredictor(
        city="Beijing",
        country="CN",
        days=7,
        use_ml_model=True
    )
    beijing_file, beijing_predictions = beijing_predictor.predict_to_json()

    print(f"\nBeijing forecast saved to: {beijing_file}")
    print("Summary: ", end="")
    print(
        f"Day 1: {beijing_predictions[0]['day_temp']}°C/{beijing_predictions[0]['night_temp']}°C")

    print("\nGenerating forecast for London...")
    london_predictor = WeatherPredictor(
        city="London",
        country="GB",
        days=7,
        use_ml_model=True
    )
    london_file, london_predictions = london_predictor.predict_to_json()

    print(f"\nLondon forecast saved to: {london_file}")
    print("Summary: ", end="")
    print(
        f"Day 1: {london_predictions[0]['day_temp']}°C/{london_predictions[0]['night_temp']}°C")

    print("\nForecasts for both cities completed successfully!")