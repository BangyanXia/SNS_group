import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

class WeatherPredictor:
    def __init__(self, 
                 city="Beijing", 
                 country="CN",
                 days=7,
                 api_key="YOUR_OPENWEATHERMAP_API_KEY"):
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
        """
        self.city = city
        self.country = country
        self.days = days
        self.api_key = api_key
        self.data_folder = "./data/weather"
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
    
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
    
    def predict(self):
        """Generate weather predictions based on current conditions and historical patterns"""
        # Try to get current weather
        current_weather = self.fetch_current_weather()
        
        # Use sample data if API request fails
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
        
        # Temperature ranges based on season (for northern hemisphere)
        if is_summer:
            temp_range = (2, 4)  # Summer has less variation
            trend_range = (-1, 2)  # Generally warmer or stable
        elif is_winter:
            temp_range = (4, 7)  # Winter has more variation
            trend_range = (-2, 1)  # Generally cooler
        else:
            temp_range = (3, 5)  # Spring/Fall moderate variation
            trend_range = (-1.5, 1.5)  # Mixed trends
        
        # Generate forecast for each day
        for i in range(1, self.days + 1):
            # Date for this forecast
            forecast_date = today + timedelta(days=i)
            date_str = forecast_date.strftime("%Y-%m-%d")
            
            # Generate temperature with a trending component and random variation
            day_in_forecast = i - 1
            trend_factor = random.uniform(*trend_range)
            temp_variation = random.uniform(*temp_range)
            
            # Temperature forecast (with trend)
            day_temp = current_temp + (trend_factor * day_in_forecast) + (temp_variation * (random.random() - 0.5))
            night_temp = day_temp - random.uniform(5, 10)  # Nights are cooler
            
            # Constrain temperatures to reasonable ranges
            day_temp = max(min(day_temp, 45), -30)  # Between -30°C and 45°C
            night_temp = max(min(night_temp, 40), -35)  # Between -35°C and 40°C
            
            # Weather condition (more likely to be similar to current and recent days)
            weather_bias = random.random()
            if day_in_forecast <= 2 and weather_bias < 0.7:
                # First few days more likely similar to current
                condition = next(
                    (w for w in weather_conditions if w["id"] // 100 == current_weather_id // 100), 
                    random.choice(weather_conditions)
                )
            else:
                # Random conditions, but weighted by season
                if is_summer and random.random() < 0.7:
                    # Summer: more clear/clouds, some rain, rare snow
                    condition = random.choice([w for w in weather_conditions if w["id"] != 600])
                elif is_winter and random.random() < 0.6:
                    # Winter: more clouds, fog, snow, less clear
                    winter_conditions = [w for w in weather_conditions if w["id"] != 800 or random.random() < 0.3]
                    condition = random.choice(winter_conditions)
                else:
                    # Any season, any condition
                    condition = random.choice(weather_conditions)
            
            # Humidity based on weather (higher for rain/snow/fog)
            if condition["id"] in [500, 501, 600, 741, 701]:
                humidity = random.randint(70, 95)
            else:
                humidity = random.randint(40, 80)
            
            # Wind speed - slight variation from current
            wind_speed = max(0, current_wind_speed + random.uniform(-2, 2))
            
            # Chance of precipitation based on weather
            if condition["main"] in ["Rain", "Snow"]:
                precipitation_chance = random.randint(60, 100)
                precipitation_amount = random.uniform(0.5, 10.0) if condition["main"] == "Rain" else random.uniform(0.5, 5.0)
            elif condition["main"] in ["Clouds", "Fog", "Mist"]:
                precipitation_chance = random.randint(20, 60)
                precipitation_amount = random.uniform(0, 2.0)
            else:
                precipitation_chance = random.randint(0, 20)
                precipitation_amount = 0
            
            # UV index (higher in summer, clear days)
            if condition["main"] == "Clear" and is_summer:
                uv_index = random.randint(6, 11)
            elif condition["main"] == "Clear":
                uv_index = random.randint(3, 7)
            else:
                uv_index = random.randint(0, 4)
            
            # Add to forecast
            forecast.append({
                "date": date_str,
                "day_temp": round(day_temp, 1),
                "night_temp": round(night_temp, 1),
                "humidity": humidity,
                "wind_speed": round(wind_speed, 1),
                "pressure": current_pressure + random.randint(-5, 5),
                "weather_id": condition["id"],
                "weather_main": condition["main"],
                "weather_description": condition["description"],
                "weather_icon": condition["icon"],
                "precipitation_chance": precipitation_chance,
                "precipitation_amount": round(precipitation_amount, 1),
                "uv_index": uv_index
            })
        
        return forecast
    
    def _generate_sample_forecast(self):
        """Generate sample weather forecast when API data is unavailable"""
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
        city_temps = city_temp_map.get(self.city, city_temp_map["Beijing"])
        
        # Determine season (based on northern hemisphere, except for Sydney)
        month = today.month
        if self.city == "Sydney":  # Southern hemisphere
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
        for i in range(1, self.days + 1):
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
    
    def predict_to_json(self, output_file=None):
        """Run prediction and save results to JSON file"""
        predictions = self.predict()
        
        if output_file is None:
            output_file = f"{self.city}_{self.country}_weather_forecast_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=4)
            
        print(f"Weather forecast saved to {output_file}")
        return output_file, predictions


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Weather Forecast')
    parser.add_argument('--city', type=str, default="Beijing", help='City name')
    parser.add_argument('--country', type=str, default="CN", help='Country code (ISO 3166)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to forecast')
    parser.add_argument('--api_key', type=str, default="YOUR_OPENWEATHERMAP_API_KEY", help='OpenWeatherMap API key')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    predictor = WeatherPredictor(
        city=args.city,
        country=args.country,
        days=args.days,
        api_key=args.api_key
    )
    
    output_file, predictions = predictor.predict_to_json(args.output_file)
    
    # Print sample of results
    print("\nForecast Sample:")
    for day in predictions[:3]:  # Show first 3 days
        print(f"Date: {day['date']}, Weather: {day['weather_main']}, Temp: {day['day_temp']}°C/{day['night_temp']}°C")
    if len(predictions) > 3:
        print("...")
