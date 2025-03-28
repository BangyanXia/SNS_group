# Oracle forecast chatbot

A comprehensive forecasting chatbot that provides predictions for weather, stock prices, and exchange rates using machine learning models.

## Overview

Forecast Assistant is a web-based application that allows users to:
- Get weather forecasts for cities around the world
- View stock price predictions for various ticker symbols
- Check exchange rate forecasts for currency pairs
- Interact through a user-friendly chatbot interface

The system uses machine learning models (iTransformer architecture) to analyze historical data and generate accurate predictions for various forecasting tasks.

## Features

### Weather Forecasting
- 7-day weather predictions for global cities
- Visualization of temperature trends, precipitation chance, and amount
- Statistical summaries of forecasted weather conditions
- Weather condition icons and detailed daily forecast cards

### Stock Price Prediction
- Price forecasts for stock tickers
- Candlestick chart visualization
- Volume prediction and analysis
- Performance statistics and detailed data tables

### Exchange Rate Forecasting
- Currency pair exchange rate predictions
- Trend charts and candlestick visualizations
- Statistical analysis of forecast data
- Daily rate change calculations

### Interactive Chatbot
- Natural language interface for accessing forecasts
- Quick action shortcuts for common requests
- Suggestion chips for guiding user interactions
- History tracking of recent searches

## Project Structure
### Directory layout 
```
├── backend/
│   ├── flask-backend.py                # Main Flask server
│   ├── exchangerate_prediction/        # Exchange rate prediction models
│   │   └── iTransformer_exchangerate/   
│   │       └── exchange_rate_predictor.py
│   ├── stock_prediction/               # Stock price prediction models
│   │   └── iTransformer_main/
│   │       └── stock_price_predictor.py
│   └── weather_prediction/             # Weather prediction models
│       └── weather_predictor.py
├── frontend/
│   ├── forecast-chatbot.html           # Main chatbot interface
│   ├── exchange-rate-visualization.html # Exchange rate visualization
│   ├── stock-price-visualization.html   # Stock price visualization
│   └── weather-visualization.html       # Weather visualization
```
### System Architecture
![Project Architecture](/system_architecture.png)

## Technologies Used

- **Backend**:
  - Python 3.x
  - Flask web framework
  - PyTorch for ML model implementation
  - iTransformer architecture for time series forecasting
  - Alpha Vantage API for financial data

- **Frontend**:
  - HTML5, CSS3, JavaScript
  - Chart.js for data visualization
  - Luxon for date/time handling
  - Responsive design for cross-device compatibility

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Flask
- Alpha Vantage API key (for stock and exchange rate data)
- OpenWeatherMap API key (for weather data)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/forecast-assistant.git
   cd forecast-assistant
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Update the API keys in the respective predictor files:
   - `backend/exchangerate_prediction/iTransformer_exchangerate/exchange_rate_predictor.py`
   - `backend/stock_prediction/iTransformer_main/stock_price_predictor.py`
   - `backend/weather_prediction/weather_predictor.py`

### Running the Application

1. Start the Flask server:
   ```
   python backend/flask-backend.py
   ```

2. Open your browser and go to:
   ```
   http://localhost:5000
   ```

3. Interact with the chatbot to get various forecasts or navigate to the specific visualization pages:
   - Weather: `http://localhost:5000/weather`
   - Stocks: `http://localhost:5000/stocks`
   - Exchange Rates: `http://localhost:5000/exchange`

## API Endpoints

- `/api/weather-forecast` - Get weather forecast for a specific city
  - Parameters: `city`, `country`, `days` (optional)

- `/api/stock-forecast` - Get stock price prediction for a ticker
  - Parameters: `ticker`, `days` (optional)

- `/api/exchange-rate-forecast` - Get exchange rate prediction for a currency pair
  - Parameters: `from_symbol`, `to_symbol`

## Machine Learning Models

The project uses the iTransformer architecture for time series forecasting, which is particularly effective for financial and weather data prediction. The models are pre-trained and loaded at runtime.

- **Weather Model**: Uses LSTM networks to predict temperature, humidity, wind speed, and pressure
- **Stock Model**: Uses iTransformer to predict Open, High, Low, Close prices and volume
- **Exchange Rate Model**: Uses iTransformer to predict currency exchange rates

## Fallback Mechanisms

All prediction modules include fallback mechanisms that generate reasonable predictions when:
- API data is unavailable
- ML models fail to load
- Unexpected errors occur

This ensures the application remains functional even when external services are unavailable.

## Future Enhancements

- Add user accounts for personalized forecasts and favorites
- Implement more sophisticated ML models for improved accuracy
- Add more visualization options for deeper data analysis
- Expand to include more forecast types (e.g., commodity prices, energy consumption)
- Develop mobile applications for iOS and Android

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- iTransformer paper and implementation
- Alpha Vantage for financial data API
- OpenWeatherMap for weather data API
- Chart.js for visualization components
