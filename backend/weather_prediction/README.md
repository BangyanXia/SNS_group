# Weather Forecasting System

This repository contains a comprehensive weather forecasting system that combines API data, machine learning models, and visualization tools to provide accurate weather predictions.

## Overview

The system supports two primary locations (Beijing and London) with dedicated machine learning models, while also offering rule-based predictions for any location worldwide. The visualization dashboard provides an interactive interface for exploring forecast data.

## Features

- **ML-Powered Forecasting**: LSTM models for Beijing and London provide accurate 7-day forecasts
- **Worldwide Coverage**: Rule-based forecasting available for any location
- **Interactive Visualization**: Web dashboard for exploring temperature, precipitation, and other metrics
- **API Integration**: OpenWeatherMap integration for current weather data
- **Caching System**: Efficient caching to reduce redundant calculations
- **Export Capabilities**: JSON and visualization outputs

## Components

### Main Files

- `weather_predictor.py`: Core prediction engine that combines ML models and rule-based forecasting
- `predict_next_week_B.py`: Beijing-specific LSTM model implementation
- `predict_next_week_L.py`: London-specific LSTM model implementation
- `weather-visualization.html`: Web-based visualization dashboard
- `weather-visualize.py`: Flask server to handle API requests and serve the visualization dashboard

### Model Files

- `Beijing/weather_lstm_model_B.pth`: Trained LSTM model for Beijing
- `Beijing/X_test_tensor_B.pth`: Test tensor data for Beijing
- `Beijing/scaler_y_B.pkl`: Scaler for inverse transform (Beijing)
- `London/weather_lstm_model_L.pth`: Trained LSTM model for London
- `London/X_test_tensor_L.pth`: Test tensor data for London
- `London/scaler_y_L.pkl`: Scaler for inverse transform (London)

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Flask
- NumPy, Pandas, Matplotlib
- Joblib

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install torch numpy pandas matplotlib flask joblib
   ```

### Usage

#### Running the Visualization Server

```bash
python weather-visualize.py
```

This will:
- Start a Flask server on port 5000
- Generate initial forecasts for Beijing and London
- Serve the visualization dashboard at http://localhost:5000

#### Generating Forecasts from Command Line

```bash
# Generate forecasts for both Beijing and London
python weather_predictor.py

# Generate forecast for a specific city
python weather_predictor.py --city London --country GB --use_ml_model
```

#### API Endpoints

- `/` - Weather visualization dashboard
- `/api/weather-forecast?city=<city>&country=<country>` - Get weather forecast data
- `/generate-forecast` - Generate and save forecast files for Beijing and London

## Visualization Dashboard

The dashboard provides:

- 7-day forecast cards with weather icons
- Temperature charts (day and night)
- Precipitation chance and amount charts
- Statistical summaries
- Detailed data tables

## How It Works

1. **Data Collection**: Fetch current weather from OpenWeatherMap when available
2. **Prediction**:
   - If ML model is available for city (Beijing or London), use it
   - Otherwise, use rule-based prediction with appropriate seasonal adjustments
3. **Visualization**: Present results through interactive charts and tables

## Customization

### Adding New Cities with ML Models

1. Train a new LSTM model following the pattern in the Beijing/London model files
2. Add model files to a city-specific directory
3. Update the WeatherPredictor class to recognize the new city

### Modifying Prediction Parameters

Adjust parameters in the `WeatherPredictor` class initialization:

```python
predictor = WeatherPredictor(
    city="Tokyo",
    country="JP",
    days=10,  # Increase forecast days
    use_ml_model=False  # Force rule-based prediction
)
```

## Troubleshooting

- **Import Errors**: Make sure the current directory is in your Python path
- **Model Loading Errors**: Verify model files exist in the correct directories
- **API Errors**: Check internet connectivity and API key validity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.







