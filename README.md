# Flight Price Predictor

A web application that predicts whether a flight price is above the median using machine learning (Logistic Regression). Users enter flight details and get instant predictions.

## Features

- Predicts flight price category (above/below median) based on user input
- Uses scikit-learn Logistic Regression model
- Handles categorical and numerical data
- Simple web interface built with Flask

## How It Works

1. User enters flight details (airline, source city, departure time, stops, etc.)
2. The app processes the input and encodes categorical features
3. The trained ML model predicts if the price is above the median
4. The result is displayed on the web page

## Project Structure

```
Flight_ML_App/
│
├── ML_Model/
│   ├── airlines_flights_data.csv
│   ├── flightdataML.py
│   └── flight_price_logistic_model.pkl
│
├── templates/
│   └── index.html
│
├── app.py
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

## Model Details

- **Algorithm:** Logistic Regression
- **Features:** airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left
- **Target:** Whether price is above the median

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## License

MIT License

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [pandas](https://pandas.pydata.org/)
