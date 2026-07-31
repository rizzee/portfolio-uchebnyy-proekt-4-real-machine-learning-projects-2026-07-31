import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

class TimeSeriesForecaster:
    def __init__(self):
        self.model = LinearRegression()
        self.data = None

    def load_data(self, path: Path):
        # Load time series data
        self.data = pd.read_csv(path, parse_dates=['date'])

    def prepare_features(self):
        # Convert dates to numerical features
        self.data['days'] = (self.data['date'] - self.data['date'].min()).dt.days
        X = self.data[['days']]
        y = self.data['value']
        self.X = X
        self.y = y
        return X, y

    def train(self):
        # Train the model
        X, y = self.prepare_features()
        self.model.fit(X, y)

    def predict(self, days_ahead: int):
        # Make future predictions
        last_day = self.data['days'].max()
        future_days = pd.DataFrame({'days': range(last_day + 1, last_day + days_ahead + 1)})
        return self.model.predict(future_days)
