import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from sklearn.exceptions import NotFittedError

from ml_models.time_series import TimeSeriesForecaster

class TestTimeSeriesForecaster(unittest.TestCase):
    """Tests for the TimeSeriesForecaster class."""

    def setUp(self):
        """Set up a temporary directory and sample data for tests."""
        self.test_dir = Path("temp_timeseries_test_data")
        self.test_dir.mkdir(exist_ok=True)

        # Sample data with a clear linear trend
        self.csv_content = (
            "date,value\n"
            "2023-01-01,10\n"
            "2023-01-02,20\n"
            "2023-01-03,30\n"
            "2023-01-04,40\n"
            "2023-01-05,50\n"
        )
        self.csv_path = self.test_dir / "time_series_samples.csv"
        self.csv_path.write_text(self.csv_content)

        self.forecaster = TimeSeriesForecaster()

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_data(self):
        """Test that data is loaded correctly from a CSV file."""
        self.forecaster.load_data(self.csv_path)
        self.assertIsNotNone(self.forecaster.data)
        self.assertIsInstance(self.forecaster.data, pd.DataFrame)
        self.assertEqual(self.forecaster.data.shape, (5, 2))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.forecaster.data['date']))

    def test_prepare_features(self):
        """Test feature preparation for the model."""
        self.forecaster.load_data(self.csv_path)
        self.forecaster.prepare_features()
        
        # Check if X and y are created
        self.assertIsNotNone(self.forecaster.X)
        self.assertIsNotNone(self.forecaster.y)
        
        # Check shapes
        self.assertEqual(self.forecaster.X.shape, (5, 1))
        self.assertEqual(self.forecaster.y.shape, (5,))

        # Check content of the feature 'time_idx'
        expected_X = np.array([[0], [1], [2], [3], [4]])
        np.testing.assert_array_equal(self.forecaster.X.values, expected_X)

    def test_train_and_predict(self):
        """Test the full train and predict cycle."""
        self.forecaster.load_data(self.csv_path)
        self.forecaster.prepare_features()
        self.forecaster.train()
        
        future_periods = 3
        predictions = self.forecaster.predict(future_periods)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (future_periods,))
        
        # Based on the linear data y = 10*x + 10 (where x is days from start)
        # Training data 'time_idx' goes from 0 to 4.
        # The model should learn the intercept is 10 and slope is 10.
        # Predictions will be for time_idx 5, 6, 7.
        expected_predictions = np.array([60.0, 70.0, 80.0])
        
        np.testing.assert_allclose(predictions, expected_predictions, rtol=1e-5)

    def test_predict_before_train(self):
        """Test that predict raises an error if called before training."""
        self.forecaster.load_data(self.csv_path)
        self.forecaster.prepare_features()
        with self.assertRaises(NotFittedError):
            self.forecaster.predict(2)

if __name__ == '__main__':
    unittest.main()
