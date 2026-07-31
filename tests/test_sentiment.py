import unittest
from pathlib import Path
import pandas as pd

from ml_models.sentiment_analysis import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and data file for testing
        self.temp_dir = Path("temp_test_data")
        self.temp_dir.mkdir(exist_ok=True)
        self.csv_path = self.temp_dir / "test_sentiment.csv"

        # Sample data for the test CSV
        self.sample_data = (
            "text,sentiment\n"
            "'I love this product, it's amazing!',positive\n"
            "'This is the worst experience ever.',negative\n"
            "'It's okay, not great but not bad.',neutral\n"
            "'Absolutely fantastic service.',positive\n"
            "'I am very disappointed.',negative\n"
            "'The quality is acceptable.',neutral\n"
        )
        self.csv_path.write_text(self.sample_data)

        self.analyzer = SentimentAnalyzer()

    def tearDown(self):
        # Clean up the temporary file and directory
        self.csv_path.unlink()
        self.temp_dir.rmdir()

    def test_load_data(self):
        # Test if data is loaded correctly
        self.analyzer.load_data(self.csv_path)
        self.assertIsNotNone(self.analyzer.data)
        self.assertIsInstance(self.analyzer.data, pd.DataFrame)
        self.assertEqual(len(self.analyzer.data), 6)
        self.assertEqual(list(self.analyzer.data.columns), ["text", "sentiment"])

    def test_train(self):
        # Test the training process
        self.analyzer.load_data(self.csv_path)
        self.analyzer.train()
        # A trained model should have learned classes
        self.assertIn("positive", self.analyzer.model.classes_)
        self.assertIn("negative", self.analyzer.model.classes_)
        self.assertIn("neutral", self.analyzer.model.classes_)

    def test_predict(self):
        # Test the prediction functionality
        self.analyzer.load_data(self.csv_path)
        self.analyzer.train()

        test_texts = [
            "This is a truly wonderful and great thing.",
            "I hate this, it is awful and terrible.",
            "It is mediocre."
        ]

        predictions = self.analyzer.predict(test_texts)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 3)
        
        # Based on our simple training data, these are the expected results
        expected_predictions = ['positive', 'negative', 'neutral']
        self.assertEqual(predictions, expected_predictions)

    def test_predict_single_text(self):
        # Test prediction with a single string instead of a list
        self.analyzer.load_data(self.csv_path)
        self.analyzer.train()

        prediction = self.analyzer.predict("This is a fantastic product!")
        self.assertIsInstance(prediction, list)
        self.assertEqual(prediction, ['positive'])

if __name__ == '__main__':
    unittest.main()
