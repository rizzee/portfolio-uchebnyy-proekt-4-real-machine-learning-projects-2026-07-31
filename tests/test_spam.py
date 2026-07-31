import unittest
from pathlib import Path
import pandas as pd
import shutil

from ml_models.spam_detection import SpamClassifier

class TestSpamClassifier(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = Path("temp_spam_test_data")
        self.test_dir.mkdir(exist_ok=True)

        # Sample data for testing
        self.sample_data = (
            "text,label\n"
            "free prize win now,spam\n"
            "hey how are you meeting,ham\n"
            "win big money,spam\n"
            "project meeting tomorrow,ham\n"
            "urgent call for free prize,spam\n"
            "let's catch up soon,ham\n"
        )
        self.csv_path = self.test_dir / "spam_samples.csv"
        self.csv_path.write_text(self.sample_data)

        self.classifier = SpamClassifier()

    def tearDown(self):
        # Clean up the temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_and_train(self):
        # Test if data loading and training works
        self.classifier.load_and_train(self.csv_path)

        # A trained scikit-learn model has a 'classes_' attribute
        # For a pipeline, we check the final step (the classifier)
        final_estimator = self.classifier.model.named_steps['classifier']
        self.assertTrue(hasattr(final_estimator, 'classes_'))
        self.assertIn('spam', final_estimator.classes_)
        self.assertIn('ham', final_estimator.classes_)

    def test_predict(self):
        # Train the model first
        df = pd.read_csv(self.csv_path)
        X = df['text']
        y = df['label']
        self.classifier.train(X, y)

        # Test with new text samples
        spam_sample = ["win free money now"]
        ham_sample = ["see you at the meeting"]

        spam_prediction = self.classifier.predict(spam_sample)
        ham_prediction = self.classifier.predict(ham_sample)

        # Check if predictions are correct
        self.assertEqual(spam_prediction[0], 'spam')
        self.assertEqual(ham_prediction[0], 'ham')

    def test_integration_load_train_predict(self):
        # Test the full flow: load, train, and then predict
        self.classifier.load_and_train(self.csv_path)

        new_texts = ["urgent prize money", "how was your day"] 
        predictions = self.classifier.predict(new_texts)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0], 'spam')
        self.assertEqual(predictions[1], 'ham')


if __name__ == '__main__':
    unittest.main()
