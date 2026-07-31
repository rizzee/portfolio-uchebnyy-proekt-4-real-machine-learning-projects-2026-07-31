import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pathlib import Path
from typing import Union, List


class SpamClassifier:
    """Basic email spam classifier using Naive Bayes."""
    
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load and return email data from CSV file."""
        return pd.read_csv(data_path)

    def load_and_train(self, data_path: Path):
        """Loads data from a file and trains the model."""
        data = self.load_data(data_path)
        # The provided test data and sample data use 'text' and 'label' columns
        data.dropna(subset=['text', 'label'], inplace=True)
        X = data['text']
        y = data['label']
        self.train(X, y)
    
    def train(self, X, y):
        """Train the classifier."""
        self.model.fit(X, y)
    
    def predict(self, email_text: Union[str, List[str]]) -> List[str]:
        """Predict if text(s) are spam or ham."""
        if isinstance(email_text, str):
            email_text = [email_text]
        return self.model.predict(email_text).tolist()
    
    def evaluate(self, X_test, y_test) -> float:
        """Return accuracy score on test data."""
        return self.model.score(X_test, y_test)


if __name__ == "__main__":
    # Example usage
    data_path = Path("data/sample_data/spam_samples.csv")
    classifier = SpamClassifier()
    
    data = classifier.load_data(data_path)
    X = data["text"]
    y = data["label"] # Corrected column name from 'is_spam'
    
    classifier.train(X, y)
    print(f"Accuracy: {classifier.evaluate(X, y)}")
    print(f"Prediction: {classifier.predict('Win a free prize now!')}")
