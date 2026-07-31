import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.data = None

    def load_data(self, data_path: Path):
        """Load data from CSV file"""
        self.data = pd.read_csv(data_path)

    def preprocess(self, df):
        """Prepare data for training"""
        X = self.vectorizer.fit_transform(df['text'])
        y = df['sentiment']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        """Train the sentiment analysis model on loaded data."""
        if self.data is None:
            raise RuntimeError("Data must be loaded before training. Call load_data() first.")
        
        X = self.vectorizer.fit_transform(self.data['text'])
        y = self.data['sentiment']
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def predict(self, text_input):
        """Predict sentiment for new text or a list of texts."""
        if isinstance(text_input, str):
            text_input = [text_input]

        vector = self.vectorizer.transform(text_input)
        return self.model.predict(vector).tolist()
