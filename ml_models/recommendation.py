import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

class Recommender:
    def __init__(self):
        self.model = NearestNeighbors()
        self.user_data = None

    def load_data(self, path: Path):
        """Load user preferences data"""
        self.user_data = pd.read_csv(path)

    def fit(self):
        """Train the recommendation model"""
        self.model.fit(self.user_data.drop('user_id', axis=1))

    def recommend(self, user_id, n_recommendations):
        """Get recommendations for a user"""
        user_row = self.user_data[self.user_data['user_id'] == user_id]
        if user_row.empty:
            return []
        
        # n_recommendations + 1 because the user itself is included in neighbors
        k = min(n_recommendations + 1, len(self.user_data))
        
        _, indices = self.model.kneighbors(user_row.drop('user_id', axis=1), n_neighbors=k)
        
        recommended_users = self.user_data.iloc[indices[0]]['user_id'].tolist()
        
        # Remove the user itself from recommendations
        if user_id in recommended_users:
            recommended_users.remove(user_id)

        return recommended_users[:n_recommendations]