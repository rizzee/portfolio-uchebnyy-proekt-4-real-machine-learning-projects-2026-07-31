import unittest
from pathlib import Path
import pandas as pd
import shutil
from io import StringIO

from ml_models.recommendation import Recommender

class TestRecommender(unittest.TestCase):
    """Tests for the Recommender class."""

    def setUp(self):
        """Set up a temporary directory and sample data for tests."""
        self.test_dir = Path("temp_recommender_test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.csv_path = self.test_dir / "user_preferences.csv"

        # Sample data representing user preferences for items.
        self.sample_data_content = (
            "user_id,item1,item2,item3,item4\n"
            "1,5,1,3,1\n"  # Most similar to user 3
            "2,1,5,1,5\n"  # Most similar to user 4
            "3,5,2,3,1\n"
            "4,1,4,2,5\n"
            "5,3,3,3,3\n"  # Equidistant from others
        )
        self.csv_path.write_text(self.sample_data_content)
        
        self.recommender = Recommender()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_load_data(self):
        """Test that data is loaded correctly from a CSV file."""
        self.recommender.load_data(self.csv_path)
        
        # Create expected dataframe from the same string
        expected_df = pd.read_csv(StringIO(self.sample_data_content))
        
        # Use pandas testing utility to compare dataframes
        pd.testing.assert_frame_equal(self.recommender.user_data, expected_df)

    def test_fit(self):
        """Test that the NearestNeighbors model is fitted."""
        self.recommender.load_data(self.csv_path)
        self.recommender.fit()

        # Check if the model has been trained by inspecting an internal attribute
        self.assertTrue(hasattr(self.recommender.model, '_fit_X'))
        self.assertIsNotNone(self.recommender.model._fit_X)
        # 5 users, 4 item features
        self.assertEqual(self.recommender.model._fit_X.shape, (5, 4))

    def test_recommend_single(self):
        """Test recommending a single most similar user."""
        self.recommender.load_data(self.csv_path)
        self.recommender.fit()
        
        # User 1 is most similar to User 3
        recommendations = self.recommender.recommend(user_id=1, n_recommendations=1)
        self.assertEqual(recommendations, [3])

        # User 2 is most similar to User 4
        recommendations = self.recommender.recommend(user_id=2, n_recommendations=1)
        self.assertEqual(recommendations, [4])

    def test_recommend_multiple_ordered(self):
        """Test recommending multiple users in order of similarity."""
        self.recommender.load_data(self.csv_path)
        self.recommender.fit()
        
        # For user 1, the order of similarity is 3, then 5.
        recommendations = self.recommender.recommend(user_id=1, n_recommendations=2)
        self.assertEqual(recommendations, [3, 5])

    def test_recommend_all_others(self):
        """Test requesting more recommendations than available users."""
        self.recommender.load_data(self.csv_path)
        self.recommender.fit()

        # Should return all other users, ordered by similarity.
        recommendations = self.recommender.recommend(user_id=1, n_recommendations=10)
        self.assertEqual(len(recommendations), 4) # There are 4 other users
        self.assertEqual(recommendations, [3, 5, 4, 2])
        self.assertNotIn(1, recommendations) # User should not be in their own recommendations

    def test_recommend_non_existent_user(self):
        """Test recommending for a user_id that does not exist."""
        self.recommender.load_data(self.csv_path)
        self.recommender.fit()
        
        recommendations = self.recommender.recommend(user_id=99, n_recommendations=3)
        self.assertEqual(recommendations, [])

if __name__ == '__main__':
    unittest.main()
