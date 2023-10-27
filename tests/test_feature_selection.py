import unittest
import pandas as pd
from smartsolve.preprocessing import SelectFeature




class TestSelectFeature(unittest.TestCase):
    def setUp(self):
        # Load test data files
        self.numerical_data = pd.read_csv('your_file.csv')
        self.categorical_data = pd.read_csv('categorical_data.csv')
        self.example_data = pd.read_csv('sample_data.csv')



    def test_correlation(self):
        # Test correlation selection method
        selector = SelectFeature(csv_file='sample_data.csv', label='a')
        features, data = selector.correlation(threshold=0.7)
        self.assertEqual(len(features), 2)  # Check if 2 features are selected


    def test_mutual_infos(self):
        # Test mutual information calculation
        selector = SelectFeature(csv_file='sample_data.csv', label='a')
        mutual_info = selector.mutual_infos()
        self.assertEqual(len(mutual_info), len(self.example_data.columns) - 1)  # Check if mutual info is computed for all features


    def test_lasso(self):
        # Test Lasso regularization
        example_coefficients = [0.2, -0.3, 0.5, 0.1, -0.7, 0.4, 0.6, -0.2, 0.8, -0.9, 0.2, -0.3, 0.5, 0.1, -0.7, 0.4, 0.6, -0.2, 0.8, -0.9]
        selector = SelectFeature()
        new_coefficients = selector.lasso(example_coefficients, lam=0.01, learning_rate=0.01, threshold=1e-4)
        self.assertTrue(len(new_coefficients) == len(example_coefficients))


    def test_best_features(self):
        # Test best features selection
        selector = SelectFeature(csv_file='sample_data.csv', label='a')
        best_features = selector.best_features(k=2, data_type='numerical')
        self.assertEqual(len(best_features), 2)  # Check if 2 features are selected


    def test_best_features_categorical(self):
        # Test best features selection with categorical data
        selector = SelectFeature(csv_file='categorical_data.csv', label='CategoryA')
        best_features = selector.best_features(k=2, data_type='categorical')
        self.assertEqual(len(best_features), 2)  # Check if 2 features are selected


    def test_variance_threshold(self):
        # Test variance threshold
        selector = SelectFeature(csv_file='sample_data.csv', label='Label')
        threshold = 0.1
        filtered_data = selector.variance_threshold(threshold=threshold)
        self.assertLess(len(filtered_data.columns), len(self.numerical_data.columns))  # Check if low-variance features are removed



if __name__ == '__main__':
    unittest.main()
