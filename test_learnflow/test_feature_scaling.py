import unittest
import numpy as np
import pandas as pd
from learnflow.preprocessing import FeatureScaling

class TestFeatureScaling(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.data = pd.DataFrame({
            'Numeric1': np.random.rand(100),
            'Numeric2': np.random.randint(1, 100, size=100),
            'Numeric3': np.random.normal(0, 1, 100),
            'Category1': np.random.choice(['A', 'B', 'C'], size=100),
            'Category2': np.random.choice(['X', 'Y', 'Z'], size=100)
        })
        self.example = FeatureScaling()

    def test_min_max(self):
        min_max_data = self.example.min_max(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(min_max_data is not None)
        self.assertTrue('Numeric1' in min_max_data.columns)
        self.assertTrue('Numeric2' in min_max_data.columns)
        self.assertTrue('Numeric3' in min_max_data.columns)

    def test_z_score(self):
        z_score_data = self.example.z_score(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(z_score_data is not None)
        self.assertTrue('Numeric1' in z_score_data.columns)
        self.assertTrue('Numeric2' in z_score_data.columns)
        self.assertTrue('Numeric3' in z_score_data.columns)

    def test_robust(self):
        robust_data = self.example.robust(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(robust_data is not None)
        self.assertTrue('Numeric1' in robust_data.columns)
        self.assertTrue('Numeric2' in robust_data.columns)
        self.assertTrue('Numeric3' in robust_data.columns)

    def test_abs_max(self):
        abs_max_data = self.example.abs_max(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(abs_max_data is not None)
        self.assertTrue('Numeric1' in abs_max_data.columns)
        self.assertTrue('Numeric2' in abs_max_data.columns)
        self.assertTrue('Numeric3' in abs_max_data.columns)

    def test_pow_transform(self):
        pow_transform_data = self.example.pow_transform(data=self.data, lam=2, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(pow_transform_data is not None)
        self.assertTrue('Numeric1' in pow_transform_data.columns)
        self.assertTrue('Numeric2' in pow_transform_data.columns)
        self.assertTrue('Numeric3' in pow_transform_data.columns)

    def test_unit_vector(self):
        unit_vector_data = self.example.unit_vector(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(unit_vector_data is not None)
        self.assertTrue('Numeric1' in unit_vector_data.columns)
        self.assertTrue('Numeric2' in unit_vector_data.columns)
        self.assertTrue('Numeric3' in unit_vector_data.columns)

    def test_log_transform(self):
        log_transform_data = self.example.log_transform(data=self.data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(log_transform_data is not None)
        self.assertTrue('Numeric1' in log_transform_data.columns)
        self.assertTrue('Numeric2' in log_transform_data.columns)
        self.assertTrue('Numeric3' in log_transform_data.columns)

    def test_box_cox(self):
        box_cox_data = self.example.box_cox(data=self.data, lam=2, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(box_cox_data is not None)
        self.assertTrue('Numeric1' in box_cox_data.columns)
        self.assertTrue('Numeric2' in box_cox_data.columns)
        self.assertTrue('Numeric3' in box_cox_data.columns)

    def test_yeo_johnson(self):
        yeo_johnson_data = self.example.yeo_johnson(data=self.data, lam=2, columns=['Numeric1', 'Numeric2', 'Numeric3'])
        self.assertTrue(yeo_johnson_data is not None)
        self.assertTrue('Numeric1' in yeo_johnson_data.columns)
        self.assertTrue('Numeric2' in yeo_johnson_data.columns)
        self.assertTrue('Numeric3' in yeo_johnson_data.columns)

if __name__ == '__main__':
    unittest.main()

