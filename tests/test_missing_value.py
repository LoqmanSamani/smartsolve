import unittest
from smartsolve import preprocessing as pre
import pandas as pd
from machine_learning.linear_algebra import intro_numpy as np


class TestMissingValue(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with missing values for testing
        self.dataframe = pd.DataFrame({'Name': ['ALI', 'Amin', 'Saman', 'hashem', np.nan],
                                       'age': [21, np.nan, np.nan, 65, 12],
                                       'single': ['no', 'no', np.nan, 'yes', np.nan]
                                       })

    def test_load_data(self):
        example = pre.MissingValue(data="sample_data.csv")
        data = example.load_data()
        self.assertIsInstance(data, pd.DataFrame)



    def test_numerical_null(self):
        example = pre.MissingValue(data=self.dataframe, replace='Null')
        modified = example.numerical()
        self.assertIsInstance(modified, pd.DataFrame)
        self.assertEqual(modified.isnull().sum().sum(), 0)  # Check if all missing values are replaced with 0

    def test_numerical_value(self):
        example = pre.MissingValue(data=self.dataframe, replace='Value', rep_value='replaced!')
        modified = example.numerical()
        self.assertIsInstance(modified, pd.DataFrame)
        self.assertEqual(modified.isnull().sum().sum(), 0)  # Check if all missing values are replaced with 'replaced!'

    def test_numerical_del(self):
        example = pre.MissingValue(data=self.dataframe, replace='Del')
        modified = example.numerical()
        self.assertIsInstance(modified, pd.DataFrame)
        self.assertFalse(modified.isnull().any().any())  # Check if all rows with missing values are removed

    def test_qualitative_str(self):
        example = pre.MissingValue(data=self.dataframe, replace='Str', rep_str='replaced!')
        modified = example.qualitative()
        self.assertIsInstance(modified, pd.DataFrame)
        self.assertEqual(modified.isnull().sum().sum(), 0)  # Check if all missing values are replaced with 'replaced!'

    def test_qualitative_del(self):
        example = pre.MissingValue(data=self.dataframe, replace='Del')
        modified = example.qualitative()
        self.assertIsInstance(modified, pd.DataFrame)
        self.assertFalse(modified.isnull().any().any())  # Check if all rows with missing values are removed


if __name__ == '__main__':
    unittest.main()


