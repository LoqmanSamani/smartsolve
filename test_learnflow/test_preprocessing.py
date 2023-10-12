import unittest
from learnflow import preprocessing as pre
import random
import pandas as pd
import numpy as np




# Define a sample dataset for testing
path = "sample_data.csv"

# Read the data from the CSV file into a DataFrame
data = pd.read_csv(path)

# Extract labels from the 'Rank' column
labels = data['Rank']

# Select features for training
features = data[['Categories', 'Suscribers', 'Country', 'Visits', 'Comments']]

# Prepare features as a list of lists
lst_features = []
for index, row in features.iterrows():
    point = list(row)
    lst_features.append(point)

sample_data = list(zip(labels, lst_features))



# Split data into data points for stratified splitting
data_points = [sample_data[:200], sample_data[200:450], sample_data[450:920], sample_data[920:]]








class TestSplitData(unittest.TestCase):

    def test_random(self):
        # Create a SplitData instance with sample data
        data_splitter = pre.SplitData(data=sample_data, method='Random', train=0.8, validation=0.1, test=0.1)

        # Call the random method to perform the split
        train_data, validation_data, test_data = data_splitter.random()

        # Check if the data has been split correctly
        total_data_count = len(train_data) + len(validation_data) + len(test_data)

        # Check the proportions of the split (adjust for potential rounding issues)
        expected_train_count = len(sample_data) * 0.8
        expected_validation_count = len(sample_data) * 0.1
        expected_test_count = len(sample_data) * 0.1

        self.assertAlmostEqual(len(train_data), expected_train_count, delta=1)
        self.assertAlmostEqual(len(validation_data), expected_validation_count, delta=1)
        self.assertAlmostEqual(len(test_data), expected_test_count, delta=1)

    def test_stratified(self):
        # Create a SplitData instance with sample data
        data_splitter = pre.SplitData(data=data_points, method='Stratified', train=0.7, validation=0.1, test=0.2)

        # Call the stratified method to perform the split
        train_data, validation_data, test_data = data_splitter.stratified()

        # Check if the data has been split correctly
        total_data_count = len(train_data) + len(validation_data) + len(test_data)

        # Check the proportions of the split (adjust for potential rounding issues)
        expected_train_count = len(sample_data) * 0.7
        expected_validation_count = len(sample_data) * 0.1
        expected_test_count = len(sample_data) * 0.2

        self.assertAlmostEqual(len(train_data), expected_train_count, delta=1)
        self.assertAlmostEqual(len(validation_data), expected_validation_count, delta=1)
        self.assertAlmostEqual(len(test_data), expected_test_count, delta=1)

    def test_time_series(self):
        # Create a SplitData instance with sample data
        date_values = [random.randint(1, 10) for _ in sample_data]  # Simulated date values
        data_splitter = pre.SplitData(data=sample_data, method='TimeSeries', train=0.8, validation=0.1, test=0.1,
                                      date=date_values)

        # Call the time_series method to perform the split
        train_data, validation_data, test_data = data_splitter.time_series()

        # Check if the data has been split correctly
        total_data_count = len(train_data) + len(validation_data) + len(test_data)

        # Check the proportions of the split (adjust for potential rounding issues)
        expected_train_count = len(sample_data) * 0.8
        expected_validation_count = len(sample_data) * 0.1
        expected_test_count = len(sample_data) * 0.1

        self.assertAlmostEqual(len(train_data), expected_train_count, delta=1)
        self.assertAlmostEqual(len(validation_data), expected_validation_count, delta=1)
        self.assertAlmostEqual(len(test_data), expected_test_count, delta=1)

    def test_cross_validation(self):
        # Create a SplitData instance with sample data
        data_splitter = pre.SplitData(data=sample_data, method='KFold', num_folds=5)

        # Call the cross_validation method to perform the split
        train_data, validation_data, test_data = data_splitter.cross_validation()

        # Check if the data has been split correctly
        total_data_count = len(train_data[0]) + len(test_data[0])

        # Check the proportions of the split (adjust for potential rounding issues)
        expected_train_count = len(sample_data) * (1 - 1 / data_splitter.num_folds)
        expected_validation_count = 0
        expected_test_count = len(sample_data) / data_splitter.num_folds

        self.assertAlmostEqual(len(train_data[0]), expected_train_count, delta=1)
        self.assertEqual(len(validation_data), expected_validation_count)
        self.assertAlmostEqual(len(test_data[0]), expected_test_count, delta=1)



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


