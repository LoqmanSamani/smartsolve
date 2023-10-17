import unittest
import pandas as pd
from learnflow.preprocessing import CategoricalData


class TestCategoricalData(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame({
            'Category1': ['A', 'B', 'C', 'A', 'B', 'G', 'A', 'B', 'G'],
            'Category2': ['X', 'Y', 'X', 'Y', 'Z', 'Z', 'Z', 'Z', 'X'],
            'Category3': ['A', 'X', 'X', 'Y', 'Z', 'B', 'B', 'Q', 'Q']
        })
        self.labels = ['A', 'B', 'G', 'C', 'X', 'Q', 'Z', 'Y']

    def test_label_encoding(self):
        example = CategoricalData()
        label_encoding = example.l_encoding(self.data, labels=self.labels)

        # Define the expected output
        expected_output = pd.DataFrame({
            'Category1': [1, 2, 4, 1, 2, 3, 1, 2, 3],
            'Category2': [5, 8, 5, 8, 7, 7, 7, 7, 5],
            'Category3': [1, 5, 5, 8, 7, 2, 2, 6, 6]
        })


        pd.testing.assert_frame_equal(label_encoding, expected_output)

    def test_onehot_encoding(self):
        example = CategoricalData()
        one_hot_encoding = example.onehot_encoding(self.data)

        # Define the expected output
        expected_output = pd.DataFrame({
            'Category1_B': [0, 1, 0, 0, 1, 0, 0, 1, 0],
            'Category1_G': [0, 0, 0, 0, 0, 1, 0, 0, 1],
            'Category1_C': [0, 0, 1, 0, 0, 0, 0, 0, 0],

        })


        pd.testing.assert_frame_equal(one_hot_encoding[['Category1_B', 'Category1_G', 'Category1_C']], expected_output)

    def test_bin_encoding(self):
        example = CategoricalData()
        binary_encoding = example.bin_encoding(self.data, labels=self.labels)

        # Define the expected output
        expected_output = pd.DataFrame({
            'Category1': ['1', '10', '100', '1', '10', '11', '1', '10', '11'],
            'Category2': ['101', '1000', '101', '1000', '111', '111', '111', '111', '101'],
        })


        pd.testing.assert_frame_equal(binary_encoding[['Category1', 'Category2']], expected_output)

    def test_count_encoding(self):
        example = CategoricalData()
        count_encoding = example.count_encoding(self.data)


        expected_output = {
            'Category1': [('B', 3), ('C', 1), ('A', 3), ('G', 2)],
            'Category2': [('Y', 2), ('X', 3), ('Z', 4)],
            'Category3': [('Z', 1), ('X', 2), ('A', 1), ('Y', 1), ('Q', 2), ('B', 2)]
        }

        self.assertEqual(count_encoding, expected_output)

    def test_mean_encoding(self):
        example = CategoricalData()
        mean_encoding = example.mean_encoding(self.data)


        expected_output = {
            'Category1': {'B': 4.0, 'C': 2.0, 'A': 3.0, 'G': 5.0},
            'Category2': {'X': 1.0, 'Z': 5.5, 'Y': 2.0},
            'Category3': {'X': 1.5, 'Y': 3.0, 'A': 0.0, 'Q': 7.0, 'Z': 4.0, 'B': 5.5}
        }

        self.assertEqual(mean_encoding, expected_output)

    def test_freq_encoding(self):
        example = CategoricalData()
        freq_encoding = example.freq_encoding(self.data, r=2)


        expected_output = {
            'Category1': [('B', 0, 2), ('C', 0, 2), ('G', 0, 2), ('A', 0, 2)],
            'Category2': [('Z', 0, 2), ('Y', 0, 2), ('X', 0, 2)],
            'Category3': [('Z', 0, 2), ('Y', 0, 2), ('Q', 0, 2), ('B', 0, 2), ('A', 0, 2), ('X', 0, 2)]
        }


        self.assertEqual(freq_encoding, expected_output)


if __name__ == '__main__':
    unittest.main()
