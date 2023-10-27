import unittest
import pandas as pd
import tempfile
import os
import sys
from io import StringIO
from smartsolve.preprocessing import AnalyseData


class TestAnalyseData(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file for testing
        self.csv_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.csv_file.close()
        data = {
            'NumericColumn': [1, 2, 3, 4, 5],
            'CategoricalColumn': ['A', 'B', 'A', 'C', 'C'],
        }
        self.test_data = pd.DataFrame(data)
        self.test_data.to_csv(self.csv_file.name, index=False)

        # Redirect stdout for capturing print statements
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        # Clean up and remove the temporary CSV file
        os.remove(self.csv_file.name)

        # Restore original stdout
        sys.stdout = self.original_stdout

    def test_load_data(self):
        model = AnalyseData(self.csv_file.name)
        loaded_data = model.load_data()
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertTrue(loaded_data.equals(self.test_data))

    def test_infos(self):
        model = AnalyseData(self.csv_file.name)
        data_info = model.infos()
        self.assertIsNone(data_info)

    def test_stats(self):
        model = AnalyseData(self.csv_file.name)
        model.stats()
        captured_text = self.captured_output.getvalue()
        self.assertIn('NumericColumn is a numerical column.', captured_text)
        self.assertIn('CategoricalColumn is a categorical column.', captured_text)

    def test_heat_map(self):
        model = AnalyseData(self.csv_file.name)
        columns = ['NumericColumn']
        heatmap = model.heat_map(columns)
        self.assertIsNotNone(heatmap)


if __name__ == '__main__':
    unittest.main()




