import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random





class AnalyseData:
    def __init__(self, data):
        """
        Initializes the AnalyseData class.

        :param data: Path to the CSV file containing the data.
        """
        self.data = data




    def load_data(self):
        """
        Load the data from the provided CSV file.

        :return: Loaded data as a Pandas DataFrame.
        """
        data = pd.read_csv(self.data)
        return data





    def infos(self):
        """
        Display information about the data.

        :return: Data information.
        """
        data = pd.read_csv(self.data)
        data_infos = data.info()
        return data_infos





    def stats(self):
        """
        Perform statistical analysis on the columns in the data.

        Analyzes each column and provides statistical information such as mean, standard deviation, etc.

        :return: None
        """
        data = pd.read_csv(self.data)

        for col in data.columns:
            column = data[col]

            if column.dtype == 'int64' or column.dtype == 'float64':
                print(f"The {col} is a numerical column.")
                print(column.describe())
            elif column.dtype == 'object' or column.dtype.name == 'category':
                print(f"The {col} is a categorical column.")
                print(column.value_counts())
            else:
                print(f"This column {col} is not numerical or categorical!")





    def heat_map(self, columns):
        """
        Create a heatmap to visualize the correlation between numeric columns.

        :param columns: List of columns to consider for the heatmap.
        :return: Matplotlib heatmap plot.
        """
        data = pd.read_csv(self.data)
        data = data[columns]  # Extract specified columns
        numeric_data = data.select_dtypes(include='number')
        heatmap = sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
        plt.show()
        return heatmap







