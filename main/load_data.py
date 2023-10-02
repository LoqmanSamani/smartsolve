class ReadData:
    """
    This class provides a comprehensive understanding of the data using Pandas and Seaborn
    """


    def __init__(self, data):
        self.data = data

    def load_data(self):

        import pandas as pd
        import seaborn as sns

        data = pd.read_csv(self.data)
        data_infos = data.info()
        data_columns = data.columns
        numeric_data = data.select_dtypes(include='number')
        heatmap = sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)

        return data, data_columns, data_infos, heatmap