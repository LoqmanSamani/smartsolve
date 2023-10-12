import pandas as pd



class MissingValue:
    """
       A class for handling missing values in a pandas DataFrame.

       :param data: str or pandas.DataFrame
           The data source, which can be a file path to a CSV or a pandas DataFrame.
       :param replace: str, optional
           The method to replace missing values. Default is 'Mean'.
       :param rep_value: str or numeric, optional
           The replacement value to use when 'replace' is 'Value'. Default is None.
       :param rep_str: str, optional
           The replacement string to use when 'replace' is 'Str'. Default is 'not defined'.

       Methods:
       -------------------
       load_data():
           Load the data source as a pandas DataFrame.

       numerical():
           Replace missing values in numeric columns using specified method.

       qualitative():
           Replace missing values in qualitative (categorical) columns using specified method.
       """

    def __init__(self, data, replace='Mean', rep_value=None, rep_str='not defined'):
        """
        Initialize a MissingValue instance.

        :param data: str or pandas.DataFrame. The data source.
        :param replace: str, optional. The method to replace missing values.
         Default is 'Mean'.
        :param rep_value: str or numeric, optional. The replacement value to use when 'replace' is 'Value'.
         Default is None.
        :param rep_str: str, optional. The replacement string to use when 'replace' is 'Str'.
         Default is 'not defined'.
        """

        self.data = data
        self.rep_value = rep_value
        self.replace = replace
        self.rep_str = rep_str





    def load_data(self):
        """
        Load the data source as a pandas DataFrame.

        :return: pandas.DataFrame
            The loaded data as a DataFrame.
        """

        data = pd.read_csv(self.data)

        return data





    def numerical(self):
        """
        Replace missing values in numeric columns using the specified method.

        :return: pandas.DataFrame
            The DataFrame with missing values replaced.
        """

        data = self.data.copy()  # Create a copy to avoid modifying the original DataFrame

        incomplete_cols = []

        for column in data.columns:
            if data[column].isnull().sum() > 0:
                incomplete_cols.append(column)


        if self.replace == "Mean":
            for col in incomplete_cols:
                data[col].fillna(data[col].mean(), inplace=True)


        elif self.replace == "Null":
            for col in incomplete_cols:
                data[col].fillna(0, inplace=True)


        elif self.replace == "Value":
            for col in incomplete_cols:
                data[col].fillna(self.rep_value, inplace=True)


        elif self.replace == "Del":
            data.dropna(inplace=True)


        else:
            raise ValueError("Please select a valid 'replace' option. "
                             "Available choices include: 'Mean,' 'Null,' 'Value,' and 'Del.'")


        return data  # Return the modified DataFrame






    def qualitative(self):
        """
        Replace missing values in qualitative (categorical) columns using the specified method.

        :return: pandas.DataFrame
            The DataFrame with missing values replaced.
        """

        data = self.data.copy()
        incomplete_cols = []

        for column in data.columns:

            if data[column].isnull().sum() > 0:
                incomplete_cols.append(column)


        if self.replace == "Str":
            for col in incomplete_cols:
                data[col].fillna(self.rep_str, inplace=True)


        elif self.replace == "Del":
            data.dropna(inplace=True)


        else:
            raise ValueError("Please select a valid 'replace' option."
                             " Available choices include: 'Del' and 'Str'.")
        return data