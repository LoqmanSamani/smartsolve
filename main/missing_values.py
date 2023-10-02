class MissingValues:

    """
    In this class, there are some methods to handle missing values
    in both numeric and qualitative datasets the input is a
    DataFrame and the output is the modified version of DataFrame

    """

    def __init__(self, data, desire_value=None, option=None, desire_string=None):

        """
        the variants of option = (mean, null, value, del, string)
        """

        self.data = data
        self.desire_value = desire_value
        self.option = option
        self.desire_string = desire_string

    def load_data(self):
        import pandas as pd

        data = pd.read_csv(self.data)

        return data


    def  handle_missing_neumerical_data(self):

        for column in self.data.columns:
            incomplete_cols = []
            if self.data[column].isnull().sum() > 0:

                incomplete_cols.append(column)

        if self.option == "mean":
            rep_mean = self.data.copy()
            replace_with_mean = [(rep_mean[col].fillna(rep_mean[col].mean(), inplace=True)) for col in incomplete_cols]
            return replace_with_mean

        elif self.option == "null":
            rep_null = self.data.copy()
            replace_with_null = [(rep_null[col].fillna(0, inplace=True)) for col in incomplete_cols]
            return replace_with_null


        elif self.option == "value":
            rep_value = self.data.copy()
            replace_with_desire_value = [(rep_value[col].fillna(self.desire_value, inplace=True)) for col in incomplete_cols]
            return replace_with_desire_value


        elif self.option == "del":
            rep_del = self.data.copy()
            delete_col = [rep_del.drop(col, axis=1, inplace=True) for col in incomplete_cols]
            return delete_col

        else:
            return self.data


    def  handle_missing_qualitative_data(self):

        for column in self.data.columns:
            incomplete_cols = []
            if self.data[column].isnull().sum() > 0:
                incomplete_cols.append(column)

        if self.option == "string":
            rep_str = self.data.copy()
            replace_with_string= [(rep_str[col].fillna(self.desire_string, inplace=True)) for col in incomplete_cols]
            return replace_with_string


        elif self.option == "del":
            rep_del = self.data.copy()
            delete_col = [rep_del.drop(col, axis=1, inplace=True) for col in incomplete_cols]
            return delete_col

        else:
            return self.data
