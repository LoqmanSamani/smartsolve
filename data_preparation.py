
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



class MissingValues:

    """
    In this class, there are some methods to handle missing values
    in both numeric and qualitative datasets the input is a
    DataFrame and the output is the modified version of DataFrame

    """

    def __init__(self, data, desire_value=None, option=None, desire_string=None):

        """
        the variants of option = [mean, null, value, del, string]
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



class FeatureScaling:

    """
    In this class, there are 9 different normalization methods for numeric data

    1) Min-max scaling
    2) Standardization (Z-score normalization)
    3) Robust scaling
    4) Max-Absolute scaling
    5) Power transformation
    6) Unit Vector Scaling (L2 Normalization)
    7) Log transformation
    8) Box-Cox transformation
    9) Yeo-Johnson transformation
    """

    def __init__(self, data, lam=None):
        self.data = data
        self.lam = lam



    def min_max_normalization(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            min_val = np.min(temporary_col)
            max_val = np.max(temporary_col)

            normalized_col = [(val - min_val)/(max_val - min_val) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def z_score_normalization(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:
            temporary_col = self.data[col].values
            mean_val = np.mean(temporary_col)
            std_val = np.std(temporary_col)

            normalized_col = [(val - mean_val) / std_val  for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def robust_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            median_val = np.median(temporary_col)
            q1 = np.percentile(temporary_col,25)
            q3 = np.percentile(temporary_col, 75)
            iqr_val = q3 - q1

            normalized_col = [(val - median_val) / iqr_val  for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def max_absolute_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            max_abs = abs(np.max(temporary_col))

            normalized_col =[val / max_abs for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def power_transform(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values
            normalized_col = [np.power(val, self.lam) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def unit_vector_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        norms = np.linalg.norm(self.data.values, axis=1, keepdims=True)

        for col in self.data.columns:
            temporary_col = self.data[col].values
            cal_norms = norms[0:len(self.data[col])]
            normalized_col = [val / norm for val, norm in zip(temporary_col, cal_norms)]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def log_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            normalized_col = [np.log(val) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def box_cox_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            normalized_col = [np.power(val, self.lam - 1) if self.lam != 0
                              else np.log(val) for val in temporary_col]

            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def yeo_johnson_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values
            normalized_col = []

            for val in temporary_col:

                if val > 0 and self.lam != 0:
                    normalized_col.append(np.power(val+1, self.lam-1) / self.lam)
                elif val < 0 and self.lam != 0:
                    normalized_col.append(-(np.power(-val+1, -self.lam+1) / self.lam))

                elif val < 0 and self.lam == 0:
                    normalized_col.append(np.log(val+1))

                elif val < 0 and self.lam == 0:
                    normalized_col.append(-(np.log(-val+1)))

                else:
                    normalized_col.append(val)

            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)
























