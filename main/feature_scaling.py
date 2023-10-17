import pandas as pd
import numpy as np


class FeatureScaling:

    """
    In this class, there are 9 different normalization methods for numeric data:

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
    def min_max(self, data, columns=None):
        """
        Min-Max scaling (Normalization).

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to normalize.
                        If None, all numeric columns are selected.
        :return: DataFrame with Min-Max normalized data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:
                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):
                    min_val = np.min(values)
                    max_val = np.max(values)
                    norm_col = [(value - min_val) / (max_val - min_val) for value in values]
                    norm_data[col] = norm_col
                else:
                    norm_data[col] = values
            else:
                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def z_score(self, data, columns=None):
        """
        Standardization (Z-score normalization).

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to standardize.
                        If None, all numeric columns are selected.
        :return: DataFrame with Z-score standardized data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    norm_col = [(value - mean_val) / std_val for value in values]
                    norm_data[col] = norm_col

            else:

                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def robust(self, data, columns=None):
        """
        Robust scaling.

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to scale robustly.
                        If None, all numeric columns are selected.
        :return: DataFrame with robustly scaled data.
        """

        norm_data = {}
        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:

            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    med_val = np.median(values)
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1

                    norm_col = [(val - med_val) / iqr for val in values]
                    norm_data[col] = norm_col

            else:

                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def abs_max(self, data, columns=None):
        """
        Max-Absolute scaling.

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to scale using Max-Absolute scaling.
                        If None, all numeric columns are selected.
        :return: DataFrame with Max-Absolute scaled data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    abs_max = abs(np.max(values))

                    norm_col = [value / abs_max for value in values]
                    norm_data[col] = norm_col
            else:
                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def pow_transform(self, data, lam=2, columns=None):
        """
        Power transformation.

        :param data: Input DataFrame containing numeric data.
        :param lam: Lambda value for the power transformation.
        :param columns: List of columns to apply the power transformation.
                        If None, all numeric columns are selected.
        :return: DataFrame with power-transformed data.
        """

        norm_data = {}
        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    norm_col = [np.power(value, lam) for value in values]
                    norm_data[col] = norm_col

            else:

                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def unit_vector(self, data, columns=None):
        """
        Unit Vector Scaling (L2 Normalization).

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to apply L2 normalization.
                        If None, all numeric columns are selected.
        :return: DataFrame with unit vector scaled data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in columns:
            values = data[col].values

            if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):
                norm_data[col] = values

        norm_df = pd.DataFrame(norm_data)

        norms = np.linalg.norm(norm_df.values, axis=1, keepdims=True)
        normalized_data = norm_df.div(norms, axis=0)

        for column in data.columns:
            if column not in columns:
                normalized_data[column] = data[column]

        return normalized_data


    def log_transform(self, data, columns=None):
        """
        Log transformation.

        :param data: Input DataFrame containing numeric data.
        :param columns: List of columns to apply the log transformation.
                        If None, all numeric columns are selected.
        :return: DataFrame with log-transformed data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    # Add a small constant to avoid logarithm of 0
                    norm_col = [np.log(val + 1e-10) for val in values]
                    norm_data[col] = norm_col

            else:

                norm_data[col] = data[col]

        return pd.DataFrame(norm_data)

    def box_cox(self, data, lam=2, columns=None):
        """
        Box-Cox transformation.

        :param data: Input DataFrame containing numeric data.
        :param lam: Lambda value for the Box-Cox transformation.
        :param columns: List of columns to apply the Box-Cox transformation.
                        If None, all numeric columns are selected.
        :return: DataFrame with Box-Cox transformed data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    norm_col = [np.power(value, lam - 1) if lam != 0 else np.log(value) for value in values]
                    norm_data[col] = norm_col

                else:

                    norm_data[col] = values

        return pd.DataFrame(norm_data)

    def yeo_johnson(self, data, lam=2, columns=None):
        """
        Yeo-Johnson transformation.

        :param data: Input DataFrame containing numeric data.
        :param lam: Lambda value for the Yeo-Johnson transformation.
        :param columns: List of columns to apply the Yeo-Johnson transformation.
                        If None, all numeric columns are selected.
        :return: DataFrame with Yeo-Johnson transformed data.
        """

        norm_data = {}

        if not columns:
            columns = data.select_dtypes(include=['int', 'float']).columns

        for col in data.columns:
            if col in columns:

                values = data[col].values

                if all(np.issubdtype(value, int) or np.issubdtype(value, float) for value in values):

                    norm_col = []
                    for value in values:
                        if value > 0 and lam != 0:
                            norm_col.append(np.power(value+1, lam-1) / lam)
                        elif value < 0 and lam != 0:
                            norm_col.append(-(np.power(-value+1, -lam+1) / lam))

                        elif value < 0 and lam == 0:
                            norm_col.append(np.log(value+1))

                        elif value < 0 and lam == 0:
                            norm_col.append(-(np.log(-value+1)))

                        else:
                            norm_col.append(value)

                    norm_data[col] = norm_col

                else:

                    norm_data[col] = values

        return pd.DataFrame(norm_data)

