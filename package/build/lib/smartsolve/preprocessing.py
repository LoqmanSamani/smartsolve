import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from smartsolve.models import LinearRegression, LogisticRegression
from smartsolve.evaluation import Validation
from scipy.stats import f_oneway, chi2_contingency



class AnalyseData:
    def __init__(self, data):
        """
        Initializes the AnalyseData class.
        parameters:

        :param data: Path to the CSV file containing the data.
        output:
        None
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





    def heat_map(self, columns=None):
        """
        Create a heatmap to visualize the correlation between numeric columns.

        :param columns: List of columns to consider for the heatmap.
        :return: Matplotlib heatmap plot.
        """
        data = pd.read_csv(self.data)
        if columns:
            data = data[columns]  # Extract specified columns
        else:
            data = data

        numeric_data = data.select_dtypes(include='number')
        heatmap = sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
        plt.show()
        return heatmap






class SplitData:
    """
    Splits input data into training, validation, and test sets using various methods.

    :param data: Input data in the form of [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])].
    :param method: The splitting method to use ('Random', 'Stratified', 'TimeSeries', or 'KFold').
    :param train: Fraction of data for training (default is 0.8).
    :param validation: Fraction of data for validation (default is None).
    :param test: Fraction of data for testing (default is 0.2).
    :param date: List of date values for time series splitting (default is None).
    :param num_folds: Number of folds for cross-validation (default is 10).
    """
    def __init__(self, data, method='Random', train=0.7, validation=None, test=0.2, date=None, num_folds=10):

        self.data = data
        self.method = method
        self.train = train
        self.validation = validation
        self.test = test
        self.date = date
        self.num_folds = num_folds

        self.training_data = []
        self.validation_data = []
        self.test_data = []







    def __call__(self):
        """
        Call method to initiate data splitting based on the selected method.

        :return: Training, validation, and test data.
        """

        if self.method == 'Random':
            return self.random()

        elif self.method == 'Stratified':
            return self.stratified()

        elif self.method == 'TimeSeries':
            return self.time_series()

        elif self.method == 'KFold':
            return self.cross_validation()

        else:
            return ("Please provide a valid splitting method. Available options are: Random,"
                    " Stratified, TimeSeries and KFold.")







    def random(self):
        """
        Randomly split the data into training, validation, and test sets.

        :return: Training, validation, and test data.
        """

        data = self.data.copy()
        random.shuffle(data)
        num_test = int(self.test * len(data))

        for i in range(num_test):

            rand_num = random.randint(0, len(data)-1)
            item = data[rand_num]

            self.test_data.append(item)
            data.remove(item)

        if self.validation:

            random.shuffle(data)
            num_validation = int(self.validation * len(self.data))

            for i in range(num_validation):

                rand_num = random.randint(0, len(data)-1)
                item = data[rand_num]

                self.validation_data.append(item)
                data.remove(item)
        else:
            self.training_data = data

        self.training_data = data

        return self.training_data, self.validation_data, self.test_data








    def stratified(self):
        """
        Stratified split the data into training, validation, and test sets.

        :return: Training, validation, and test data.
        """
        classified_data = self.data

        train_data = []
        validation_data = []
        test_data = []

        for i in range(len(classified_data)):
            subset = classified_data[i]

            # Calculate the number of samples for each set
            num_test = int(len(subset) * self.test)
            num_validation = 0
            if self.validation:
                num_validation = int(len(subset) * self.validation)

            random.shuffle(list(subset))

            test_data.extend(subset[:num_test])
            validation_data.extend(subset[num_test:num_test + num_validation])
            train_data.extend(subset[num_test + num_validation:])

        self.training_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        return self.training_data, self.validation_data, self.test_data







    def time_series(self):
        """
        Split the data into training, validation, and test sets based on time series.

        :return: Training, validation, and test data.
        """
        data = zip(self.data, self.date)
        sorted_data = sorted(data, key=lambda x: x[-1])

        num_test = int(len(sorted_data) * self.test)

        if self.validation:

            num_validation = int(len(sorted_data) * self.validation)
            num_training = len(sorted_data) - (num_test + num_validation)

            self.training_data = sorted_data[: num_training]
            self.validation_data = sorted_data[num_training: num_training + num_validation]
            self.test_data = sorted_data[num_training + num_validation:]
        else:
            num_training = len(sorted_data) - num_test
            self.training_data = sorted_data[: num_training]
            self.test_data = sorted_data[num_training:]

        return self.training_data, self.validation_data, self.test_data







    def cross_validation(self):
        """
        Perform K-fold cross-validation and split the data into training and test sets.

        :return: Training, validation, and test data.
        """
        data = self.data

        len_test = int(self.test * len(self.data))
        len_validation = 0

        if self.validation:
            len_validation = int(self.validation * len(self.data))

        k_sets = []
        training = []
        validation = []
        test = []

        for i in range(self.num_folds):
            copied_data = data.copy()
            test_set = []

            while len(test_set) < len_test:
                random.shuffle(copied_data)
                rand_num = random.randint(0, len(copied_data) - 1)
                item = copied_data[rand_num]

                if item not in k_sets:
                    test_set.append(item)
                    k_sets.append(item)
                    copied_data.remove(item)

            if len_validation > 0:
                validation_set = []
                for j in range(len_validation):
                    rand_num = random.randint(0, len(copied_data) - 1)
                    item = copied_data[rand_num]

                    validation_set.append(item)
                    copied_data.remove(item)

                validation.append(validation_set)
            test.append(test_set)
            training.append(copied_data)

        self.training_data = training
        self.validation_data = validation
        self.test_data = test

        return self.training_data, self.validation_data, self.test_data







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




class SelectFeature:
    """
    This class offers six methods to select the most suitable features for predicting the target:

        1) Correlation Selection
        2) Mutual Information
        3) Lasso Regularization
        4) Recursive Feature Elimination
        5) Select K Best
        6) Variance Threshold

    Args:
        csv_file (str): The path to the CSV file containing the data.
        label (str): The name of the column in the data used as the label.

    """

    def __init__(self, csv_file=None, label=None):

        self.csv_file = csv_file
        self.label = label  # The name of the column in data, which is used as the label




    def correlation(self, threshold=0.5):
        """
        Select features based on their correlation with the target label.

        Args:
            threshold (float): The correlation threshold for feature selection.

        Returns:
            Tuple[List[str], pd.DataFrame]: A tuple containing a list of relevant
            features and a DataFrame with those features.
        """

        relevant_features = []
        relevant_data = pd.DataFrame()

        data = pd.read_csv(self.csv_file)
        features = list(data.columns)
        num_feature = len(list(data.columns)) - 1

        label_index = data.columns.get_loc(self.label)
        data_corr = data.corr()

        for i in range(num_feature):
            if abs(data_corr.iloc[i, label_index]) >= threshold:

                temp_rel_feature = features[i]
                relevant_features.append(temp_rel_feature)
                relevant_data[temp_rel_feature] = data[temp_rel_feature]

        return relevant_features, relevant_data





    def mutual_infos(self):
        """
        Compute mutual information between features and the target label.

        Returns:
            Dict[str, float]: A dictionary mapping feature names to their mutual information scores.
        """

        mutual = {}

        data = pd.read_csv(self.csv_file)
        labels = list(data[self.label])

        for col in data.columns:
            if col != self.label:

                feature = list(data[col].values)
                discrete_feature = [int(value) for value in feature]  # convert values in feature to discrete values
                target = [int(label) for label in labels]  # convert values in load_target to discrete values

                feature_target = [(i, j) for i, j in zip(discrete_feature, target)]

                unique_fea_tar = list(set(feature_target))

                marginal_pxs = []
                marginal_pys = []
                marginal_pxy = []

                for tup in unique_fea_tar:
                    marginal_pxy.append(feature_target.count(tup) / len(feature_target))

                for val in set(discrete_feature):
                    marginal_pxs.append(discrete_feature.count(val) / len(discrete_feature))

                for tar in set(target):
                    marginal_pys.append(target.count(tar) / len(target))

                mutual_info = 0

                for i in range(len(unique_fea_tar)):
                    px = marginal_pxs[i // len(unique_fea_tar)]
                    py = marginal_pys[i % len(unique_fea_tar)]
                    pxy = marginal_pxy[i]
                    mutual_info += pxy * np.log2(pxy / (px * py))

                mutual[col] = mutual_info

        return mutual






    def lasso(self, coefficients, lam=0.01, learning_rate=0.001, threshold=1e-4):
        """
        Apply Lasso regularization to feature coefficients for feature selection.

        Args:
            coefficients (List[float]): Coefficients of features.
            lam (float): Lasso regularization parameter.
            learning_rate (float): Learning rate for Lasso.
            threshold (float): Convergence threshold.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing the original coefficients and updated coefficients.
        """

        new_coefficients = []

        for coefficient in coefficients:
            temp_coefficient = coefficient

            if coefficient > lam * learning_rate:
                temp_coefficient -= lam * learning_rate

            elif coefficient < -lam * learning_rate:
                temp_coefficient += lam * learning_rate

            else:
                temp_coefficient = 0

            new_coefficients.append(temp_coefficient)

        # Check convergence
        diff = [abs(new - old) for new, old in zip(new_coefficients, coefficients)]

        if all(d < threshold for d in diff):

            return new_coefficients

        else:
            return self.lasso(new_coefficients, lam, learning_rate, threshold)





    def elimination(self, model='LinearRegression', evaluation='accuracy', norm='yes'):
        """
        Select features using recursive feature elimination.

        Args:
            model (str): The machine learning model to use for evaluation
            ('LinearRegression' or 'LogisticRegression').
            evaluation (str): The evaluation metric to use
            ('accuracy', 'recall', 'precision', and 'MSE').

        Returns:
            Dict[Tuple[str], float]: A dictionary containing
            model performance scores for each feature subset.
        """

        train_models = ['LinearRegression', 'LogisticRegression', 'DecisionTree']
        eva_models = ['accuracy', 'recall', 'precision', 'MSE']

        load_data = pd.read_csv(self.csv_file)
        features = []
        for col in load_data.columns:
            if col != self.label:
                features.append(col)

        model_scores = {}

        while len(features) >= 1:

            split = SplitData(load_data, train=0.8, test=0.2)
            train_set, _, test_set = split.random()

            test_data = np.array([feature[-1] for feature in test_set])
            test_label = [label[0] for label in test_data]

            predicted = None
            feature_importance = None
            evaluate = None

            if model in train_models:
                if model == 'LinearRegression':
                    train_model = LinearRegression(train_data=train_set)
                    train_model.train()
                    feature_importance = train_model.coefficients
                    predicted = train_model.predict(
                        data=test_data,
                        norm=norm
                    )

                elif model == 'LogisticRegression':
                    train_model = LogisticRegression(train_data=train_set)
                    train_model.train()
                    predicted = train_model.predict(test_data)
                    feature_importance = train_model.coefficients

            else:
                raise ValueError("Please choose a valid train model. Available models are:"
                                 " 'LinearRegression' and 'LogisticRegression'.")

            if evaluation in eva_models:
                eva_model = Validation()
                if evaluation == 'accuracy':
                    evaluate = eva_model.accuracy(labels=test_label, predicted=predicted)
                elif evaluation == 'recall':
                    eva_model = eva_model.recall(labels=test_label, predicted=predicted)
                elif evaluation == 'precision':
                    eva_model = eva_model.precision(labels=test_label, predicted=predicted)
                elif evaluation == 'MSE':
                    eva_model = eva_model.mean_squared_error(actual=test_label, predicted=predicted)

            else:
                raise ValueError("Please choose a valid evaluation model. Available models are:"
                                 " 'accuracy', 'recall', 'precision', ... .")

            model_scores[tuple(features)] = evaluate

            eliminated_feature = features[feature_importance.index(min(feature_importance))]
            load_data = load_data.drop(columns=[eliminated_feature], inplace=True)
            features.remove(eliminated_feature)

        return model_scores





    def best_features(self, k, data_type='numerical'):
        """
        Select the best 'k' features based on their importance.

        Args:
            k (int): The number of best features to select.
            data_type (str): The data type ('numerical' or 'categorical').

        Returns:
            List[str]: A list of the best 'k' features.
        """

        load_data = pd.read_csv(self.csv_file)
        features = pd.DataFrame()
        features_name = []

        for feature in load_data.columns:
            if feature != self.label:
                features_name.append(feature)
                features[feature] = load_data[feature]

        target_values = list(load_data[self.label].values.flatten())

        best_features = []

        if data_type == 'numerical':
            f_scores = []
            for name in features_name:
                f_statistic, _ = f_oneway(features[name], target_values)
                f_scores.append((name, f_statistic))

            sorted_f_scores = sorted(f_scores, key=lambda x: x[-1], reverse=True)
            best_features.extend([feature for feature, _ in sorted_f_scores[:k]])

        elif data_type == 'categorical':
            contingency_tables = {}
            for name in features_name:
                contingency_table = pd.crosstab(features[name], target_values)
                contingency_tables[name] = contingency_table

            chi2_stats = []
            for feature, table in contingency_tables.items():
                chi2, _, _, _ = chi2_contingency(table)
                chi2_stats.append((feature, chi2))

            sorted_chi2_stats = sorted(chi2_stats, key=lambda x: x[-1], reverse=True)
            best_features.extend([feature for feature, _ in sorted_chi2_stats[:k]])

        else:
            raise ValueError("Please choose a valid data_type: available data_types are: 'numerical' and 'categorical'.")

        return best_features





    def variance_threshold(self, threshold):
        """
        Remove features with variance below the specified threshold.

        Args:
            threshold (float): The threshold for feature variance.

        Returns:
            pd.DataFrame: The data with low-variance features removed.
        """

        load_data = pd.read_csv(self.csv_file)

        features = []

        for feature in load_data.columns:
            if feature != self.label:
                features.append(feature)

        for feature in features:

            variance = np.var(load_data[feature].values.flatten())

            if variance < threshold:

                load_data.drop(columns=[feature], inplace=True)

        return load_data


class CategoricalData:

    def l_encoding(self, data, labels=None, nums=None, index=1):
        """
        Label Encoding: Assigns a unique integer to each category. Suitable for
        ordinal categorical variables with a meaningful order.

        :param data: Input DataFrame containing categorical data.
        :param labels: Optional custom labels to use for encoding.
        :param nums: Optional list of integers to use for encoding.
        :param index: Optional custom index to start numbering from.
        :return: DataFrame with label-encoded categorical data.
        """
        labeled_data = {}

        if labels:
            if not nums:
                nums = [i for i in range(index, len(labels) + index)]
                labels_dict = {label: num for label, num in zip(labels, nums)}
            else:
                labels_dict = {label: num for label, num in zip(labels, nums)}

        else:
            labels = []
            for col in data.columns:
                labels.extend(set(data[col].values))
            nums = [i for i in range(len(labels))]
            labels_dict = {label: num for label, num in zip(set(labels), nums)}

        for col in data.columns:
            values = data[col].values
            mapped_labels = [labels_dict[value] for value in values]
            labeled_data[col] = mapped_labels

        return pd.DataFrame(labeled_data)

    def onehot_encoding(self, data, yes=None, no=None):
        """
        One-Hot Encoding: Creates binary columns for each category.
        Suitable for nominal categorical variables without a meaningful order.

        :param data: Input DataFrame containing categorical data.
        :param yes: Value to use for encoding category presence (default: 1).
        :param no: Value to use for encoding absence of category (default: 0).
        :return: DataFrame with one-hot encoded categorical data.
        """

        labeled_data = {}
        if yes:
            i = yes
        else:
            i = 1
        if no:
            j = no
        else:
            j = 0

        for col in data.columns:

            temp_cols = {}
            values = list(data[col].values)
            labels = set(values)

            for label in labels:

                new_col = [i if value == label else j for value in values]
                name = f"{col}_{label}"
                temp_cols[name] = new_col

            labeled_data.update(temp_cols)

        return pd.DataFrame(labeled_data)

    def bin_encoding(self, data, labels=None, nums=None, index=1):
        """
        Binary Encoding: Represents each category with binary code, combining label
        and one-hot encoding. Efficient for high-cardinality features.

        :param data: Input DataFrame containing categorical data.
        :param labels: Optional custom labels to use for encoding.
        :param nums: Optional binary encoding values for labels.
        :param index: Optional custom index to start binary encoding from.
        :return: DataFrame with binary encoded categorical data.
        """

        labeled_data = {}
        if labels and not nums:
            nums = [bin(i)[2:] for i in range(index, len(labels) + index)]  # Remove the '0b' prefix from the binary numbers
            labels_dict = dict(zip(labels, nums))

        elif labels and nums:
            bin_nums = [bin(num)[2:] for num in nums]
            labels_dict = dict(zip(labels, bin_nums))

        elif not labels and nums:
            labels = []
            for col in data.columns:
                labels.extend(set(data[col].values))
            labels = set(labels)
            bin_nums = [bin(num)[2:] for num in nums]
            labels_dict = dict(zip(labels, bin_nums))
        else:
            labels = []
            for col in data.columns:
                labels.extend(set(data[col].values))
            labels = set(labels)
            nums = [bin(i)[2:] for i in range(len(labels))]
            labels_dict = dict(zip(labels, nums))

        for col in data.columns:

            values = data[col].values
            mapped_labels = [labels_dict[value] for value in values]
            labeled_data[col] = mapped_labels

        return pd.DataFrame(labeled_data)

    def count_encoding(self, data):
        """
        Count Encoding: Replaces each category with its occurrence
        count in the dataset. Captures category prevalence information.

        :param data: Input DataFrame containing categorical data.
        :return: DataFrame with count-encoded categorical data.
        """

        labeled_data = {}

        for col in data.columns:

            values = list(data[col].values)
            labels = set(values)

            encoded_col = [(label, values.count(label)) for label in labels]

            labeled_data[col] = encoded_col

        return labeled_data

    def mean_encoding(self, data, target=None):
        """
        Mean Encoding (Target Encoding): Replaces categories with the mean
        of the target variable for that category. Captures feature-target relationship.

        :param data: Input DataFrame containing categorical data.
        :param target: Optional target variable for encoding.
                       If not provided, unique target values from the data will be used.
        :return: DataFrame with target-encoded categorical data.
        """
        if not target:
            target = []
            for col in data.columns:
                target.extend(set(data[col].values))
            target = list(set(target))
            target = [i for i in range(len(target))]

        labeled_data = {}

        for col in data.columns:

            temp_dict = {}

            values = list(data[col].values)

            temp_list = [value for value in zip(values, target)]

            labels = list(set(values))

            for label in labels:

                mean_value = np.mean([value[1] for value in temp_list if value[0] == label])

                temp_dict[label] = mean_value

            labeled_data[col] = temp_dict

        return labeled_data

    def freq_encoding(self, data, r=2):
        """
        Frequency Encoding: Replaces categories with their frequency in the dataset.
        Suitable for nominal categorical features.

        :param data: Input DataFrame containing categorical data.
        :param r: Number of decimal places to round the frequency values.
        :return: DataFrame with frequency-encoded categorical data.
        """

        labeled_data = {}

        for col in data.columns:

            values = list(data[col].values)
            labels = list(set(values))

            encoded_col = [(label, round(values.count(label)/len(values)), r) for label in labels]

            labeled_data[col] = encoded_col

        return labeled_data




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


