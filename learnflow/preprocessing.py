import random
import pandas as pd






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