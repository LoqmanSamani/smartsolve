import pandas as pd
import numpy as np


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















