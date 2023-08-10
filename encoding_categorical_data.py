class EncodingCategoricalData:

    """

    This class provides various categorical encoding methods
    for transforming categorical features.

    Available methods:

    1) Label Encoding:
       Assigns a unique integer to each category. Suitable for
       ordinal categorical variables with a meaningful order.

    2) One-Hot Encoding:
       Creates binary columns for each category. Suitable for
       nominal categorical variables without a meaningful order.

    3) Binary Encoding:
       Represents each category with binary code, combining label
       and one-hot encoding. Efficient for high-cardinality features.

    4) Count Encoding:
       Replaces each category with its occurrence count in the dataset.
       Captures category prevalence information.

    5) Target Encoding (Mean Encoding):
       Replaces categories with the mean of the target variable for
       that category. Captures feature-target relationship.

    6) Frequency Encoding:
       Replaces categories with their frequency in the dataset.
        Suitable for nominal categorical features.

    """

    def __init__(self, data, labels=None, target=None):

        self.data = data
        self.labels = labels
        self.target = target


    def __call__(self, method):

        if method == "LabelEncoding":
            return self.label_encoding()

        elif method == "OneHotEncoding":
            return self.one_hot_encoding()


        elif method == "BinaryEncoding":
            return self.binary_encoding()


        elif method == "CountEncoding":
            return self.count_encoding()


        elif method == "MeanEncoding":
            return self.target_encoding()


        elif method == "FrequencyEncoding":
            return self.frequency_encoding()


        else:
            return ("Please provide a valid encoding method. Available options are: 'LabelEncoding',"
                    " 'OneHotEncoding', 'BinaryEncoding', 'CountEncoding', 'MeanEncoding', and 'FrequencyEncoding'.")





    def label_encoding(self):

        import pandas as pd

        labeled_data = {}
        nums = [i for i in range(len(self.labels))]
        labels_dict = dict(zip(self.labels, nums))

        for col in self.data.columns:
            temporary_col = self.data[col].values
            mapped_labels = [labels_dict[value] for value in temporary_col]
            labeled_data[col] = mapped_labels

        return pd.DataFrame(labeled_data)



    def one_hot_encoding(self):

        import pandas as pd

        labeled_data = {}

        for col in self.data.columns:
            temporary_cols = {}
            temporary_col = list(self.data[col].values)

            labels = list(set(temporary_col))

            for label in labels:

                new_col = [1 if val == label else 0 for val in temporary_col]
                new_col_name = f"{col}_{label}"
                temporary_cols[new_col_name] = new_col

            labeled_data.update(temporary_cols)

        return pd.DataFrame(labeled_data)




    def binary_encoding(self):

        import pandas as pd

        labeled_data = {}

        nums = [bin(i)[2:] for i in range(len(self.labels))]  # Remove the '0b' prefix

        labels_dict = dict(zip(self.labels, nums))

        for col in self.data.columns:
            temporary_col = self.data[col].values
            mapped_labels = [labels_dict[value] for value in temporary_col]
            labeled_data[col] = mapped_labels

        return pd.DataFrame(labeled_data)



    def count_encoding(self):

        labeled_data = {}

        for col in self.data.columns:

            temporary_col = list(self.data[col].values)
            labels = list(set(temporary_col))

            encoded_col = [temporary_col.count(label) for label in labels]

            labeled_data[col] = encoded_col

        return labeled_data




    def target_encoding(self):

        import numpy as np

        encoded_data = {}

        for col in self.data.columns:

            temporary_dict = {}

            temporary_col = list(self.data[col].values)
            temporary_lst = [val for val in zip(temporary_col, self.target)]
            labels = list(set(temporary_col))

            for label in labels:

                mean_value = np.mean([val[1] for val in temporary_lst if val[0] == label])

                temporary_dict[label] = mean_value

            encoded_data[col] = temporary_dict

        return encoded_data




    def frequency_encoding(self):

        labeled_data = {}

        for col in self.data.columns:

            temporary_col = list(self.data[col].values)
            labels = list(set(temporary_col))

            encoded_col = [temporary_col.count(label)/len(temporary_col) for label in labels]

            labeled_data[col] = encoded_col

        return labeled_data
















