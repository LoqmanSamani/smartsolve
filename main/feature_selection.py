import pandas as pd
from machine_learning.linear_algebra import intro_numpy as np
from smartsolve.preprocessing import SplitData
from smartsolve.models import LinearRegression, LogisticRegression, DecisionTree
from smartsolve.evaluation import Validation
from scipy.stats import f_oneway, chi2_contingency




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

    def lasso(self, coefficients, lam=0.01, learning_rate=0.001, threshold=0.0001):
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

            return coefficients, new_coefficients

        else:
            return self.lasso(new_coefficients, lam, learning_rate, threshold)

    def elimination(self, model='LinearRegression', evaluation='accuracy'):
        """
        Select features using recursive feature elimination.

        Args:
            model (str): The machine learning model to use for evaluation
            ('LinearRegression', 'LogisticRegression', or 'DecisionTree').
            evaluation (str): The evaluation metric to use
            ('accuracy', 'recall', 'precision', 'MSE', or other).

        Returns:
            Dict[Tuple[str], float]: A dictionary containing
            model performance scores for each feature subset.
        """

        train_models = ['LinearRegression', 'LogisticRegression', 'DecisionTree']
        eva_models = ['accuracy', 'recall', 'precision', 'MSE', '...']  # TODO: you should add mode models
        load_data = pd.read_csv(self.csv_file)
        features = []
        for col in load_data.columns:
            if col != self.label:
                features.append(col)

        model_scores = {}

        while len(features) >= 1:

            split = SplitData(load_data, train=0.8, test=0.2)
            train_set, test_set = split.random()

            test_data = [feature[-1] for feature in test_data]
            test_label = [label[0] for label in test_data]

            predicted = None
            feature_importance = None
            evaluate = None

            if model in train_models:
                if model == 'LinearRegression':
                    train_model = LinearRegression(train_set)
                    train_model.train()
                    feature_importance = train_model.coefficients
                    predicted = train_model.predict(test_set)

                elif model == 'LogisticRegression':
                    train_model = LogisticRegression(train_set)
                    train_model.train()
                    predicted = train_model.predict(test_set)
                    feature_importance = train_model.coefficients

                elif model == 'DecisionTree':
                    train_model = DecisionTree(train_set)
                    train_model.train()
                    predicted = train_model.predict(test_set)
                    feature_importance = train_model.gini_impurity(load_data)

            else:
                raise ValueError("Please choose a valid train model. Available models are:"
                                 " 'LinearRegression', 'LogisticRegression' and 'DecisionTree'.")

            # TODO: you should change the linear regression and other models based on the code here!!!

            if evaluation in eva_models:
                eva_model = Validation()
                if evaluation == 'accuracy':
                    evaluate = eva_model.accuracy(actual=test_label, predicted=predicted)
                elif evaluation == 'recall':
                    eva_model = eva_model.recall(actual=test_label, predicted=predicted)
                elif evaluation == 'precision':
                    eva_model = eva_model.precision(actual=test_label, predicted=predicted)
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


