class FeatureSelection:

    """
    This class offers six methods to select the most
    suitable features for predicting the target

       1) Correlation Selection
       2) Mutual Information
       3) Lasso Regularization
       4) Recursive Feature Elimination
       5) Select K Best
       6) Variance Threshold
    """

    def __init__(self, data, target):

        self.data = data
        self.target = target




    def __call__(self, method):

        if method == "CorrelationSelection":
            return self.correlation_selection()

        elif method == "MutualInformation":
            return self.mutual_information()

        elif method == "Lasso":
            return self.lasso_regularization()

        elif method == "FeatureElimination":
            return self.recursive_feature_elimination()

        elif method == "SelectkBest":
            return self.select_k_best_features()

        elif method == "VarianceThreshold":
            return self.variance_threshold()

        else:
            return ("Please provide a valid selection method. Available options are: CorrelationSelection,"
                    " MutualInformation, Lasso, FeatureElimination, SelectkBest and VarianceThreshold.")




    def correlation_selection(self, threshold=0.5):

        import pandas as pd

        relevant_features = []

        load_data = pd.read_csv(self.data)
        features = list(load_data.columns)
        load_data['target'] = self.target # add target values to the end of feature's dataframe
        data_corr = load_data.corr()

        for i in range(len(features)):
            if abs(data_corr.iloc[i, -1]) >= threshold:

                temp_rel_feature = features[i]
                relevant_features.append(temp_rel_feature)

        return relevant_features





    def mutual_information(self):

        import pandas as pd
        import numpy as np

        mutual_dict = {}

        load_features = pd.read_csv(self.data)
        load_target = list(self.target.values)

        for col in load_features.columns:

            feature = list(load_features[col].values)
            discrete_feature = [int(value) for value in feature]  # convert values in feature to discrete values
            target = [int(value) for value in load_target]  # convert values in load_target to discrete values

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

            mutual_dict[col] = mutual_info

        return mutual_dict





    def lasso_regularization(self, coefficients, lam=0.01, learning_rate=0.001, threshold=0.0001):

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
            return self.lasso_regularization(lam, learning_rate, threshold, new_coefficients)





        def recursive_feature_elimination(self, data_split, training_model, model_evaluating, feature_evaluating):

            """
            Perform Recursive Feature Elimination (RFE) to iteratively remove
            least important features and evaluate the model's performance

            Parameters:

               1) data_split:  A function that takes the dataset and splits it into training and test sets.

               2) training_model:  function that takes training data and labels,
                   and returns a trained machine learning model.

               3) model_evaluating: A function that takes the trained model, test data, and test labels,
                  and returns a performance metric (e.g., MSE)

               4) feature_evaluating: A function that takes the dataset and evaluates the importance
                  of features, returning a score for each feature (e.g., Gini impurity)

            Return:
               a dictionary containing model performance scores for each iteration of feature elimination.
            """

            import pandas as pd

            load_data = pd.read_csv(self.data)
            features = list(load_data.columns)

            load_data['target'] = self.target

            model_scores = {}

            while len(features) >= 1:

                training_data, training_labels, test_data, test_labels = data_split(load_data)
                model = training_model(training_data, training_labels)
                evaluate_score = evaluating_model(test_data, test_labels)

                model_scores[tuple(features)] = evaluate_score

                feature_evaluating_scores = feature_evaluating(load_data)
                eliminated_feature = min(feature_evaluating_scores)
                load_data = load_data.drop(columns=[eliminated_feature], inplace=True)
                features.remove(eliminated_feature)



            return model_scores






        def select_k_best_features(self,k, data_type = 'numerical'):

            import pandas as pd
            import numpy as np
            from scipy.stats import f_oneway, chi2_contingency


            load_data = pd.read_csv(self.data)
            load_target = pd.read_csv(self.target)

            features = list(load_data.columns)
            target_values = list(load_target.values.flatten())

            best_features = []

            if data_type == 'numerical':
                f_scores = []
                for feature in features:
                    f_statistic, _ = f_oneway(load_data[feature], target_values)
                    f_scores.append((feature, f_statistic))

                sorted_f_scores = sorted(f_scores, key=lambda x: x[-1], reverse=True)
                best_features.extend([feature for feature, _ in sorted_f_scores[:k]])

            elif data_type == 'categorical':
                contingency_tables = {}
                for feature in features:
                    contingency_table = pd.crosstab(load_data[feature], target_values)
                    contingency_tables[feature] = contingency_table

                chi2_stats = []
                for feature, table in contingency_tables.items():
                    chi2, _, _, _ = chi2_contingency(table)
                    chi2_stats.append((feature, chi2))

                sorted_chi2_stats = sorted(chi2_stats, key=lambda x: x[-1], reverse=True)
                best_features.extend([feature for feature, _ in sorted_chi2_stats[:k]])

            else:
                print('There are just two possible inputs for data_type: numerical, categorical')

            return best_features





        def variance_threshold(self, threshold):

            import numpy as np
            import pandas as pd

            load_data = pd.read_csv(self.data)

            features = list(load_data.columns)

            for feature in features:

                variance = np.var(load_data[feature].values.flatten())

                if variance < threshold:

                    load_data.drop(columns=[feature], inplace=True)


            return load_data





















