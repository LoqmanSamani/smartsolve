import pandas as pd
from smartsolve.models import LogisticRegression
import numpy as np
from sklearn.metrics import roc_curve, auc


class Validation:

    def accuracy(self, labels, predicted):
        """
        It calculates the accuracy of predicted labels.

        :param labels: List of actual labels.
        :param predicted: List of predicted labels.
        :return: Accuracy as a percentage.
        """

        if len(labels) == len(predicted):

            accurate = sum([1 if label == predict else 0 for label, predict in zip(labels, predicted)])
            accuracy = (accurate / len(predicted)) * 100

        else:
            raise ValueError("The number of actual and predicted lists do not match!")

        return accuracy


    def precision(self, labels, predicted):
        """
        it calculates precision for each label.

        :param labels: List of actual labels.
        :param predicted: List of predicted labels.
        :return: Dictionary of precision scores for each label.
        """

        if len(labels) == len(predicted):
            precision = {}  # Dictionary to store precisions for each label
            labels = list(set(predicted))  # all possible labels

            for label in labels:

                true_positive = sum([1 if label == labels[j] and label == predicted[j] else 0 for j in range(len(labels))])
                total_positive = predicted.count(label)
                accuracy = (true_positive / total_positive) * 100
                precision[label] = accuracy

        else:
            raise ValueError("The number of actual and predicted lists do not match!")

        return precision







    def recall(self, labels, predicted):
        """
        Calculate recall for each label.

        :param labels: List of actual labels.
        :param predicted: List of predicted labels.
        :return: Dictionary of recall scores for each label.
        """

        if len(labels) == len(predicted):
            recall = {}  # Dictionary to store recalls for each label
            labels = list(set(predicted))  # all possible labels

            for label in labels:

                true_positive = sum([1 if label == labels[j] and label == predicted[j] else 0 for j in range(len(labels))])
                total_positive = labels.count(label)
                accuracy = (true_positive / total_positive) * 100
                recall[label] = accuracy

        else:
            raise ValueError("The number of actual and predicted lists do not match!")
        return recall







    def f_score(self, labels, predicted, beta=None):
        """
        Calculate F1-score for each label.

        :param labels: List of actual labels.
        :param predicted: List of predicted labels.
        :param beta: Weighting factor for F-beta score.
        :return: Dictionary of F1-scores for each label.
        """

        if len(labels) == len(predicted):

            f1_scores = {}
            f_beta_scores = {}
            labels = list(set(predicted))

            for label in labels:

                true_positive = sum(1 for a, p in zip(labels, predicted) if a == label and p == label)
                r_total_positive = labels.count(label)
                p_total_positive = predicted.count(label)

                # Calculate precision and recall
                precision = (true_positive / p_total_positive) if p_total_positive > 0 else 0
                recall = (true_positive / r_total_positive) if r_total_positive > 0 else 0

                # Calculate F1-score
                f1_scores[label] = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

                if beta:
                    f_beta_scores[label] = ((1 + beta ** 2) * (precision * recall)) / ((beta ** 2 * precision) + recall) if (beta ** 2 * precision) + recall > 0 else 0

            if f_beta_scores:
                return f1_scores, f_beta_scores
            else:
                return f1_scores
        else:
            raise ValueError("The number of actual and predicted lists do not match!")






    def confusion_matrix(self, labels, predicted, positive_label=None):
        """
        Generate a confusion matrix.

        :param labels: List of actual labels.
        :param predicted: List of predicted labels.
        :param positive_label: Specify the positive label.
        :return: Pandas DataFrame representing the confusion matrix.
        """

        if positive_label:
            p_label = positive_label
        else:
            p_label = labels[0]

        num_p_label = predicted.count(p_label)
        num_n_label = len(predicted) - num_p_label

        if len(labels) == len(predicted):

            true_positive = sum(1 for a, p in zip(labels, predicted) if a == p_label and p == p_label)
            false_positive = num_p_label - true_positive
            true_negative = sum(1 for a, p in zip(labels, predicted) if a != p_label and p != p_label)
            false_negative = num_n_label - true_negative

            confusion_matrix = pd.DataFrame({
                'Positive Label': [p_label],
                'True Positive': [true_positive],
                'True Negative': [true_negative],
                'False Positive': [false_positive],
                'False Negative': [false_negative]
            })

            return confusion_matrix

        else:
            raise ValueError("The number of actual and predicted lists do not match!")

    def roc_curve(self, training_data, validation_data, coefficients=None, bias=None, learning_rate=1e-4,
                  positive_label=None, negative_label=None, max_iter=100, seed=42, norm='yes', num_iter=10,
                  lower_bound=0.4, upper_bound=0.6):
        """
        Generate Receiver Operating Characteristic (ROC) curves and calculate Area Under the Curve (AUC) values.

        :param training_data: Training data for logistic regression.
        :param validation_data: Validation data for generating ROC curves.
        :param coefficients: Initial coefficients for logistic regression (if None, they will be initialized).
        :param bias: Initial bias for logistic regression (if None, it will be initialized).
        :param learning_rate: Learning rate for logistic regression.
        :param positive_label: Specify the positive label (default is the first unique label in data).
        :param negative_label: Specify the negative label (default is the second unique label in data).
        :param max_iter: Maximum number of iterations for logistic regression.
        :param seed:
        :param norm: Should the data before training process be normalized.
        :param num_iter: Number of threshold values for ROC curve.
        :param lower_bound: Lower bound for ROC curve threshold.
        :param upper_bound: Upper bound for ROC curve threshold.
        :return: List of ROC curve data as tuples (threshold, AUC).
        """

        roc_curves = []
        data = [point[1] for point in validation_data]
        actual = [point[0] for point in validation_data]

        labels = list(set(actual))

        if positive_label:
            p_label = positive_label
        else:
            p_label = labels[0]

        if negative_label:
            n_label = negative_label
        else:
            n_label = labels[1]

        thresholds = np.linspace(lower_bound, upper_bound, num_iter)

        for threshold in thresholds:

            model = LogisticRegression(
                train_data=training_data,
                coefficients=coefficients,
                bias=bias,
                learning_rate=learning_rate,
                max_iter=max_iter,
                threshold=threshold,
                seed=seed,
                norm=norm

            )

            model.train()

            predicted = model.predict(data=np.array(data))

            fpr, tpr, thresholds = roc_curve(actual, predicted)
            roc_auc = auc(fpr, tpr)
            roc_curves.append((threshold, roc_auc))

        return roc_curves






    def mean_squared_error(self, actual, predicted):
        """
        Calculate the Mean Squared Error (MSE) between actual and predicted values.

        :param actual: List of actual values.
        :param predicted: List of predicted values.
        :return: Mean Squared Error (MSE).
        """

        actual = np.array(actual)
        predicted = np.array(predicted)

        mse = np.mean((actual - predicted) ** 2)

        return mse






    def mean_absolute_error(self, actual, predicted):
        """
        Calculate the Mean Absolute Error (MAE) between actual and predicted values.

        :param actual: List of actual values.
        :param predicted: List of predicted values.
        :return: Mean Absolute Error (MAE).
        """

        actual = np.array(actual)
        predicted = np.array(predicted)

        mae = np.mean(abs(actual - predicted))

        return mae






    def r_squared(self, actual, predicted):
        """
        Calculate the R-squared (Coefficient of Determination) between actual and predicted values.
         parameters:

        :param actual: List of actual values.
        :param predicted: List of predicted values.

        output:
        :return: R-squared value.
        """

        actual = np.array(actual)
        predicted = np.array(predicted)

        # sum of squares of residuals
        rss = np.sum(np.power(actual - predicted, 2))

        # total sum of squares
        tss = np.sum(np.power(predicted - np.mean(predicted), 2))

        r_squared = 1 - (rss / tss)

        return r_squared


