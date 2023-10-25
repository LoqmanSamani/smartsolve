from learnflow.models import DecisionTree
from collections import Counter
import numpy as np


class RandomForest:

    def __init__(self, train_data, num_trees=30, max_depth=20, min_points=2, num_features=None, curr_depth=0):
        """
        Initializes a Random Forest classifier/regressor.

        :param train_data: Input training data in the form of [(y1, [x11, x12, ..., x1n]), (y2, [x21, x22, x2n]), ...].
        :param num_trees: The number of decision trees in the random forest (default is 30).
        :param max_depth: The maximum depth of each decision tree (default is 20).
        :param min_points: The minimum number of data points in a leaf node (default is 2).
        :param num_features: The number of features to consider at each split (default is None).
        :param curr_depth: The current depth of the random forest (default is 0).
        """

        self.train_data = train_data
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_points = min_points
        self.num_features = num_features
        self.curr_depth = curr_depth

        self.trees = []

    def fit(self):
        """
        Fits the Random Forest model by training multiple decision trees.
        """

        for _ in range(self.num_trees):

            tree = DecisionTree(
                train_data=self.train_data,
                min_points=self.min_points,
                max_depth=self.max_depth,
                num_features=self.num_features,
                curr_depth=self.curr_depth
            )

            tree.train()
            self.trees.append(tree)

    def best_label(self, prediction):
        """
        Finds the most common label in a list of predictions.

        :param prediction: List of predicted labels.
        :return: The most common label in the predictions.
        """
        counter = Counter(prediction)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, features):
        """
        Predicts labels or values for the given features using the Random Forest.

        :param features: The features to make predictions for.
        :return: Predicted labels or values for the features.
        """

        predictions = np.array([tree.predict(features) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.best_label(prediction=prediction) for prediction in tree_predictions])

        return predictions

