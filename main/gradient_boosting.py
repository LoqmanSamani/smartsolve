from learnflow.models import DecisionTree
import numpy as np


class GradientBoosting:
    """
    A gradient boosting classifier.

    Args:
    train_data (list): Training data in the form [(label, features), ...].
    min_points (int): Minimum number of data points for a node (default is 2).
    max_depth (int): Maximum depth of decision trees (default is 2).
    num_features (int): Number of features to consider for each tree (default is None).
    num_trees (int): Number of boosting iterations (default is 10).
    curr_depth (int): Current depth during training (default is 0).
    threshold (float): Classification threshold (default is 0.5).
    """

    def __init__(self, train_data, min_points=2, max_depth=2, num_features=None, num_trees=10,
                 curr_depth=0, threshold=0.5):

        self.train_data = train_data
        self.max_depth = max_depth
        self.num_features = num_features
        self.min_points = min_points
        self.num_trees = num_trees
        self.curr_depth = curr_depth
        self.threshold = threshold
        self.trees = []

    def train(self):
        """
        Train the gradient boosting classifier.
        """

        features = [point[1] for point in self.train_data]
        labels = [point[0] for point in self.train_data]
        data = [(label, point) for label, point in zip(labels, features)]

        for i in range(self.num_trees):

            tree = DecisionTree(train_data=data,
                                min_points=self.min_points,
                                max_depth=self.max_depth,
                                num_features=self.num_features,
                                curr_depth=self.curr_depth
                                )

            tree.train()
            predicted = tree.predict(features)
            labels = [np.array(labels) - predicted]

            self.trees.append(tree)

    def predict(self, features):
        """
        Make predictions using the trained gradient boosting classifier.

        Args:
        features (list): List of feature vectors for prediction.

        Returns:
        numpy.ndarray: Predicted labels (0 or 1) for each input feature vector.
        """

        # Initialize a list to store the prediction value for each tree
        trees_predictions = np.empty((len(features), len(self.trees)))

        for i, tree in enumerate(self.trees):
            trees_predictions[:, i] = tree.predict(features)

        predictions = np.sum(trees_predictions, axis=1)

        predictions = np.float64(predictions >= self.threshold)

        return predictions

