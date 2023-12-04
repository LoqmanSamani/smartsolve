from machine_learning.linear_algebra import intro_numpy as np
from collections import Counter


class DecisionTree:
    """
    A Decision Tree classifier for binary classification problems.

    :param train_data: The training data.
    :param min_points: The minimum number of data points required to split a node (default is 2).
    :param max_depth: The maximum depth of the decision tree (default is 10).
    :param num_features: The number of random features to consider for each split (default is None).
    :param curr_depth: The current depth of the tree while building (default is 0).
    """

    def __init__(self, train_data, min_points=2, max_depth=10, num_features=None, curr_depth=0):

        self.train_data = train_data
        self.min_points = min_points
        self.max_depth = max_depth
        self.num_features = num_features
        self.curr_depth = curr_depth

        self.root = None

    def train(self):
        """
        Train the decision tree on the provided training data.
        """
        features = np.array([point[1] for point in self.train_data])
        labels = np.array([point[0] for point in self.train_data])
        if not self.num_features:
            self.num_features = features.shape[1]
        else:
            self.num_features = min(features.shape[1], self.num_features)

        self.root = self.tree(
            features=features,
            labels=labels,
            curr_depth=self.curr_depth
        )

    def tree(self, features, labels, curr_depth):
        """
        Recursively build the decision tree.

        :param features: The features of the data.
        :param labels: The labels of the data.
        :param curr_depth: The current depth in the tree.
        :return: The root node of the decision tree.
        """

        num_points, num_features = features.shape
        num_labels = len(np.unique(labels))

        if curr_depth >= self.max_depth or num_labels == 1 or num_points < self.min_points:
            leaf_value = self.best_label(labels=labels)

            return DecisionTreeNode(value=leaf_value)

        feature_index = np.random.choice(num_features, self.num_features, replace=False)

        best_feature, best_threshold = self.best_split(
            features=features,
            labels=labels,
            feature_index=feature_index
        )

        left_index, right_index = self.split(
            column=features[:, best_feature],
            threshold=best_threshold
        )

        left = self.tree(features[left_index, :], labels[left_index], self.curr_depth + 1)
        right = self.tree(features[right_index, :], labels[right_index], self.curr_depth + 1)

        return DecisionTreeNode(best_feature, best_threshold, left, right)

    def best_split(self, features, labels, feature_index):
        """
        Find the best feature and threshold for splitting the data.

        :param features: The features of the data.
        :param labels: The labels of the data.
        :param feature_index: The indices of features to consider for splitting.
        :return: The best feature and threshold for splitting.
        """
        best_gain = -1
        split_index, split_threshold = None, None

        for index in feature_index:
            column = features[:, index]
            thresholds = np.unique(column)

            for threshold in thresholds:

                gain = self.info_gain(
                    labels=labels,
                    column=column,
                    threshold=threshold
                )

                if gain > best_gain:
                    best_gain = gain
                    split_index = index
                    split_threshold = threshold

        return split_index, split_threshold

    def info_gain(self, labels, column, threshold):
        """
        Calculate the information gain for a split.

        :param labels: The labels of the data.
        :param column: The column (feature) being split.
        :param threshold: The threshold for splitting the column.
        :return: The information gain for the split.
        """

        parent_entropy = self.entropy(labels)

        left_index, right_index = self.split(column, threshold)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(labels)
        n_left, n_right = len(left_index), len(right_index)
        e_left, e_right = self.entropy(labels=labels[left_index]), self.entropy(labels=labels[right_index])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        info_gain = parent_entropy - child_entropy

        return info_gain

    def split(self, column, threshold):
        """
        Split a column into left and right indices based on a threshold.

        :param column: The column (feature) to be split.
        :param threshold: The threshold for splitting the column.
        :return: Indices of the left and right subsets after the split.
        """
        left_index = np.argwhere(column <= threshold).flatten()
        right_index = np.argwhere(column > threshold).flatten()
        return left_index, right_index

    def entropy(self, labels):
        """
        Calculate the entropy of a set of labels.

        :param labels: The labels for which to calculate entropy.
        :return: The entropy of the labels.
        """

        hist = np.bincount(labels)
        ps = hist / len(labels)

        entropy = -np.sum([p * np.log(p) for p in ps if p > 0])

        return entropy

    def best_label(self, labels):
        """
        Find the most common label in a set of labels.

        :param labels: The labels for which to find the most common label.
        :return: The most common label.
        """
        counter = Counter(labels)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, data):
        """
        Predict the labels for a set of data points.

        :param data: The data points for which to make predictions.
        :return: An array of predicted labels.
        """

        return np.array([self.traverse_tree(point=point, node=self.root) for point in data])

    def traverse_tree(self, point, node):
        """
        Traverse the decision tree to predict a label for a data point.

        :param point: The data point for which to make a prediction.
        :param node: The current node in the decision tree.
        :return: The predicted label.
        """

        if node.leaf_node():
            return node.value

        if point[node.feature] <= node.threshold:
            return self.traverse_tree(point, node.left)

        return self.traverse_tree(point, node.right)


class DecisionTreeNode:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initialize a node for a decision tree.

        :param feature: The feature index for the node.
        :param threshold: The threshold for splitting the feature.
        :param left: The left child node.
        :param right: The right child node.
        :param value: The predicted value for a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_node(self):
        """
        Check if the current node is a leaf node.

        :return: True if the node is a leaf, False otherwise.
        """
        return self.value is not None





