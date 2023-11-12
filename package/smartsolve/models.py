import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal



class LinearRegression:
    def __init__(self, train_data,  coefficients=None, bias=None, learning_rate=1e-2, max_iter=1000, threshold=1e-8,
                 seed=42, norm='yes'):
        """
        Initialize a Linear Regression model.
        parameters:
        :param train_data: Training data in the form of [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n])
                          ,...,(ym,[xm1,xm2,...,xmn])].
        :param coefficients: Initial coefficients for the model (default is None).
        :param bias: Bias term for the model (default is None).
        :param learning_rate: Learning rate for gradient descent (default is 1e-2).
        :param max_iter: Maximum number of iterations for gradient descent (default is 1000).
        :param threshold: Convergence threshold for gradient descent (default is 1e-8).
        :param seed: Random seed for reproducibility (default is 42).
        :param norm: Normalize features or not ('yes' or 'no', default is 'yes').
        output:
        None
        """

        self.train_data = train_data
        self.coefficients = coefficients
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.seed = seed
        self.norm = norm

        self.means = None
        self.stds = None
        self.mse = []

    def mean_squared_error(self, labels, predicted):
        """
        Calculate the Mean Squared Error (MSE) between actual labels and predicted labels.

        :param labels: Actual labels.
        :param predicted: Predicted labels.
        :return: Mean Squared Error.
        """

        mse = np.mean(np.power(labels - predicted, 2))

        return mse

    def stoch_gradient(self, features, labels, coefficients, index):
        """
        Perform a stochastic gradient descent step.

        :param features: Feature matrix.
        :param labels: Actual labels.
        :param coefficients: Current model coefficients.
        :param index: Index of the data point for the stochastic gradient descent step.
        :return: Updated coefficients after the step.
        """

        difference = np.dot(features[index, :], coefficients) - labels[index]
        gradient = features[index, :] * difference

        num_points = len(labels)

        new_weights = coefficients - 2 * (self.learning_rate / num_points) * gradient

        return new_weights

    def stoch_gradient_descent(self, features, labels, coefficients):
        """
        Perform stochastic gradient descent to optimize model coefficients.

        :param features: Feature matrix.
        :param labels: Actual labels.
        :param coefficients: Initial model coefficients.
        :return: Optimized coefficients.
        """

        co_dist = np.inf
        coefficients = coefficients

        num_iter = 0

        np.random.seed(self.seed)

        while co_dist > self.threshold and num_iter < self.max_iter:

            random_index = np.random.randint(features.shape[0])

            new_coefficients = self.stoch_gradient(
                features=features,
                labels=labels,
                coefficients=coefficients,
                index=random_index
            )

            mse = self.mean_squared_error(labels, np.dot(features, coefficients))
            self.mse.append(mse)

            num_iter += 1

            co_dist = np.sum(np.power(coefficients - co_dist, 2))

            coefficients = new_coefficients

        return coefficients

    def train(self):
        """
        Train the Linear Regression model on the provided training data.
        """

        features = np.array([point[1] for point in self.train_data])
        labels = np.array([point[0] for point in self.train_data])

        # Store the means and stds to normalize the features when needed for predictions
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        if not self.bias:
            self.bias = 1

        if self.norm == 'yes':
            features = (features - self.means) / self.stds

        features = np.concatenate((np.ones(len(features)).reshape(len(features), self.bias), features), axis=1)

        if not self.coefficients:
            self.coefficients = np.random.random(features.shape[1])

        self.coefficients = self.stoch_gradient_descent(
            features=features,
            labels=labels,
            coefficients=self.coefficients
        )

    def predict(self, data, norm='yes'):
        """
        Predict labels for input data.

        :param data: Input data for which to make predictions.
        :param norm: Normalize data or not ('yes' or 'no', default is 'no').
        :return: Predicted labels.
        """

        data = np.array(data)

        if norm == 'yes':
            data = (data - self.means) / self.stds

        data = np.concatenate((np.ones(len(data)).reshape(len(data), self.bias), data), axis=1)

        predicted = np.dot(data, self.coefficients)

        return predicted




class LogisticRegression:

    def __init__(self, train_data, coefficients=None, bias=None, learning_rate=1e-2, max_iter=1000, threshold=1e-8,
                 seed=42, norm='yes'):
        """
        Initialize a LogisticRegression model.

        :param train_data: Training data in the form [(y1,[x11,x12,...,x1n]), ... ,(ym,[xm1,xm2,...,xmn])].
        :param coefficients: Initial coefficients for the model (default is None).
        :param bias: Bias term (default is None).
        :param learning_rate: Learning rate for gradient descent (default is 1e-2).
        :param max_iter: Maximum number of iterations for gradient descent (default is 1000).
        :param threshold: Convergence threshold for stopping criteria (default is 1e-8).
        :param seed: Random seed for reproducibility (default is 42).
        :param norm: Whether to normalize features ('yes' or 'no', default is 'yes').

        """

        self.train_data = train_data
        self.coefficients = coefficients
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.seed = seed
        self.norm = norm

        self.means = None
        self.stds = None
        self.mse = []

    def mean_squared_error(self, labels, predicted):
        """
        Calculate the mean squared error between actual labels and predicted labels.

        :param labels: Actual labels.
        :param predicted: Predicted labels.
        :return: Mean squared error.
        """

        mse = np.mean(np.power(labels - predicted, 2))

        return mse

    def sigmoid(self, predicted):
        """
        Compute the sigmoid function for the given predicted values.

        :param predicted: Predicted values.
        :return: Sigmoid-transformed values.
        """
        predicted = 1 / (1 + np.exp(-predicted))

        return predicted

    def stoch_gradient(self, features, labels, coefficients, index):
        """
        Perform a stochastic gradient descent step.

        :param features: Feature matrix.
        :param labels: Actual labels.
        :param coefficients: Current model coefficients.
        :param index: Index of the data point for the stochastic gradient descent step.
        :return: Updated coefficients after the step.
        """

        predicted = np.dot(features[index, :], coefficients)

        predicted = self.sigmoid(predicted=predicted)

        difference = predicted - labels[index]

        gradient = features[index, :] * difference

        num_points = len(labels)

        new_coefficients = coefficients - 2 * (self.learning_rate / num_points) * gradient

        return new_coefficients

    def stoch_gradient_descent(self, features, labels, coefficients):
        """
        Perform stochastic gradient descent.

        :param features: Feature matrix.
        :param labels: Actual labels.
        :param coefficients: Initial model coefficients.
        :return: Updated coefficients after gradient descent.
        """

        co_dist = np.inf
        coefficients = coefficients

        num_iter = 0
        np.random.seed(self.seed)

        while co_dist > self.threshold and num_iter < self.max_iter:

            random_index = np.random.randint(features.shape[0])

            new_coefficients = self.stoch_gradient(
                features=features,
                labels=labels,
                coefficients=coefficients,
                index=random_index
            )

            mse = self.mean_squared_error(labels, np.dot(features, coefficients))
            self.mse.append(mse)

            num_iter += 1

            co_dist = np.sum(np.power(coefficients - co_dist, 2))

            coefficients = new_coefficients

        return coefficients

    def train(self):
        """
        Train the logistic regression model.
        """

        features = np.array([point[1] for point in self.train_data])
        labels = np.array([point[0] for point in self.train_data])

        # Store the means and stds to normalize the features when needed for predictions
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        if not self.bias:
            self.bias = 1

        if self.norm == 'yes':
            features = (features - self.means) / self.stds

        features = np.concatenate((np.ones(len(features)).reshape(len(features), self.bias), features), axis=1)

        if not self.coefficients:
            self.coefficients = np.random.random(features.shape[1])

        self.coefficients = self.stoch_gradient_descent(features=features, labels=labels,
                                                        coefficients=self.coefficients)

    def predict(self, data, norm='no'):
        """
        Predict labels for the given data.

        :param data: Data for which to make predictions.
        :param norm: Whether to normalize features ('yes' or 'no', default is 'no').
        :return: Predicted labels.
        """

        data = np.array(data)

        if norm == 'yes':
            data = (data - self.means) / self.stds

        data = np.concatenate((np.ones(len(data)).reshape(len(data), self.bias), data), axis=1)

        predicted = np.dot(data, self.coefficients)

        predicted = np.int64(predicted >= 0.5)

        return predicted


    def ppredict(self, data):
        """
        Perform probability predictions for the given data.

        :param data: Data for which to make probability predictions.
        :return: Probability predictions.
        """
        data = np.array(data)

        if self.norm == 'yes':
            data = (data - self.means) / self.stds

        data = np.concatenate((np.ones(len(data)).reshape(len(data), self.bias), data), axis=1)

        predicted = np.dot(data, self.coefficients)

        predicted = self.sigmoid(predicted)

        return predicted




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




class KMeansClustering:

    def __init__(self, train_data, num_clusters=2, max_iter=100, threshold=1e-3):
        """
        Initializes a KMeansClustering object.

        :param train_data: The training data.
        :param num_clusters: The number of clusters to create.
        :param max_iter: The maximum number of iterations for the K-Means algorithm.
        :param threshold: The convergence threshold for centroid updates.
        """

        self.train_data = train_data
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.num_dims = len(self.train_data[0])
        self.threshold = threshold

        self.clusters = {}  # a dictionary to store the detected clusters

    def train(self):
        """
        Train the KMeans model on the provided data.
        """

        data = np.array(self.train_data)
        prev_centroids = self.initialization(data, self.num_clusters)
        cluster = [0 for i in range(len(self.train_data))]
        cluster_keys = [i for i in range(self.num_clusters)]
        change_centroids = 1e+10

        for i in range(self.max_iter):

            if change_centroids > self.threshold:

                cluster = self.cluster(self.train_data, self.num_clusters, prev_centroids)

                new_centroids = self.centroids(self.train_data, self.num_clusters, cluster)
                change_centroids = self.measure_change(new_centroids, prev_centroids)
                prev_centroids = new_centroids

        for key in cluster_keys:
            for j in range(len(self.train_data)):
                if key == cluster[j]:
                    if f"Cluster {key+1}" not in self.clusters:
                        self.clusters[f"Cluster {key+1}"] = [self.train_data[j]]
                    else:
                        self.clusters[f"Cluster {key+1}"].append(self.train_data[j])

    def initialization(self, data, num_clusters):
        """
        Initialize cluster centroids using random data points from the dataset.

        :param data: The dataset for initialization.
        :param num_clusters: The number of clusters to create.
        :return: A NumPy array of initial cluster centroids.
        """

        centroids = []
        num_features = len(data[0])

        for i in range(num_clusters):
            centroid = []
            for j in range(num_features):
                cx = np.random.uniform(min(data[:, j]), max(data[:, j]))
                centroid.append(cx)

            centroids.append(centroid)

        return np.asarray(centroids)

    def cluster(self,data, num_cluster, prev_centroids):
        """
        Assign data points to clusters based on the closest centroid.

        :param data: The dataset to cluster.
        :param num_cluster: The number of clusters.
        :param prev_centroids: The previous cluster centroids.
        :return: An array of cluster assignments.
        """

        cluster = [-1 for _ in range(len(data))]
        for i in range(len(data)):
            dist_arr = []
            for j in range(num_cluster):
                dist_arr.append(self.distance(data[i], prev_centroids[j]))
            idx = np.argmin(dist_arr)
            cluster[i] = idx
        return np.asarray(cluster)

    def distance(self, a, b):
        """
        Compute the Euclidean distance between two data points.

        :param a: The first data point.
        :param b: The second data point.
        :return: The Euclidean distance between a and b.
        """
        distance = np.sqrt(sum(np.square(a - b)))
        return distance

    def centroids(self, data, num_cluster, cluster):
        """
        Compute new centroids for each cluster based on the assigned data points.

        :param data: The dataset.
        :param num_cluster: The number of clusters.
        :param cluster: The cluster assignments for each data point.
        :return: An array of updated cluster centroids.
        """
        cg_arr = []
        for i in range(num_cluster):
            arr = []
            for j in range(len(data)):
                if cluster[j] == i:
                    arr.append(data[j])
            cg_arr.append(np.mean(arr, axis=0))
        return np.asarray(cg_arr)

    def measure_change(self, prev_centroids, new_centroids):
        """
        Measure the change in centroids between iterations to check for convergence.

        :param prev_centroids: The centroids from the previous iteration.
        :param new_centroids: The updated centroids in the current iteration.
        :return: The measure of change in centroids.
        """
        res = 0
        for a, b in zip(prev_centroids, new_centroids):
            res += self.distance(a, b)
        return res



class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) algorithm for classification and regression.

    Parameters:
        train_data (list): The training data in the format [(y1,[x11,x12,...,x1n]), ...].
        num_neighbors (int): The number of neighbors to consider (default is 5).
        distance (str): The distance metric to use ('EU', 'MA, 'MI', 'CH', or 'CO').
        algorithm (str): The type of task ('classification' or 'regression').
        p_value (int): The p-value for the Minkowski distance (default is 2).

    Attributes:
        features (numpy.ndarray): The features of the training data.
        labels (numpy.ndarray): The labels of the training data.

    Methods:
        fit(): Fit the model to the training data.
        custom_distance(point1, point2): Calculate the custom distance between two data points.
        euclidean (point1, point2): Calculate the Euclidean distance between two points.
        manhattan(point1, point2): Calculate the Manhattan distance between two points.
        minkowski(point1, point2): Calculate the Minkowski distance between two points.
        chebyshev(point1, point2): Calculate the Chebyshev distance between two points.
        cosine(point1, point2): Calculate the Cosine distance between two points.
        predict(data): Make predictions for a list of data points.
        point_predict(point): Predict the class or value of a single data point.
    """
    def __init__(self, train_data, num_neighbors=5, distance='EU', algorithm='classification', p_value=2):
        self.train_data = train_data
        self.num_neighbors = num_neighbors
        self.distance = distance
        self.algorithm = algorithm
        self.p_value = p_value
        self.features = None
        self.labels = None

    def fit(self):
        """
        Fit the model to the training data.

        Returns:
            None
        """
        self.features = np.array([point[1] for point in self.train_data])
        self.labels = np.array([point[0] for point in self.train_data])

    def custom_distance(self, point1, point2):
        """
        Calculate the custom distance between two data points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The distance between the two points.
        """

        if self.distance == 'EU':
            return self.euclidean(point1=point1, point2=point2)

        elif self.distance == 'MA':
            return self.manhattan(point1=point1, point2=point2)

        elif self.distance == 'MI':
            return self.minkowski(point1=point1, point2=point2)

        elif self.distance == 'CH':
            return self.chebyshev(point1=point1, point2=point2)

        elif self.distance == 'CO':
            return self.cosine(point1=point1, point2=point2)

        else:
            raise ValueError("Invalid distance algorithm. available are: 'EU' (euclidean distance),"
                             " 'MA' (manhattan distance), 'MI'(minkowski distance), 'CH' (chebyshev distance),"
                             "'CO' (cosine distance).")

    def euclidean(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Euclidean distance between the two points.
        """

        distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

        return distance

    def manhattan(self, point1, point2):
        """
        Calculate the Manhattan distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Manhattan distance between the two points.
        """

        distance = np.sum(np.abs(np.array(point1) - np.array(point2)))

        return distance

    def minkowski(self, point1, point2):
        """
        Calculate the Minkowski distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Minkowski distance between the two points.
        """

        distance = (np.sum(np.abs(np.array(point1) - np.array(point2))
                           ** self.p_value)) ** (1 / self.p_value)

        return distance

    def chebyshev(self, point1, point2):
        """
        Calculate the Chebyshev distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Chebyshev distance between the two points.
        """

        distance = np.max(np.abs(np.array(point1) - np.array(point2)))

        return distance

    def cosine(self, point1, point2):
        """
        Calculate the Cosine distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Cosine distance between the two points.
        """

        dot_product = np.dot(point1, point2)
        norm_p = np.linalg.norm(point1)
        norm_q = np.linalg.norm(point2)
        distance = 1.0 - (dot_product / (norm_p * norm_q))
        return distance

    def predict(self, data):
        """
        Make predictions for a list of data points.

        Parameters:
            data (list): A list of data points to make predictions for.

        Returns:
            list: Predictions for each data point.
        """
        predictions = [self.point_predict(point=point) for point in data]
        return predictions

    def point_predict(self, point):
        """
        Predict the class or value of a single data point.

        Parameters:
            point (list): A single data point.

        Returns:
            int or float: The predicted class (for classification) or value (for regression).
        """
        distances = [self.custom_distance(point1=point, point2=train_point) for train_point in self.features]
        k_indices = np.argsort(distances)[:self.num_neighbors]
        k_nearest_labels = [self.labels[i] for i in k_indices]

        if self.algorithm == 'classification':
            most_common = Counter(k_nearest_labels).most_common()
            return most_common[0][0]
        elif self.algorithm == 'regression':
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Invalid algorithm. Supported values: 'classification' or 'regression'")




class NaiveBayes:
    def __init__(self, train_data, algorithm='classification'):
        """
        Initializes a NaiveBayes classifier/regressor.

        :param train_data: Input training data in the form of [(y1, [x11, x12, ..., x1n]), (y2, [x21, x22, x2n]), ...].
        :param algorithm: The type of algorithm, either 'classification' or 'regression' (default is 'classification').
        """
        self.train_data = train_data
        self.algorithm = algorithm

        self.classes = None
        self.mean = None  # For storing class means
        self.variance = None  # For storing class variances
        self.prior = None
        self.labels = None

    def train(self):
        """
        Trains the NaiveBayes model with the provided training data.
        """
        features = np.array([point[1] for point in self.train_data])
        labels = np.array([point[0] for point in self.train_data])

        num_points, num_features = features.shape
        self.classes = np.unique(labels)

        self.mean = np.zeros(shape=(len(self.classes), num_features), dtype=np.float64)
        self.variance = np.zeros(shape=(len(self.classes), num_features), dtype=np.float64)
        self.prior = np.zeros(len(self.classes), dtype=np.float64)

        for index, cls in enumerate(self.classes):
            feature_cls = features[labels == cls]
            self.mean[index, :] = feature_cls.mean(axis=0)
            self.variance[index, :] = feature_cls.var(axis=0)
            self.prior[index] = feature_cls.shape[0] / float(num_points)

        self.labels = labels

    def likelihood(self, data, mean, variance):
        """
        Calculates the likelihood of the data given the class mean and variance using Gaussian distribution.

        :param data: The data point for which to calculate the likelihood.
        :param mean: The mean of the class.
        :param variance: The variance of the class.
        :return: The likelihood of the data point.
        """

        epsilon = 1e-4
        coefficient = 1 / np.sqrt(2 * np.pi * variance + epsilon)

        exponent = np.exp(-((data - mean) ** 2 / (2 * variance + epsilon)))
        likelihood = coefficient * exponent

        return likelihood

    def predict(self, features):
        """
        Predicts the labels or values of the given features.

        :param features: The features to make predictions for.
        :return: Predicted labels or values for the features.
        """
        if self.algorithm == 'classification':

            predictions = [self.point_predict(feature=feature) for feature in features]

            return np.array(predictions)

        elif self.algorithm == 'regression':

            num_samples, _ = features.shape
            predictions = np.empty(num_samples)

            for index, feature in enumerate(features):

                posteriors = []

                for label_index, label in enumerate(self.classes):

                    prior = np.log((self.labels == label).mean())
                    pairs = zip(feature, self.mean[label_index], self.variance[label_index])

                    likelihood = np.sum([np.log(self.likelihood(data=data, mean=mean, variance=variance))
                                         for data, mean, variance in pairs])

                    posteriors.append(prior + likelihood)

                predictions[index] = self.classes[np.argmax(posteriors)]

            return predictions

        else:

            raise ValueError("Invalid algorithm. Supported values: 'classification' or 'regression")

    def point_predict(self, feature):
        """
        Predicts the label or value for a single data point (feature).

        :param feature: The data point (feature) to make a prediction for.
        :return: The predicted label or value for the data point.
        """

        posteriors = []

        for index, cls in enumerate(self.classes):

            prior = np.log(self.prior[index])
            posterior = np.sum(np.log(self.probability_density(class_index=index, feature=feature)))
            posterior = posterior + prior

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def probability_density(self, class_index, feature):
        """
        Calculates the probability density of a feature for a specific class.

        :param class_index: The index of the class.
        :param feature: The feature for which to calculate the probability density.
        :return: The probability density of the feature for the specified class.
        """

        mean = self.mean[class_index]
        var = self.variance[class_index]

        numerator = np.exp(-(np.power(feature - mean, 2)) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator



class PrincipalComponentAnalysis:

    def __init__(self, features, num_components):
        """
        Initializes a Principal Component Analysis (PCA) object.

        :param features: Input features as a 2D array.
        :param num_components: Number of principal components to retain.

        """

        self.features = features
        self.num_components = num_components

        self.components = None
        self.mean = None

    def train(self):
        """
        Performs PCA training on the input features.
        Computes principal components and updates the mean.
        """

        self.mean = np.mean(self.features, axis=0)
        features = self.features - self.mean

        covariance = np.cov(features.T)

        eigenvectors, eigenvalues = np.linalg.eig(covariance)

        eigenvectors = eigenvectors.T

        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[index]

        self.components = eigenvectors[:self.num_components]

    def transform(self):
        """
        Transforms the input features using the learned principal components.

        :return: Transformed features with reduced dimensions.
        """

        features = self.features - self.mean
        transformed = np.dot(features, self.components.T)

        return transformed





class SupportVectorMachines:
    """
    Support Vector Machines (SVM) classifier.

    Parameters:
    - training_data: List of training data points in the form [(y, [x1, x2, ..., xn]), ...].
    - optimization_algorithm: Optimization algorithm to use ('GD' for Gradient Descent,
      'SMO' for Sequential Minimal Optimization).

    - kernel: Kernel function to use ('linear', 'quadratic', 'gaussian').
    - learning_rate: Learning rate for GD (default is 1e-4).
    - lam: Regularization parameter (default is 1e-4).
    - max_iteration: Maximum number of iterations for training (default is 1000).
    - threshold: Convergence threshold for GD (default is 1e-5).
    - regularization_parameter: Regularization parameter for SMO (default is 1).
    - epsilon: Convergence tolerance for SMO (default is 1e-4).
    - sigma: Sigma value for the Gaussian kernel (default is -0.1).

    The training_data must have the structure:
    [(y1, [x11, x12, ..., x1n]), (y2, [x21, x22, x2n]), ..., (ym, [xm1, xm2, ..., xmn])]

    Methods:
    - train: Train the SVM model using the selected optimization algorithm.
    - predict: Make predictions on input data points after training.

    Note: Call the 'train' method before making predictions.

    """

    def __init__(self, training_data, optimization_algorithm='GD', kernel='linear', learning_rate=1e-4, lam=1e-4,
                 max_iteration=1000, threshold=1e-5, regularization_parameter=1, epsilon=1e-4, sigma=-0.1):

        self.training_data = training_data
        self.optimization_algorithm = optimization_algorithm

        kernels = {
            'linear': self.linear_kernel,
            'quadratic': self.quadratic_kernel,
            'gaussian': self.gaussian_kernel
        }

        self.kernel = kernels[kernel]
        self.learning_rate = learning_rate
        self.lam = lam
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.regularization_parameter = regularization_parameter
        self.epsilon = epsilon
        self.sigma = sigma
        self.weights = None
        self.bias = None
        self.algorithms = ['SGD', 'SMO']

    def train(self):
        """
        Train the Support Vector Machines (SVM) model using the specified optimization algorithm.

        This method iteratively updates the weights and bias to find the optimal decision boundary.

        Returns:
        None
        """
        # Training data
        points = np.array([item[1] for item in self.training_data])
        labels = np.array([item[0] for item in self.training_data])

        # Initialize weights and bias
        self.weights = np.zeros(points.shape[1])
        self.bias = 0.0

        num_iter_count = 0

        if self.optimization_algorithm == 'GD':
            # Train using Gradient Descent (GD)
            for j in range(self.max_iteration):
                for i, point in enumerate(points):
                    error = labels[i] * (np.dot(point, self.weights)) + self.bias
                    if error <= 1:
                        # Update weights and bias
                        d_weights, d_bias = self.der_hinge_loss(error, point, labels[i])
                        self.weights -= self.learning_rate * d_weights
                        self.bias -= self.learning_rate * d_bias
                    num_iter_count += 1
        elif self.optimization_algorithm == 'SMO':
            # Train using Sequential Minimal Optimization (SMO)
            # Add the SMO training logic here
            pass

        self.weights = self.weights  # Save the learned weights
        self.bias = self.bias  # Save the learned bias

    def der_hinge_loss(self, error, point, label):

        if error > 1:
            d_weights = 2 * self.lam * self.weights

            d_bias = 0

        else:
            d_weights = 2 * self.lam * self.weights - (point * label)

            d_bias = - label

        return d_weights, d_bias

    def linear_kernel(self, point1, point2):

        import numpy as np

        calc = np.dot(point1, point2.T)

        return calc

    def quadratic_kernel(self, point1, point2):

        import numpy as np

        calc = np.power(self.linear_kernel(point1, point2), 2)

        return calc

    def gaussian_kernel(self, point1, point2):

        import numpy as np

        calc = np.exp(-np.power(np.linalg.norm(point1 - point2), 2) / (2 * self.sigma ** 2))

        return calc

    def random_num(self, a, b, z):

        import numpy as np

        lst = list(range(a, z)) + list(range(z + 1, b))

        return np.random.choice(lst)

    def weight_calculator(self, point, label, alpha):

        import numpy as np

        weights = np.dot(point.T, np.multiply(alpha, label))

        return weights

    def bias_calculator(self, point, label, weights):

        import numpy as np

        bias = np.mean(label - np.dot(point, weights))

        return bias

    def prediction_error(self, point_k, label_k, weights, bias):

        import numpy as np

        error = np.sign(np.dot(point_k, weights) + bias) - label_k

        return error

    def compute_l_h(self, c, alpha_pj, alpha_pi, label_j, label_i):

        import numpy as np

        if label_i != label_j:

            bounds = np.max(0, alpha_pj - alpha_pi), np.min(c, c - alpha_pi + alpha_pj)

            return bounds

        else:

            bounds = np.max(0, alpha_pj + alpha_pi), np.min(c, c - alpha_pi + alpha_pj)

            return bounds

    def predict(self, points):
        """
        Make predictions using the trained SVM model.

        Parameters:
        - points: List of data points to make predictions on.

        Returns:
        List of predictions (-1 or 1) for each input point.
        """
        points = np.array(points)

        if self.weights is not None and self.bias is not None:

            predictions = [np.sign(np.dot(point, self.weights) + self.bias) for point in points]

            return predictions

        else:

            raise ValueError("Model has not been trained. You need to call the 'train' method first.")



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



class GaussianMixtureModel:
    def __init__(self, train_data, num_clusters, means=None, covariances=None, coefficients=None, max_iter=1000,
                 threshold=1e-4):
        """
        Initialize a Gaussian Mixture Model.

        :param train_data: Training data as a list of data points.
        :param num_clusters: Number of clusters in the model.
        :param means: Initial means for clusters (default is None).
        :param covariances: Initial covariances for clusters (default is None).
        :param coefficients: Initial coefficients for clusters (default is None).
        :param max_iter: Maximum number of iterations during training (default is 1000).
        :param threshold: Convergence threshold for log-likelihood (default is 1e-4).
        """

        self.train_data = train_data
        self.num_clusters = num_clusters
        self.means = means
        self.covariances = covariances
        self.coefficients = coefficients
        self.max_iter = max_iter
        self.threshold = threshold

        self.responsibilities = None
        self.loglikelihood = None
        self.loglikelihood_trace = []

    def train(self):
        """
        Train the Gaussian Mixture Model using the provided data and parameters.
        """

        if not self.means:

            np.random.seed(42)
            random_indices = np.random.choice(len(self.train_data), self.num_clusters, replace=False)
            self.means = [self.train_data[i] for i in random_indices]

        if not self.covariances:

            self.covariances = [np.identity(len(self.train_data[0])) for _ in range(self.num_clusters)]

        if not self.coefficients:
            self.coefficients = [1.0 / self.num_clusters] * self.num_clusters

        num_point = len(self.train_data)

        self.responsibilities = np.zeros((num_point, self.num_clusters))

        self.loglikelihood = self.compute_loglikelihood(
            data=self.train_data,
            coefficients=self.coefficients,
            means=self.means,
            covariances=self.covariances,
        )

        self.loglikelihood_trace = [self.loglikelihood]

        for i in range(self.max_iter):

            self.responsibilities = self.compute_response(
                data=self.train_data,
                coefficients=self.coefficients,
                means=self.means,
                covariances=self.covariances,
            )

            counts = self.soft_counts(
                responsibilities=self.responsibilities
            )

            self.coefficients = self.compute_coefficients(
                counts=counts
            )

            self.means = self.compute_means(
                data=self.train_data,
                responsibilities=self.responsibilities,
                counts=counts
            )

            covariances = self.compute_covariances(
                data=self.train_data,
                responsibilities=self.responsibilities,
                counts=counts,
                means=self.means
            )

            l_loglikelihood = self.compute_loglikelihood(
                data=self.train_data,
                coefficients=self.coefficients,
                means=self.means,
                covariances=covariances
            )

            self.loglikelihood_trace.append(l_loglikelihood)


    def compute_loglikelihood(self, data, coefficients, means, covariances):
        """
        Calculate the log-likelihood of the data given the model parameters.

        :param data: Data points for log-likelihood calculation.
        :param coefficients: Cluster coefficients.
        :param means: Cluster means.
        :param covariances: Cluster covariances.

        :return: Log-likelihood of the data.
        """

        num_clusters = len(means)
        num_dimension = len(data[0])

        loglikelihood = 0

        for point in data:

            results = np.zeros(num_clusters)

            for i in range(num_clusters):

                delta = np.array(point) - means[i]
                exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covariances[i]), delta))

                results[i] += np.log(coefficients[i])
                results[i] -= 1 / 2. * (num_dimension * np.log(2 * np.pi) + np.log(np.linalg.det(covariances[i])) + exponent_term)

            loglikelihood += self.log_sum_exp(results=results)

        return loglikelihood

    def log_sum_exp(self, results):
        """
        Calculate the log of the sum of exponentials.

        :param results: List of results to calculate the log-sum-exp for.

        :return: Log of the sum of exponentials.
        """

        loglikelihood = np.max(results) + np.log(np.sum(np.exp(results - np.max(results))))

        return loglikelihood

    def compute_response(self, data, coefficients, means, covariances):
        """
        Compute the responsibilities for each data point.

        :param data: Data points.
        :param coefficients: Cluster coefficients.
        :param means: Cluster means.
        :param covariances: Cluster covariances.

        :return: Responsibilities for each data point.
        """

        num_point = len(data)
        num_clusters = len(means)
        responsibilities = np.zeros((num_point, num_clusters))

        for i in range(num_point):
            for k in range(num_clusters):
                responsibilities[i, k] = coefficients[k] * multivariate_normal.pdf(data[i], means[k], covariances[k])

        row_sums = responsibilities.sum(axis=1)[:, np.newaxis]
        resp = responsibilities / row_sums

        return resp

    def soft_counts(self, responsibilities):
        """
        Calculate soft counts based on responsibilities.

        :param responsibilities: Responsibilities for data points.

        :return: Soft counts for each cluster.
        """

        counts = np.sum(responsibilities, axis=0)

        return counts

    def compute_coefficients(self, counts):
        """
        Calculate cluster coefficients based on soft counts.

        :param counts: Soft counts for each cluster.

        :return: Cluster coefficients.
        """

        num_clusters = len(counts)
        coefficients = [0.] * num_clusters

        for k in range(num_clusters):

            coefficients[k] = counts[k]

        return coefficients

    def compute_means(self, data, responsibilities, counts):
        """
        Calculate updated means for clusters.

        :param data: Data points.
        :param responsibilities: Responsibilities for data points.
        :param counts: Soft counts for each cluster.

        :return: Updated cluster means.
        """

        num_clusters = len(counts)
        num_data = len(data)
        means = [np.zeros(len(data[0]))] * num_clusters

        for k in range(num_clusters):

            weighted_sum = 0.
            for i in range(num_data):
                weighted_sum += responsibilities[i, k] * np.array(data[i])
            means[k] = weighted_sum / counts[k]

        return means

    def compute_covariances(self, data, responsibilities, counts, means):
        """
        Calculate updated covariances for clusters.

        :param data: Data points.
        :param responsibilities: Responsibilities for data points.
        :param counts: Soft counts for each cluster.
        :param means: Cluster means.

        :return: Updated cluster covariances.
        """

        num_clusters = len(counts)
        num_dim = len(data[0])
        num_data = len(data)
        covariances = [np.zeros((num_dim, num_dim))] * num_clusters

        for k in range(num_clusters):

            weighted_sum = np.zeros((num_dim, num_dim))

            for i in range(num_data):

                weighted_sum += responsibilities[i, k] * np.outer(data[i] - means[k], data[i] - means[k])

            covariances[k] = weighted_sum / counts[k]

        return covariances





class SingularValueDecomposition:

    def __init__(self, data, num_dimension):

        self.data = data  # the input matrix. can be a nested list,
        # which will be converted to a matrix during the process
        self.num_dimension = num_dimension  # number of dimensions to return after processing the input data
        self.U = None  # The left singular vectors
        self.S = None  # The singular values
        self.Vt = None  # The transpose of the right singular vectors
        self.variance = None  # Proportion of total variance explained
        self.t_matrix = None  # transformed matrix
        """
        Singular Value Decomposition (SVD) for dimensionality reduction.

        Args:
            data (list or numpy.ndarray): Input data in the form of a nested list or a numpy array.
            num_dimension (int): Number of dimensions to return after processing the input data.

        Attributes:
            data: The input matrix, which can be a nested list or a numpy array.
            num_dimension: The number of dimensions to return after processing the input data.
            U: The left singular vectors.
            S: The singular values.
            Vt: The transpose of the right singular vectors.
            variance: Proportion of total variance explained.
            t_matrix: Transformed matrix with reduced dimensions.
        """


    def matrix_decomposer(self, matrix):
        """
        Perform SVD decomposition on the input matrix.

        Args:
            matrix (numpy.ndarray): Input matrix.

        Returns:
            left_singular_vector (numpy.ndarray): The left singular vectors.
            singular_values (numpy.ndarray): The singular values.
            t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.
        """

        # using nampy.linalg.svd to calculate all three returned matrices

        left_singular_vector, singular_values, t_right_singular_vectors = np.linalg.svd(
            matrix, full_matrices=False, compute_uv=True
        )

        return left_singular_vector, singular_values, t_right_singular_vectors

    def dimension_reducer(self, left_singular_vector, singular_values, t_right_singular_vectors):
        """
        Reduce dimensions of the decomposed matrices.

        Args:
            left_singular_vector (numpy.ndarray): The left singular vectors.
            singular_values (numpy.ndarray): The singular values.
            t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.

        Returns:
            k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
            k_singular_matrix (numpy.ndarray): Reduced singular values as a diagonal matrix.
            k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.
        """
        # Reduce the dimensions of left_singular_vector
        k_left_singular_vector = left_singular_vector[:, :self.num_dimension]

        # Create a diagonal matrix using singular values and truncate it to the first num_dimension values
        k_singular_matrix = np.diag(singular_values[:self.num_dimension])

        # Reduce the dimensions of t_right_singular_vectors
        k_right_singular_vectors = t_right_singular_vectors[:self.num_dimension, :]

        return k_left_singular_vector, k_singular_matrix, k_right_singular_vectors



    def matrix_reconstructor(self, k_left_singular_vector, k_singular_matrix, k_right_singular_vectors):
        """
        Reconstruct the original matrix from reduced dimensions.

        Args:
            k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
            k_singular_matrix (numpy.ndarray): Reduced singular values as a diagonal matrix.
            k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.

        Returns:
            reconstructed_matrix (numpy.ndarray): Reconstructed matrix.
        """
        # reconstruct the matrix using this formula: A = U * S * Vt
        reconstructed_matrix = np.dot(k_left_singular_vector, k_singular_matrix).dot(k_right_singular_vectors)

        return reconstructed_matrix




    def transform(self):
        """
        Perform dimensionality reduction and reconstruction in a single step.

        Returns:
            reconstructed_matrix (numpy.ndarray): Transformed matrix with reduced dimensions.
        """
        # check the type of the input data
        if isinstance(self.data, np.ndarray):
            matrix = self.data
        elif isinstance(self.data, list):
            matrix = np.array(self.data)  # converts the input data to a matrix for further calculations
        else:
            raise ValueError("The input must be in form numpy.ndarray or a nested list of points")

        if not self.num_dimension:
            raise ValueError("Please define the number of dimensions (num_dimension=?)of the output matrix.")

        # call matrix_decomposer to decompose the input data(matrix)
        self.U, self.S, self.Vt = self.matrix_decomposer(matrix=matrix)

        # call dimension_reducer  to reduce dimensions of the decomposed vectors
        k_left_singular_vector, k_singular_matrix, k_right_singular_vectors = self.dimension_reducer(
            left_singular_vector=self.U,
            singular_values=self.S,
            t_right_singular_vectors=self.Vt
        )

        # call matrix_reconstructor to combine the decomposed vectors to build the dimension-reduced matrix
        self.t_matrix = self.matrix_reconstructor(
            k_left_singular_vector=k_left_singular_vector,
            k_singular_matrix=k_singular_matrix,
            k_right_singular_vectors=k_right_singular_vectors
        )
        self.variance = self.variance_explained(
            singular_values=self.S,
            num_dimension=self.num_dimension
        )

        return self.t_matrix



    def variance_explained(self, singular_values, num_dimension):
        """
        Calculate the variance explained by the first num_dimensions singular values.

        Args:
            singular_values (numpy.ndarray): The singular values.
            num_dimension (int): Number of dimensions to consider.

        Returns:
            variance_explained (float): Proportion of total variance explained.
        """

        squared_values = np.power(singular_values[:num_dimension], 2)
        sum_all_squared_values = np.sum(np.power(singular_values, 2))

        variance_explained = squared_values / sum_all_squared_values

        self.variance = variance_explained

        return variance_explained


