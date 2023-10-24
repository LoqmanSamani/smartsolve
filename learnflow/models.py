import numpy as np
import random
import os
import joblib
import sys
from collections import Counter


class LinearRegression:

    def __init__(self, train_data, coefficients=None, bias=None, algorithm='GD', learning_rate=1e-4,
                 max_iter=200,  threshold=1e-6, proportion=0.1, alpha=0.01):
        """
        Initialize a LinearRegression model.

        Args:
            train_data (list): List of training data points with the structure:
                [(y1, [x11, x12, ..., x1n]), (y2, [x21, x22, x2n]), ..., (ym, [xm1, xm2, ..., xmn])]
            coefficients (numpy.ndarray, optional): Initial coefficients for the model.
            bias (float, optional): Initial bias for the model.
            algorithm (str, optional): The training algorithm to use ('GD', 'SGD', 'MGD', 'NE', 'RR').
            learning_rate (float, optional): Learning rate for gradient-based algorithms.
            max_iter (int, optional): Maximum number of training iterations.
            threshold (float, optional): Convergence threshold for stopping the training.
            proportion (float, optional): Proportion of data used in each iteration (only for MGD algorithm).
            alpha (float, optional): Regularization parameter for Ridge Regression ('RR').

        Attributes:
            train_data (list): The training data.
            algorithm (str): The training algorithm.
            learning_rate (float): The learning rate.
            max_iter (int): Maximum number of iterations.
            threshold (float): Convergence threshold.
            proportion (float): Proportion of data used in each iteration.
            alpha (float): Regularization parameter for Ridge Regression.
            mse (list): List to store Mean Squared Error during training.
            coefficients (numpy.ndarray): Coefficients for the linear regression model.
            bias (float): Bias term for the linear regression model.
        """

        self.train_data = train_data
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold  # convergence threshold
        self.proportion = proportion  # a proportion of data which will be used in each iteration using MGD algorithm
        self.alpha = alpha

        self.mse = []
        self.coefficients = coefficients
        self.bias = bias  # regularization parameter used in ridge regression

    def train(self):
        """
        Train the Linear Regression model using the specified algorithm.

        If a pre-trained model is available, it loads the model instead of training.

        Note: If a pre-trained model is not available, this method trains the model,
        saves it to a file, and can later be loaded for inference.
        """

        if os.path.exists("linear_regression_model.pkl"):
            # Load the model from the file
            with open("linear_regression_model.pkl", "rb") as model_file:
                saved_model = joblib.load(model_file)
                self.coefficients = saved_model.coefficients
                self.bias = saved_model.bias

        else:
            algorithms = ['GD', 'SGD', 'MGD', 'NE', 'RR']
            # Separate independent and dependent variables
            features = np.array([item[1] for item in self.train_data])
            labels = np.array([item[0] for item in self.train_data])

            num_points, num_feats = features.shape
            num_iter = 1  # tracks the number of iterations

            if self.algorithm in algorithms:

                if self.algorithm == 'NE':
                    self.coefficients, self.bias = self.norm_equation(features=features, labels=labels)

                elif self.algorithm == 'RR':
                    self.coefficients, self.bias = self.ridge_regression(features=features, labels=labels, alpha=self.alpha)

                else:
                    # initialize weights and bias term
                    if not self.coefficients:
                        self.coefficients = np.full(shape=num_feats, fill_value=float(0))
                    if not self.bias:
                        self.bias = float(0)

                    for _ in range(self.max_iter):

                        predicted = self.predict(features=features, coefficients=self.coefficients, bias=self.bias)

                        self.mse.append(self.mean_squared_error(predicted=predicted, labels=labels))

                        pre_coefficients = self.coefficients
                        pre_bias = self.bias

                        if self.algorithm == 'GD':

                            self.gradient_descent(
                                features=features,
                                labels=labels,
                                predicted=predicted,
                                num_points=num_points
                            )

                        elif self.algorithm == 'SGD':

                            self.stoch_gradient_descent(
                                features=features,
                                labels=labels,
                                predicted=predicted
                            )

                        elif self.algorithm == 'MGD':

                            self.minibatch_gradient_descent(
                                features=features,
                                labels=labels,
                                predicted=predicted
                            )

                        w_diffs = self.coefficients - pre_coefficients
                        bias_diff = self.bias - pre_bias

                        if all(w_diffs) < self.threshold and bias_diff < self.threshold:
                            print(f"After {num_iter} iterations the model is trained!")
                            break

                        else:
                            num_iter += 1

            else:
                raise ValueError("Please provide a valid algorithm. Available options are:"
                                 "'GD', 'SGD', 'MGD', 'NE' and 'RR'.")

            with open("linear_regression_model.pkl", "wb") as model_file:
                joblib.dump(self, model_file)

    def predict(self, features, coefficients, bias):
        """
        Make predictions using the trained Linear Regression model.

        Args:
            features (numpy.ndarray): Input features for prediction.
            coefficients (numpy.ndarray): Coefficients for the model.
            bias (float): Bias term for the model.

        Returns:
            numpy.ndarray: Predicted values.
        """

        predicted = np.dot(features, coefficients) + bias

        return predicted

    def mean_squared_error(self, labels, predicted):
        """
        Calculate Mean Squared Error (MSE) for model predictions.

        Args:
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted values.

        Returns:
            float: Mean Squared Error (MSE).
        """

        mse = np.mean(np.power(labels - predicted, 2))

        return mse

    def gradient_descent(self, features, labels, predicted, num_points):
        """
        Perform a single step of Gradient Descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted values.
            num_points (int): Number of data points.

        Updates:
            The model's coefficients and bias based on the gradient descent step.
        """
        d_coefficients = (1/num_points) * np.dot(features.T, (predicted - labels))
        d_bias = (1/num_points) * np.sum(predicted - labels)

        self.coefficients -= self.learning_rate * d_coefficients
        self.bias -= self.learning_rate * d_bias

    def stoch_gradient_descent(self, features, labels, predicted):
        """
        Perform a single step of Stochastic Gradient Descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted values.

        Updates:
            The model's coefficients and bias based on the stochastic gradient descent step.
        """
        # Randomly choose one data point for the update
        index = random.choice(range(len(features)))
        point = np.array([features[index, :]])
        predict = predicted[index]
        label = labels[index]

        # Update coefficients
        update_coefficients = np.dot(point.T, predict - label) * self.learning_rate
        self.coefficients -= update_coefficients

        # Update bias
        update_bias = (predict - label) * self.learning_rate
        self.bias -= update_bias

    def minibatch_gradient_descent(self, features, labels, predicted):
        """
        Perform a single step of Mini-Batch Gradient Descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted values.

        Updates:
            The model's coefficients and bias based on the mini-batch gradient descent step.
        """
        # Choose a proportion of data to do the calculation
        num_points = int(len(features) * self.proportion)

        start_index = random.choice(range(len(features) - num_points))
        stop_index = start_index + num_points

        points = features[start_index:stop_index, :]
        p_labels = labels[start_index:stop_index]
        p_predicted = predicted[start_index:stop_index]

        # Calculate the gradients for the mini-batch
        d_coefficients = np.dot(points.T, p_predicted - p_labels) / num_points
        d_bias = np.mean(p_predicted - p_labels)

        # Update coefficients and bias
        self.coefficients -= d_coefficients
        self.bias -= d_bias

    def norm_equation(self, features, labels):
        """
        Calculate coefficients using the Normal Equation.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.

        Returns:
            tuple: Tuple containing the coefficients (numpy.ndarray) and bias (float).
        """
        if self.bias is not None:
            # Add a column of ones for the bias term
            x = np.hstack((np.ones((len(features), 1)), features))
            bias = self.bias
        else:
            # No bias term
            x = features
            bias = 0

        if np.linalg.cond(x) < 1 / sys.float_info.epsilon:
            x_t_x_inverse = np.linalg.inv(np.dot(x.T, x))
            x_t_y = np.dot(x.T, labels)
            coefficients = np.dot(x_t_x_inverse, x_t_y)
        else:
            raise ValueError("The matrix provided in this calculation is not reversible,"
                             " please choose another method to train your model!")

        return coefficients, bias

    def ridge_regression(self, features, labels, alpha):

        """
        Calculate coefficients using Ridge Regression.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            alpha (float): Regularization parameter.

        Returns:
            tuple: Tuple containing the coefficients (numpy.ndarray) and bias (float).
        """
        if self.bias:
            x = np.hstack((np.ones((len(features), self.bias)), features))  # Add a column of bias for the bias term
            bias = self.bias
        else:
            x = np.hstack((np.ones((len(features), 1)), features))  # Add a column of bias for the bias term
            bias = 1

        identity_matrix = np.eye(x.shape[1])  # Identity matrix with the same number of columns as X
        x_t_x_regularized_inverse = np.linalg.inv(np.dot(x.T, x) + alpha * identity_matrix)
        x_t_y = np.dot(x.T, labels)
        coefficients = np.dot(x_t_x_regularized_inverse, x_t_y)

        return coefficients, bias




class LogisticRegression:
    """
        A class for training and using a logistic regression model.

        Args:
            train_data (list): Training data in the format: [(label, [feature1, feature2, ..., featureN]), ...].
            coefficients (numpy.ndarray): Initial coefficients for the model. Default is None.
            bias (float): Initial bias for the model. Default is None.
            algorithm (str): The optimization algorithm for training (e.g., 'GD', 'SGD', 'MGD', 'NO').
            learning_rate (float): Learning rate for optimization algorithms. Default is 1e-4.
            max_iter (int): Maximum number of training iterations. Default is 100.
            threshold (float): Threshold for converting probabilities to binary predictions. Default is 0.5.
            con_threshold (float): Convergence threshold for stopping training. Default is 1e-6.
            epsilon (float): A small value to avoid log(0) or log(1) in the cross-entropy loss function. Default is 1e-10.
            proportion (float): Proportion of data to use in mini-batch gradient descent. Default is 0.1.

        Attributes:
            train_data (list): Training data.
            coefficients (numpy.ndarray): Model coefficients.
            bias (float): Model bias.
            algorithm (str): Optimization algorithm.
            learning_rate (float): Learning rate.
            max_iter (int): Maximum number of iterations.
            threshold (float): Threshold for binary predictions.
            con_threshold (float): Convergence threshold.
            epsilon (float): Small value for numerical stability.
            proportion (float): Mini-batch proportion.
            cost (list): List to store the cost at each iteration during training.

        Methods:
            train: Train the logistic regression model using the specified algorithm.
            get_predictions: Calculate predicted probabilities using the model.
            cross_entropy: Calculate the cross-entropy loss for the model.
            gradient_descent: Perform a single step of gradient descent.
            stoch_gradient_descent: Perform a single step of stochastic gradient descent.
            minibatch_gradient_descent: Perform a single step of mini-batch gradient descent.
            newton_method: Perform a single step of the Newton-Raphson method.
            predict: Make binary predictions using the trained model.

        Example:
        ```python
        # Create a LogisticRegression model and train it using gradient descent.
        model = LogisticRegression(train_data, algorithm='GD', max_iter=1000)
        model.train()

        # Make predictions on new data.
        new_data = np.array([...])
        predictions = model.predict(new_data)
        ```
        """

    def __init__(self, train_data, coefficients=None, bias=None, algorithm='GD', learning_rate=1e-4,
                 max_iter=100, threshold=0.5, con_threshold=1e-6, epsilon=1e-10, proportion=0.1):

        self.train_data = train_data
        self.coefficients = coefficients
        self.bias = bias
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.con_threshold = con_threshold
        self.epsilon = epsilon  # Add epsilon to avoid log(0) or log(1) in cross entropy loss function
        self.proportion = proportion
        self.cost = []

    def train(self):
        """
        rain the Logistic Regression model using the specified algorithm.

        If a pre-trained model is available, it loads the model instead of training.

        Note: If a pre-trained model is not available, this method trains the model,
        saves it to a file, and can later be loaded for inference.
        """

        if os.path.exists("logistic_regression_model.pkl"):
            # Load the model from the file
            with open("linear_regression_model.pkl", "rb") as model_file:
                saved_model = joblib.load(model_file)
                self.coefficients = saved_model.coefficients
                self.bias = saved_model.bias

        else:
            algorithms = ['GD', 'SGD', 'MGD', 'NO']

            # Separate independent and dependent variables
            features = np.array([item[1] for item in self.train_data])
            labels = np.array([item[0] for item in self.train_data])

            num_points, num_feats = features.shape
            num_iter = 1  # tracks the number of iterations

            if self.algorithm in algorithms:

                # initialize weights and bias term
                if not self.coefficients:
                    self.coefficients = np.full(shape=num_feats, fill_value=float(0))
                if not self.bias:
                    self.bias = float(0)

                for _ in range(self.max_iter):

                    predicted = self.get_predictions(features=features, coefficients=self.coefficients, bias=self.bias)

                    self.cost.append(self.cross_entropy(predicted=predicted, labels=labels))

                    pre_coefficients = self.coefficients
                    pre_bias = self.bias

                    if self.algorithm == 'GD':

                        self.gradient_descent(
                            features=features,
                            labels=labels,
                            predicted=predicted,
                            num_points=num_points
                        )

                    elif self.algorithm == 'SGD':

                        self.stoch_gradient_descent(
                            features=features,
                            labels=labels,
                            predicted=predicted
                        )

                    elif self.algorithm == 'MGD':

                        self.minibatch_gradient_descent(
                            features=features,
                            labels=labels,
                            predicted=predicted
                        )

                    elif self.algorithm == 'NO':

                        self.newton_method(
                            features=features,
                            labels=labels,
                            predicted=predicted
                        )

                    w_diffs = self.coefficients - pre_coefficients
                    bias_diff = self.bias - pre_bias

                    if all(w_diffs) < self.threshold and bias_diff < self.threshold:
                        print(f"After {num_iter} iterations the model is trained!")
                        break

                    else:
                        num_iter += 1

            else:
                raise ValueError("Please provide a valid algorithm. Available options are: 'GD','SGD',"
                                 "'MGD', 'NO' and 'AG'.")

            with open("linear_regression_model.pkl", "wb") as model_file:
                joblib.dump(self, model_file)

    def get_predictions(self, features, coefficients, bias):
        """
        calculate predicted probabilities using the logistic regression model.

        Args:
            features (numpy.ndarray): Input features.
            coefficients (numpy.ndarray): Model coefficients.
            bias (float): Model bias.

        Returns:
                numpy.ndarray: Predicted probabilities.
        """

        x = np.dot(features, coefficients) + bias
        predicted = 1 / (1 + np.exp(-x))

        return predicted

    def cross_entropy(self, predicted, labels):
        """
        Calculate the cross-entropy loss for the logistic regression model.

        Args:
            predicted (numpy.ndarray): Predicted probabilities.
            labels (numpy.ndarray): Actual labels.

        Returns:
            float: Cross-entropy loss.
        """

        predicted = np.clip(predicted, self.epsilon, 1 - self.epsilon)

        cost = - (labels * np.log(predicted) + (1 - labels) * np.log(1 - predicted))

        mean_cost = np.mean(cost)

        return mean_cost

    def gradient_descent(self, features, labels, predicted, num_points):
        """
        Perform a single step of gradient descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted probabilities.
            num_points (int): Number of data points.

        Updates:
            The model's coefficients and bias based on the gradient descent step.
        """

        d_coefficients = (1 / num_points) * np.dot(features.T, (predicted - labels))
        d_bias = (1 / num_points) * np.sum(predicted - labels)

        self.coefficients -= self.learning_rate * d_coefficients
        self.bias -= self.learning_rate * d_bias

    def stoch_gradient_descent(self, features, labels, predicted):
        """
        Perform a single step of stochastic gradient descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted probabilities.

        Updates:
            The model's coefficients and bias based on the stochastic gradient descent step.
        """

        # Randomly choose one data point for the update

        index = random.choice(range(len(features)))
        point = np.array([features[index, :]])
        predict = predicted[index]
        label = labels[index]

        # Update coefficients
        update_coefficients = np.dot(point.T, predict - label) * self.learning_rate
        self.coefficients -= update_coefficients

        # Update bias
        update_bias = (predict - label) * self.learning_rate
        self.bias -= update_bias

    def minibatch_gradient_descent(self, features, labels, predicted):
        """
        Perform a single step of Mini-Batch Gradient Descent.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted probabilities.

        Updates:
            The model's coefficients and bias based on the mini-batch gradient descent step.
        """

        # Choose a proportion of data to do the calculation
        num_points = int(len(features) * self.proportion)

        start_index = random.choice(range(len(features) - num_points))
        stop_index = start_index + num_points

        points = features[start_index:stop_index, :]
        p_labels = labels[start_index:stop_index]
        p_predicted = predicted[start_index:stop_index]

        # Calculate the gradients for the mini-batch
        d_coefficients = np.dot(points.T, p_predicted - p_labels) / num_points
        d_bias = np.mean(p_predicted - p_labels)

        # Update coefficients and bias
        self.coefficients -= d_coefficients
        self.bias -= d_bias

    def newton_method(self, features, labels, predicted):
        """
        Perform a single step of the Newton-Raphson method.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.
            predicted (numpy.ndarray): Predicted probabilities.

        Updates:
            The model's coefficients and bias based on the Newton-Raphson method.
        """

        differences = labels - predicted

        gradient = (-1 / len(labels)) * np.dot(features.T, differences)

        num_weights = len(self.coefficients)
        hessian_matrix = np.zeros((num_weights, num_weights))

        for i in range(num_weights):
            for j in range(num_weights):
                # Calculate the second derivative of the likelihood function with respect to coefficients i and j
                second_derivative = np.mean(features[:, i] * features[:, j] * predicted * (1 - predicted))
                hessian_matrix[i][j] = second_derivative

        invert_hessian_matrix = np.linalg.inv(hessian_matrix)

        update = np.dot(invert_hessian_matrix, gradient)

        self.coefficients -= update
        self.bias -= update[0]  # Update bias term by the first element of the update vector

    def predict(self, data):
        """
        Make binary predictions using the logistic regression model.

        Args:
            data (numpy.ndarray): Input data for making predictions.

        Returns:
            numpy.ndarray: Binary predictions (0 or 1).

        Raises:
            ValueError: If the model has not been trained.
        """
        if self.coefficients and self.bias:

            x = np.dot(data, self.coefficients) + self.bias
            y_pre = 1 / (1 + np.exp(-x))
            predicted = (y_pre > self.threshold).astype(int)
            return predicted
        else:
            raise ValueError("Please first train the model, after that you can use the predict function")


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



