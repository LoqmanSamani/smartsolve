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

