import os
import joblib
from machine_learning.linear_algebra import intro_numpy as np
import random


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

