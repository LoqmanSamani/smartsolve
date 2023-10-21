import numpy as np
import random
import os
import joblib


class LinearRegression:

    def __init__(self, train_data, coefficients=None, bias=None, algorithm='GD', learning_rate=0.01,
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
        self.bias = bias

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

        self.coefficients -= d_coefficients * self.learning_rate
        self.bias -= d_bias * self.learning_rate

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

        #  Choose one point to do the calculation
        index = random.choice([i for i in range(len(features))])
        point = np.array([features[index, :]])
        predict = predicted[index]
        label = labels[index]

        d_coefficients = np.dot(point.T, predict - label) * self.learning_rate
        d_bias = predict - label

        self.coefficients -= d_coefficients * self.learning_rate
        self.bias -= d_bias * self.learning_rate

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

        #  Choose a proportion of data to do the calculation
        num_points = int(len(features) * self.proportion)
        start_index = random.choice([i for i in range(len(features) - num_points)])
        stop_index = start_index + num_points
        points = np.array(features[start_index:stop_index, :])
        p_labels = np.array(labels[start_index:stop_index])
        p_predicted = np.array(predicted[start_index:stop_index])

        d_coefficients = np.dot(points.T, p_predicted - p_labels) * self.learning_rate
        d_bias = p_predicted - p_labels

        self.coefficients -= d_coefficients * self.learning_rate
        self.bias -= d_bias * self.learning_rate

    def norm_equation(self, features, labels):

        """
        Calculate coefficients using the Normal Equation.

        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Actual labels.

        Returns:
            tuple: Tuple containing the coefficients (numpy.ndarray) and bias (float).
        """
        if self.bias:
            x = np.hstack((np.ones((len(features), self.bias)), features))  # Add a column of bias for the bias term
            bias = self.bias
        else:
            x = np.hstack((np.ones((len(features), 1)), features))  # Add a column of bias for the bias term
            bias = 1

        x_t_x_inverse = np.linalg.inv(np.dot(x.T, x))
        x_t_y = np.dot(x.T, labels)

        coefficients = np.dot(x_t_x_inverse, x_t_y)

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
    pass


class DecisionTree:
    pass

