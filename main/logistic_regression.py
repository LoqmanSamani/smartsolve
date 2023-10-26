import numpy as np


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

    def mean_squared_errot(self, labels, predicted):
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

            mse = self.mean_squared_errot(labels, np.dot(features, coefficients))
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


