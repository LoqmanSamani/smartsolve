import numpy as np


class LinearRegression:
    def __init__(self, train_data,  coefficients=None, bias=None, learning_rate=1e-2, max_iter=1000, threshold=1e-8,
                 seed=42, norm='yes'):
        """
        Initialize a Linear Regression model.

        :param train_data: Training data in the form of [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n])
                          ,...,(ym,[xm1,xm2,...,xmn])].
        :param coefficients: Initial coefficients for the model (default is None).
        :param bias: Bias term for the model (default is None).
        :param learning_rate: Learning rate for gradient descent (default is 1e-2).
        :param max_iter: Maximum number of iterations for gradient descent (default is 1000).
        :param threshold: Convergence threshold for gradient descent (default is 1e-8).
        :param seed: Random seed for reproducibility (default is 42).
        :param norm: Normalize features or not ('yes' or 'no', default is 'yes').
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

    def predict(self, data, norm='no'):
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



