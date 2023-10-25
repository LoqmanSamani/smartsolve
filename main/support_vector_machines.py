import numpy as np


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

        calc = np.exp(-np.power(np.linalg.norm(point1 - point2), 2) / (2 * self.sigma**2))

        return calc


    def random_num(self, a, b, z):

        import numpy as np

        lst = list(range(a, z)) + list(range(z+1, b))

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





