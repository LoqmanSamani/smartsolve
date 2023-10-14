import numpy as np


class LinearRegression:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    """
    def __init__(self, training_data, validation_data=None, intercept=0, algorithm ='GradientDescent', num_parts=10):

        self.training_data = training_data
        self.validation_data = validation_data
        self.intercept = intercept
        self.algorithm = algorithm
        self.num_parts = num_parts

        self.coefficients = None




    def __call__(self):


        if self.algorithm == 'GradientDescent':
            return self.gradient_descent()


        elif self.algorithm == 'StochasticGradientDescent':
            return self.stochastic_gradient_descent()


        elif self.algorithm == 'MiniBatchGradientDescent':
            return self.mini_batch_gradient_descent()


        elif self.algorithm == 'NormalEquation':
            return self.normal_equation()


        elif self.algorithm == 'RidgeRegression':
            return self.ridge_regression()

        else:
            return ("Please provide a valid algorithm. Available options are: GradientDescent,"
                    " StochasticGradientDescent, MiniBatchGradientDescent, NormalEquation and RidgeRegression.")




    def train(self):
        pass

    def prediction(self):

        import numpy as np
        import random


        if self.algorithm == 'GradientDescent':
            features = np.array([item[1] for item in self.training_data])
            coefficients = self.gradient_descent()
            coefficients_row = coefficients.reshape(1, -1)
            predicted = self.intercept + np.dot(coefficients_row, features.T)


        elif self.algorithm == 'StochasticGradientDescent':

            features = np.array([item[1] for item in self.training_data])
            targets = np.array([item[0] for item in self.training_data])

            coefficients = self.stochastic_gradient_descent()
            coefficients_row = coefficients.reshape(1, -1)

            maximum = len(targets) - 1
            random_number = random.randint(0, maximum)

            predicted = self.intercept + np.dot(coefficients_row, features[random_number].T)

        elif self.algorithm == 'MiniBatchGradientDescent':


            num_points = int(len(self.training_data) / self.num_parts)
            index = [i * num_points for i in range(self.num_parts)]

            data = [self.training_data[i:i + num_points] for i in index] # Split data into subsets

            for subdata in data:

                features = np.array([item[1] for item in subdata])
                targets  = np.array([item[0] for item in subdata])

                coefficients = self.mini_batch_gradient_descent()

                coefficients_row = coefficients.reshape(1, -1)

                predicted = self.intercept + np.dot(coefficients_row, features.T)

        else:
            return "There is an issue with this try!!!"


        return predicted





    def mean_squared_error(self):
        """
        Calculates MSE (Mean Squared Error):
        MSE = (1/n) * sum((y_actual - y_predicted)^2)
        where n is the number of data points, y_actual is the actual target value,
        and y_predicted is the predicted value from the linear regression model.
        """

        import numpy as np

        targets = np.array([item[0] for item in self.training_data])

        predicted = self.get_predictions()

        mse = np.mean((targets - predicted)**2)

        return mse







    def prediction(self, features, coefficients):

        predicted = np.dot(coefficients, features.T)
        return predicted






    def gradient_descent(self, learning_rate = 0.01, num_iterations = 100, initial_coefficients = 0, convergence_threshold=1e-6):

        """
        Gradient Descent is an optimization algorithm for finding a local minimum
        of a differentiable function. Gradient descent in machine learning is simply
        used to find the values of a function's parameters (coefficients)
        that minimize a cost function as far as possible.
        """

        import numpy as np

        targets = np.array([item[0] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        for i in range(num_iterations):

            mse = self.mean_squared_error()

            coefficients1 = [coefficient - (learning_rate * np.gradient(mse, coefficient)) for coefficient in coefficients]

            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients

            if coefficients_diff < convergence_threshold:

                print(f"Converged after {i + 1} iterations")

                break

            if self.validation_data:

                validation_targets = np.array([item[0] for item in self.validation_data])
                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.predict(validation_features, coefficients)

                validation_mse = np.mean((validation_targets - validation_predictions) ** 2)

                print(f"Iteration {i + 1}: Validation MSE = {validation_mse}")


            coefficients = coefficients1


        return coefficients







    def stochastic_gradient_descent(self, learning_rate = 0.01, num_iterations = 500, initial_coefficients = 0, convergence_threshold=1e-6):

        import numpy as np

        targets = np.array([item[0] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        for i in range(num_iterations):

            mse = self.mean_squared_error()

            coefficients1 = [coefficient - (learning_rate * np.gradient(mse, coefficient)) for coefficient in coefficients]

            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients

            if coefficients_diff < convergence_threshold:
                print(f"Converged after {i + 1} iterations")

                break

            if self.validation_data:
                validation_targets = np.array([item[0] for item in self.validation_data])
                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.predict(validation_features, coefficients)

                validation_mse = np.mean((validation_targets - validation_predictions) ** 2)

                print(f"Iteration {i + 1}: Validation MSE = {validation_mse}")

            coefficients = coefficients1

        return coefficients






    def mini_batch_gradient_descent(self, learning_rate = 0.01, num_iterations = 200, initial_coefficients = 0, convergence_threshold=1e-6):

        import numpy as np

        targets = np.array([item[0] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        for i in range(num_iterations):

            mse = self.mean_squared_error()

            coefficients1 = [coefficient - (learning_rate * np.gradient(mse, coefficient)) for coefficient in coefficients]

            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients

            if coefficients_diff < convergence_threshold:
                print(f"Converged after {i + 1} iterations")

                break

            if self.validation_data:
                validation_targets = np.array([item[0] for item in self.validation_data])
                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.predict(validation_features, coefficients)

                validation_mse = np.mean((validation_targets - validation_predictions) ** 2)

                print(f"Iteration {i + 1}: Validation MSE = {validation_mse}")

            coefficients = coefficients1

        return coefficients







    def normal_equation(self):

        """
        Calculates the optimal coefficients (β) using the
        normal equation for linear regression.

        The normal equation provides a direct way to find
        the optimal coefficients:

        β = (X^T * X)^-1 * X^T * y

        This approach is particularly suitable for small
        datasets with a relatively small number of features.

        Returns:
        coefficients (numpy.ndarray): An array of coefficients
        (including the intercept) that minimize the mean squared error.
    """

        import numpy as np

        targets = np.array([target[0] for target in self.training_data])
        features = np.array([feature[1] for feature in self.training_data])

        X = np.hstack((np.ones((len(features), 1)), features))  # Add a column of ones for the bias term

        X_T_X_inverse = np.linalg.inv(np.dot(X.T, X))

        X_T_y = np.dot(X.T, targets)

        coefficients = np.dot(X_T_X_inverse, X_T_y)

        return coefficients







    def ridge_regression(self, alpha = 0.01):

        """
        Calculates the optimal coefficients (β) using the normal
        equation for Ridge Regression.

        Ridge Regression is a variant of linear regression that
        adds a regularization term

        to the normal equation to prevent overfitting:

        β = (X^T * X + alpha * I)^-1 * X^T * y

        where alpha is the regularization parameter
        and I is the identity matrix.

        This approach is particularly suitable when dealing
        with multicollinearity or overfitting.

        Args:
        alpha (float): The regularization parameter
        controlling the strength of regularization.

        Returns:
        coefficients (numpy.ndarray): An array of coefficients
        (including the intercept) that minimize the mean
        squared error with Ridge regularization.
        """

        import numpy as np

        targets = np.array([target[0] for target in self.training_data])
        features = np.array([feature[1] for feature in self.training_data])

        X = np.hstack((np.ones((len(features), 1)), features))  # Add a column of ones for the bias term

        identity_matrix = np.eye(X.shape[1])  # Identity matrix with the same number of columns as X

        X_T_X_regularized_inverse = np.linalg.inv(np.dot(X.T, X) + alpha * identity_matrix)

        X_T_y = np.dot(X.T, targets)

        coefficients = np.dot(X_T_X_regularized_inverse, X_T_y)

        X_T_X_inverse = np.linalg.inv(np.dot(X.T, X))


        return coefficients







