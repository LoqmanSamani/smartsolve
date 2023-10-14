class LogisticRegression:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    """

    def __init__(self, training_data, validation_data=None, intercept=0, threshold=0.5, algorithm='GradientDescent', num_iteration=100):

        self.training_data = training_data
        self.validation_data = validation_data
        self.intercept = intercept
        self.threshold = threshold
        self.algorithm = algorithm
        self.num_iteration = num_iteration
        self.coefficients = None




    def __call__(self):

        if self.algorithm == 'RMSprop':
            return self.root_mean_square_propagation()


        elif self.algorithm == 'GradientDescent':
            return self.gradient_descent()


        elif self.algorithm == 'NewtonOptimization':
            return self.newton_method()


        elif self.algorithm == 'Adagrad':
            return self.adaprive_gradient_algorithm()


        else:

            return ("Please provide a valid algorithm. Available options are: GradientDescent,"
                    " RMSprop, NewtonOptimization and Adagrad.")




    # TODO: you should define a train function which calls other functions

    def train(self):
        pass


    def predict(self, data):

        import numpy as np


        features = np.array([item[1] for item in self.training_data])

        coefficients = self.gradient_descent()

        coefficients_row = coefficients.reshape(1, -1)

        power = self.intercept + np.dot(coefficients_row, features.T)

        calculated_predictions = 1 / (1 + np.exp(-power))

        predicted = np.array([1 if i > self.threshold else 0 for i in calculated_predictions])



        return predicted







    def cross_entropy_loss(self):

        """
        Cross-entropy loss, or log loss, measures the performance
        of a classification model whose output is a probability
        value between 0 and 1.

        log loss = -y * log(p) - (1 - y) * log(1 - p)
        where, p is the predicted probability of the positive class,
        y is the actual label.

        """
        import numpy as np



        targets = np.array([item[0] for item in self.training_data])

        predicted = self.get_predictions()

        epsilon = 1e-10 # Add epsilon to avoid log(0) or log(1)

        predicted = np.clip(predicted, epsilon, 1 - epsilon)

        logloss = - (targets * np.log(predicted) + (1 - targets) * np.log(1 - predicted))

        mean_logloss = np.mean(logloss)

        return mean_logloss








    def get_validation_predictions(self, features, coefficients):

        import numpy as np


        coefficients_row = coefficients.reshape(1, -1)

        power = self.intercept + np.dot(coefficients_row, features.T)

        calculated_predictions = 1 / (1 + np.exp(-power))

        predicted = np.array([1 if i > self.threshold else 0 for i in calculated_predictions])

        return predicted









    def gradient_descent(self, learning_rate = 0.001, num_iterations = 100, initial_coefficients = 0, convergence_threshold=1e-6):


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

            mean_logloss = self.cross_entropy_loss()

            coefficients1 = [coefficient - (learning_rate* np.gradient(mean_logloss, coefficient)) for coefficient in coefficients]

            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients

            if coefficients_diff < convergence_threshold:

                print(f"Converged after {i + 1} iterations")

                break

            if self.validation_data:

                validation_targets = np.array([item[0] for item in self.validation_data])
                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.get_validation_predictions(validation_features, coefficients)

                num_accurated = sum([1 if item[0] == item[1] else 0 for item in zip(validation_targets, validation_predictions)])

                accuracy = (num_accurated / len(validation_targets)) * 100

                print(f"Iteration {i + 1}: Validation accuracy = {accuracy} %")


            coefficients = coefficients1


        return coefficients






    def newton_method(self, num_iterations = 100, initial_coefficients = 0):

        import numpy as np

        targets = np.array([item[0] for item in self.training_data])

        features = np.array([item[1] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        for i in range(num_iterations):

            mean_logloss = self.cross_entropy_loss()

            predicted = self.get_predictions()

            differences = targets - predicted

            gradient = (-1 / len(targets)) * np.sum([difference * feature for difference, feature in zip(differences, features)], axis=0)

            hessian_matrix = np.zeros((len(coefficients), len(coefficients)))

            for i in range(len(coefficients)):
                for j in range(len(coefficients)):
                    # Calculate the second derivative of the gradient with respect to coefficients i and j
                    second_derivative = np.mean(features[:, i] * features[:, j] * predicted * (1 - predicted))
                    hessian_matrix[i][j] = second_derivative


            invert_hessian_matrix = np.linalg.inv(hessian_matrix)

            coefficients -= np.dot(invert_hessian_matrix, gradient)

        return coefficients








    def root_mean_square_propagation(self,  learning_rate = 0.001, decay_rate = 0.9, num_iterations = 200, initial_coefficients = 0,
                                     epsilon = 1e-8, convergence_threshold=1e-6):

        import numpy as np



        features = np.array([item[1] for item in self.training_data])

        targets = np.array([item[0] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        squared_gradient_move = np.zeros_like(coefficients)

        for i in range(num_iterations):

            mean_logloss = self.cross_entropy_loss()

            predicted = self.get_predictions()

            differences = targets - predicted

            gradient = (-1 / len(targets)) * np.sum([difference * feature for difference, feature in zip(differences, features)], axis=0)

            squared_gradient = np.power(gradient, 2)

            squared_gradient_move = (decay_rate * squared_gradient_move) + ((1 - decay_rate) * squared_gradient)

            rms = np.sqrt(squared_gradient + epsilon)

            coefficients1 = [coefficinet - (learning_rate / rms) * gradient for coefficient in coefficients]



            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients

            if coefficients_diff < convergence_threshold:

                print(f"Converged after {i + 1} iterations")

                break



            if self.validation_data:

                validation_targets = np.array([item[0] for item in self.validation_data])

                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.get_validation_predictions(validation_features, coefficients)

                num_accurated = sum([1 if item[0] == item[1] else 0 for item in zip(validation_targets, validation_predictions)])

                accuracy = (num_accurated / len(validation_targets)) * 100

                print(f"Iteration {i + 1}: Validation accuracy = {accuracy} %")



            coefficients = coefficients1



        return coefficients








    def adaptive_gradient_algorithm(self, learning_rate = 0.001, num_iterations = 200, initial_coefficients = 0,
                                    convergence_threshold=1e-6, epsilon = 1e-8):

        import numpy as np


        features = np.array([item[1] for item in self.training_data])

        targets = np.array([item[0] for item in self.training_data])

        coefficients = np.full(len(self.training_data[0][1]), initial_coefficients)

        learning_rates = np.full(len(coefficients), learning_rate)

        accumulated_gradients = np.full(len(coefficients), epsilon)  # Initialize with epsilon to avoid division by zero



        for i in range(num_iterations):


            mean_logloss = self.cross_entropy_loss()

            predicted = self.get_predictions()

            differences = targets - predicted

            gradients = (-1 / len(targets)) * np.sum([difference * feature for difference, feature in zip(differences, features)], axis=0)

            squared_gradients = np.power(gradients, 2)

            accumulated_gradients += squared_gradients

            learning_rates = learning_rates / np.sqrt(accumulated_gradients + epsilon)

            coefficients1 = coefficients - (learning_rates * gradients)

            coefficients_diff = np.linalg.norm(coefficients1 - coefficients)  # Calculate change in coefficients


            if coefficients_diff < convergence_threshold:


                print(f"Converged after {i + 1} iterations")

                break



            if self.validation_data:


                validation_targets = np.array([item[0] for item in self.validation_data])

                validation_features = np.array([item[1] for item in self.validation_data])

                validation_predictions = self.get_validation_predictions(validation_features, coefficients)

                num_accurated = sum([1 if item[0] == item[1] else 0 for item in zip(validation_targets, validation_predictions)])

                accuracy = (num_accurated / len(validation_targets)) * 100

                print(f"Iteration {i + 1}: Validation accuracy = {accuracy} %")



            coefficients = coefficients1



        return coefficients
































