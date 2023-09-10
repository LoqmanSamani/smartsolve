class SupportVectorMachines:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    points: [[x11,x12,...,x1n],y2,[x21,x22,x2n],...,[xm1,xm2,...,xmn]] without labels
    """

    def __init__(self, training_data, validation_data=None, optimization_algorithm='GD', kernel='linear', learning_rate=1e-4, lam=1e-4, max_iteration=1000,
                 threshold=1e-5, regularization_parameter=1, epsilon=1e-4, sigma=-0.1):

        self.training_data = training_data
        self.validation_data = validation_data
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

        import numpy as np
        from random import uniform

        # training data
        points = np.array(item[1] for item in self.training_data)
        labels = np.array([item[0] for item in self.training_data])

        # validating data
        v_points = np.array(item[1] for item in self.validation_data)
        v_labels = np.array(item[0] for item in self.validation_data)


        if self.optimization_algorithm == self.algorithms[0]:


            # initialize weights and bias
            self.weights = np.array([uniform(-0.01, 0.01) for _ in range(points.shape[1])])
            self.bias = 0

            num_iter_count = 0

            # train
            for j in range(self.max_iteration):

                weights = self.weights

                for i, point in enumerate(points):
                    error = labels[i] * (np.dot(point, self.weights)) + self.bias

                    d_weights, d_bias = self.der_hinge_loss(error, point, labels[i])

                    self.weights -= self.learning_rate * d_weights
                    self.bias -= self.learning_rate * d_bias
                    num_iter_count += 1

                # stop the training process if the threshold condition is true
                w_diff = abs(weights - self.weights)
                max_diff = np.max(w_diff)

                if max_diff < self.threshold:

                    break

                else:

                    weights = self.weights

            predicted = np.array([self.predict(point) for point in v_points])

            accuracy = self.accuracy(predicted, v_labels)

            full_iter = f"After full number of iteration {self.max_iteration} the accuracy of the trained model based on the validation data is {accuracy} percent."
            not_full_iter = f"After {num_iter_count} iteration the train process is stopped and the accuracy of the model based on validation data is {accuracy} percent."

            if num_iter_count == self.max_iteration:

                return full_iter, self.weights, self.bias

            else:

                return not_full_iter, self.weights, self.bias




        elif self.optimization_algorithm == self.algorithms[1]:

            num_points, num_features = self.training_data.shape

            alphas = np.zeros(num_points)

            num_iter_count_1 = 0

            for _ in range(self.max_iteration):

                alphas_pre = alphas

                for j in range(num_points):

                    i = self.random_num(0, num_points, j)

                    k_ij = self.kernel(points[i], points[i]) + self.kernel(points[j], points[j]) - 2 * self.kernel(points[i], points[j])

                    if k_ij <= 0:
                        continue

                    alpha_pj, alpha_pi = alphas[j], alphas[i]
                    l, h = self.compute_l_h(self.regularization_parameter, alpha_pj, alpha_pi, labels[j], labels[i])

                    self.weights = self.weight_calculator(points[j], labels[j], alphas[j])
                    self.bias = self.bias_calculator(points[j], labels[j], self.weights)

                    prediction_error_i = self.prediction_error(points[i], points[i], self.weights, self.bias)
                    prediction_error_j = self.prediction_error(points[j], points[j], self.weights, self.bias)

                    alphas[j] = alpha_pj + float(labels[j] * (prediction_error_i - prediction_error_j)) / k_ij

                    alphas[j] = np.max(alphas[j], l)
                    alphas[j] = np.min(alphas[i], h)

                    alphas[i] = alpha_pi + labels[i] * labels[j] * (alpha_pj - alphas[j])

                alphas_diff = np.linalg.norm(alphas - alphas_pre)

                if alphas_diff < self.epsilon:

                    num_iter_count_1 += 1

                    break

            predicted = np.array([self.predict(point) for point in v_points])

            accuracy = self.accuracy(predicted, v_labels)

            full_iter1 = f"After full number of iteration {self.max_iteration} the accuracy of the trained model based on the validation data is {accuracy} percent."
            not_full_iter1 = f"After {num_iter_count_1} iteration the train process is stopped and the accuracy of the model based on validation data is {accuracy} percent."

            if num_iter_count_1 == self.max_iteration:

                return full_iter1, self.weights, self.bias

            else:

                return not_full_iter1, self.weights, self.bias




        else:

            return 'Please inter a valid optimization_algorithm. The available algorithms are SGD(Stochastic Gradient Descent) & (Sequential Minimal Optimization)SMO'





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








    def predict(self, point):
        """
        The input of predict must be just one point in the form of a numpy array
        """

        import numpy as np

        predict = np.sign(np.dot(point, self.weights) + self.bias)

        return predict


    def accuracy(self, predicted, labels):

        """
        Both predicted and labels must be numpy arrays
        """

        import numpy as np

        correct_predicted = np.sum([1 if predict == label else 0 for predict, label in zip(predicted, labels)])
        accuracy = (correct_predicted / len(labels)) * 100

        return accuracy





































