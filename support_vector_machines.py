class SupportVectorMachines:
    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    points: [[x11,x12,...,x1n],y2,[x21,x22,x2n],...,[xm1,xm2,...,xmn]] without labels
     """

    def __init__(self, training_data, labels, validation_data=None, num_spv=10, learning_rate=0.01, max_iterations=100):
        self.training_data = training_data
        self.labels = labels
        self.validation_data = validation_data
        self.num_spv = num_spv
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.bias = 1
        self.weights = None




    def define_hyperplane(self, weights):

        import numpy as np

        points = np.array([point[1] for point in self.training_data])
        hyperplane = np.dot(weights, points) + self.bias

        return hyperplane



    def class_separator(self):

        import numpy as np

        # separate each data point based on its label
        # (there are supposed to be just two labels(binary data) in training data)
        class1 = [point for point in self.training_data if point[0] == self.labels[0]]
        class2 = [point for point in self.training_data if point[0] == self.labels[1]]

        return np.array(class1), np.array(class2)




    def initializer(self, num_weights):

        import numpy as np

        weights = np.random.randn(num_weights)

        return weights




    def distance_calculator(self, data, weights, bias):

        import numpy as np

        distances = []

        for point in data:

            w_dot_point = np.dot(weights, point[1])
            distance = abs(w_dot_point + bias) / np.linalg.norm(weights)
            distances.append((point[0], distance))

        return np.array(distances)





    def support_vector_finder(self, data):

        import numpy as np

        # finds the nearest points(support vectors)
        euclidean_norms = np.array([np.sqrt(np.sum(p[1] ** 2) for p in data)])

        margins = [(point[0], 2 / point[1]) for point in euclidean_norms]

        sorted_data = sorted(margins, key=lambda x: x[1])

        support_vectors = sorted_data[0: self.num_spv]

        return np.array(support_vectors)





    def weight_optimizer(self, data, alphas):

        import numpy as np

        labels = np.array([point[0] for point in data])

        distances = np.array([point[1] for point in data])

        weights = np.dot(alphas * labels, distances)

        bias = np.mean(labels - np.dot(weights, distances.T))

        return weights, bias





    def train(self):

        import numpy as np

        class1, class2 = self.class_separator()

        num_weights = len(self.training_data[0][1])

        self.weights = self.initializer(num_weights)

        num_samples = len(self.training_data)

        alphas = np.zeros(num_samples) # Initialize Lagrange Multipliers (Alphas) with zero

        for _ in range(self.max_iterations):

            distances1 = self.distance_calculator(class1, self.weights, self.bias)

            distances2 = self.distance_calculator(class2, self.weights, self.bias)

            spv1 = self.support_vector_finder(distances1)

            spv2 = self.support_vector_finder(distances2)

            self.weights, self.bias = self.weight_optimizer(np.vstack((spv1, spv2)), alphas)





    def predict(self, data_point):

        import numpy as np

        if self.weights is None:

            raise ValueError("Model is not trained. Call train() method first.")

        w_dot_point = np.dot(self.weights, data_point)

        return np.sign(w_dot_point + self.bias)

