
import numpy as np
from scipy.special import softmax
import pickle




class NeuralNetwork:
    def __init__(self, training_data, validation_data=None, input_layer_neurons=None, output_layer_neurons=None,
                 num_hidden_layers=1, hidden_layer_neurons=None, max_iteration=100, learning_rate=1e-4,
                 converge_threshold=1e-6, cost_threshold=1e-3, cost_algorithm='Regression',
                 activation_algorithm='Sigmoid', alpha=1e-5):

        """
        Initialize a feedforward neural network.

        :param training_data: The training data as a list of tuples (label, features).
        :param validation_data: Optional validation data for monitoring model performance during training.
        :param input_layer_neurons: Number of neurons in the input layer.
        :param output_layer_neurons: Number of neurons in the output layer.
        :param num_hidden_layers: Number of hidden layers.
        :param hidden_layer_neurons: Number of neurons in each hidden layer (as a list).
        :param max_iteration: Maximum number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param converge_threshold: Threshold for detecting convergence.
        :param cost_threshold: Threshold for stopping based on cost function change.
        :param cost_algorithm: Cost function ('Regression' or 'Classification').
        :param activation_algorithm: Activation function for hidden layers.
        :param alpha: Alpha parameter for the ELU activation function.
        """


        self.training_data = training_data
        self.validation_data = validation_data


        if input_layer_neurons:
            self.input_layer_neurons = input_layer_neurons  # number of neurons in the input layer
        else:
            self.input_layer_neurons = self.input_layer_neurons()


        if output_layer_neurons:
            self.output_layer_neurons = output_layer_neurons  # number of neurons in the output layer
        else:
            self.output_layer_neurons = self.output_layer_neurons()


        if hidden_layer_neurons:
            self.hidden_layer_neurons = hidden_layer_neurons  # number of neurons in each hidden layer,
            # a list or ndarray of integers, which defines the number of neurons in each layer
        else:
            self.hidden_layer_neurons = self.hidden_layer_neurons()


        self.max_iteration = max_iteration
        self.num_hidden_layers = num_hidden_layers  # number of hidden layers
        self.learning_rate = learning_rate
        self.converge_threshold = converge_threshold  # A threshold to stop training, if the parameters do not change anymore
        self.cost_threshold = cost_threshold # A threshold to stop training, if cost function's output is smaller than it.
        self.cost_algorithm = cost_algorithm
        self.activation_algorithm = activation_algorithm
        self.alpha = alpha  # An activation parameter used in Exponential Linear Unit (ELU) algorithm


        self.weights = self.initialize_weight(
            input_size=self.input_layer_neurons,
            hidden_layer_sizes=list(self.hidden_layer_neurons),
            output_size=self.output_layer_neurons
            )


        self.biases = self.initialze_bias(
            input_size=self.input_layer_neurons,
            hidden_layer_sizes=list(self.hidden_layer_neurons),
            output_size=self.output_layer_neurons
        )


        self.cost = []  # A list to store the values calculated from the cost function in each iteration





    def input_layer_neurons(self):
        if len(self.training_data[0]) > 1:
            num_features = len(self.training_data[0][1])
        else:
            raise ValueError("Please define the number of neurons in the input layer(input_layer_neurons=?)")
        return num_features




    def output_layer_neurons(self):
        if len(self.training_data[0]) > 1:
            num_labels = len(set([self.training_data[i][0] for i in range(len(self.training_data))]))

        else:
            raise ValueError("Please define the number of output layers(output_layer_neurons=?)")
        return num_labels





    def hidden_layer_neurons(self):
        if len(self.training_data[0]) > 1:
            num_features = len(self.training_data[0][1])
        else:
            raise ValueError("Please define the number of neurons in each hidden layer(input_layer_neurons=?)."
                             " It should be a list, which contains the number of neurons in each hidden layer.")
        return num_features






    def initialize_weight(self, input_size, hidden_layer_sizes, output_size):
        """
        Initialize weights for the neural network layers.

        :param input_size: Number of neurons in the input layer.
        :param hidden_layer_sizes: List of integers, specifying the number of neurons in each hidden layer.
        :param output_size: Number of neurons in the output layer.
        :return: List of weight matrices for each layer.
        """

        weights = []
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        for layer in range(1, len(layer_sizes)):
            # Initialize weights for the current layer
            # Use random initialization (e.g., Xavier/Glorot initialization)
            # Scale the initial weights by a factor to control the variance
            scale = np.sqrt(2.0 / (layer_sizes[layer - 1] + layer_sizes[layer]))
            weights.append(np.random.randn(layer_sizes[layer], layer_sizes[layer - 1]) * scale)

        return weights







    def initialze_bias(self, input_size, hidden_layer_sizes, output_size):
        """
        Initialize biases for the neural network layers.

        :param input_size: Number of neurons in the input layer.
        :param hidden_layer_sizes: List of integers, specifying the number of neurons in each hidden layer.
        :param output_size: Number of neurons in the output layer.
        :return: List of bias vectors for each layer.
        """

        biases = []
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # Initialize biases for the current layer with zeros (a small constant)
        for layer in range(1, len(layer_sizes)):
            biases.append(np.zeros((layer_sizes[layer], 1)))

        return biases






    def forward_propagation(self, data, weights, biases):
        """
        Perform forward propagation through the neural network.

        :param data: Input data.
        :param weights: List of weight matrices for each layer.
        :param biases: List of bias vectors for each layer.
        :return: Predicted output and a list of activated data for each layer.
        """

        activated_data = [data]  # Initialize with the input data

        for i in range(self.num_hidden_layers):
            weighted_sum = self.linear_transformation(activated_data[-1], weights[i], biases[i])
            activated = self.activation_function(data=weighted_sum, algorithm=self.activation_algorithm)
            activated_data.append(activated)

        predicted = activated_data[-1]

        return predicted, activated_data






    def linear_transformation(self, activated_data, weights, biases):
        """
        Perform the linear transformation of data for a neural network layer.

        :param activated_data: Activated data from the previous layer.
        :type activated_data: numpy.ndarray
        :param weights: Weight matrix for the current layer.
        :type weights: numpy.ndarray
        :param biases: Bias vector for the current layer.
        :type biases: numpy.ndarray
        :return: Weighted sum of the input data with bias.
        :rtype: numpy.ndarray
        """
        weighted_sum = np.dot(weights, activated_data) + biases
        return weighted_sum






    def activation_function(self, data, algorithm):
        """
        Apply an activation function to the input data.

        :param data: Input data to be activated.
        :type data: numpy.ndarray
        :param algorithm: Activation algorithm to be applied ('Sigmoid', 'Tanh', 'ReLU', 'ELU', 'Softmax').
        :type algorithm: str
        :return: Activated data.
        :rtype: numpy.ndarray
        """

        if algorithm == 'Sigmoid':
            data = self.sigmoid_activation(data=data)
        elif algorithm == 'Tanh':
            data = self.tanh_activation(data=data)
        elif algorithm == 'ReLU':
            data = self.relu_activation(data=data)
        elif algorithm == 'ELU':
            data = self.elu_activation(data=data, alpha=self.alpha)
        elif algorithm == 'Softmax':
            data = self.softmax_activation(predicted=data)

        return data





    def sigmoid_activation(self, data):

        activated_data = []
        for point in data:
            new_point = []
            for feature in point:
                new_feature = 1 / (1 + np.exp(-feature))
                new_point.append(new_feature)
            activated_data.append(new_point)

        return activated_data





    def tanh_activation(self, data):
        activated_data = []
        for point in data:
            new_point = []
            for feature in point:
                new_feature = (np.exp(feature) - np.exp(-feature)) / (np.exp(feature) + np.exp(feature))
                new_point.append(new_feature)
            activated_data.append(new_point)

        return activated_data







    def relu_activation(self, data):

        activated_data = []
        for point in data:
            new_point = []
            for feature in point:
                new_feature = max(0, feature)
                new_point.append(new_feature)
            activated_data.append(new_point)

        return activated_data







    def elu_activation(self, data, alpha):

        activated_data = []
        for point in data:
            new_point = []
            for feature in point:
                if feature >= 0:
                    new_feature = feature
                else:
                    new_feature = alpha * (np.exp(feature) - 1)
                new_point.append(new_feature)
            activated_data.append(new_point)

        return activated_data







    def softmax_activation(self, predicted):

        exp_predicted = [np.exp(predict) for predict in predicted]
        normalization_constant = sum(exp_predicted)
        normalized_predicted = [exp_predict / normalization_constant for exp_predict in exp_predicted]

        return normalized_predicted







    def cost_function(self, predicted, actual):

        if self.cost_algorithm == 'Regression':
            cost = self.mean_squared_error(predicted=predicted, actual=actual)

        elif self.cost_algorithm == 'Classification':
            cost = self.cross_entropy(predicted=predicted, actual=actual)

        else:
            raise ValueError("Please define a valid algorithm (cost_algorithm)! "
                             "If the model is a regression model: cost_algorithm='Regression' or the"
                             " model is a classification: cost_algorithm='Classification'!")

        return cost







    def cross_entropy(self, predicted, actual):

        epsilon = 1e-10 # Add epsilon to avoid log(0) or log(1)

        predicted = np.clip(predicted, epsilon, 1 - epsilon)

        log_loss = - (actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

        mean_log_loss = np.mean(log_loss)

        return mean_log_loss







    def mean_squared_error(self, predicted, actual):

        mse = np.mean(np.power(actual - predicted, 2))
        return mse





    def back_propagation(self, predicted, actual, weights, biases, activated_data):
        """
        Perform backpropagation to update weights and biases.

        :param predicted: Predicted output.
        :param actual: Actual output.
        :param weights: List of weight matrices for each layer.
        :param biases: List of bias vectors for each layer.
        :param activated_data: List of activated data for each layer.
        :return: Updated weights and biases.
        """

        # Initialize the list to store delta values for each layer
        deltas = []

        # Compute the initial delta for the output layer
        delta_output = predicted - actual
        deltas.append(delta_output)

        # Back_propagate the delta values through hidden layers
        for i in range(self.num_hidden_layers - 1, -1, -1):

            # Calculate the delta for the current layer based on the delta from the next layer
            delta_hidden = np.dot(np.transpose(weights[i + 1]), deltas[-1]) * self.activation_derivative(
                predicted=predicted,
                algorithm=self.activation_algorithm
            )

            deltas.append(delta_hidden)

        # Reverse the order of deltas to match layer indices
        deltas = list(reversed(deltas))

        # Update weights and biases using computed deltas
        for i in range(self.num_hidden_layers, -1, -1):
            d_weight = np.dot(deltas[i], np.transpose(activated_data[i]))
            d_bias = deltas[i].mean(axis=1, keepdims=True)

            # Update weights and biases for the current layer
            weights[i] -= self.learning_rate * d_weight
            biases[i] -= self.learning_rate * d_bias

        return weights, biases






    def activation_derivative(self, predicted, algorithm):
        """
        Calculate the derivative of the activation function for a given algorithm.

        :param predicted: The predicted values.
        :param algorithm: Activation algorithm ('Sigmoid', 'Tanh', 'ReLU', 'ELU', or 'Softmax').
        :return: Derivative of the activation function.
        """
        if algorithm == 'Sigmoid':
            return self.sigmoid_derivative(predicted=predicted)
        elif algorithm == 'Tanh':
            return self.tanh_derivative(predicted=predicted)
        elif algorithm == 'ReLU':
            return self.relu_derivative(predicted=predicted)
        elif algorithm == 'ELU':
            return self.elu_derivative(predicted=predicted, alpha=self.alpha)
        elif algorithm == 'Softmax':
            return self.softmax_derivative(predicted=predicted)

        raise NotImplementedError("Derivative of softmax is not implemented here.")






    def sigmoid_derivative(self, predicted):
        return predicted * (1 - predicted)




    def tanh_derivative(self, predicted):
        return 1 - predicted ** 2




    def relu_derivative(self, predicted):
        return np.where(predicted > 0, 1, 0)






    def elu_derivative(self, predicted, alpha):
        return np.where(predicted > 0, 1, alpha * np.exp(predicted))





    def softmax_derivative(self, predicted):
        # Calculate the softmax probabilities
        probabilities = softmax(predicted)

        # Calculate the derivative of softmax
        # The derivative matrix is calculated using an outer product
        derivative = np.diag(probabilities) - np.outer(probabilities, probabilities)

        return derivative







    def train(self):
        """
        Train the neural network using the provided training data.

        :return: A message indicating training completion.
        """

        num_iter = 0

        for i in range(self.max_iteration):

            points = np.array([point[1] for point in self.training_data])
            labels = [label[0] for label in self.training_data]
            prev_weights = self.weights
            prev_biases = self.biases

            predicted, activated_data = self.forward_propagation(
                data=points,
                weights=self.weights,
                biases=self.biases
            )

            cost = self.cost_function(
                predicted=predicted,
                actual=labels
            )

            if self.cost:
                if abs(self.cost[-1] - cost) < self.cost_threshold:
                    print(f"The training process is stopped after {num_iter} iterations,"
                          " because the change in the output of cost function is smaller "
                          "than the cost_threshold({self.cost_threshold})!"
                          " If you want the training continued, please reduce the cost_threshold!")
                    break
            else:
                self.cost.append(cost)

            self.weights, self.biases = self.back_propagation(
                predicted=predicted,
                actual=labels,
                weights=self.weights,
                biases=self.biases,
                activated_data=activated_data
            )

            # check convergence
            not_converged = False
            for j in range(len(self.weights)):
                if (np.max(self.weights[j] - prev_weights[j]) > self.converge_threshold
                        or np.max(self.biases[j] - prev_biases[j]) > self.converge_threshold):

                    not_converged = True
            if not not_converged:
                print(f"The training process is stopped after {num_iter} iterations,"
                      " because the change in biases and weights is smaller than the"
                      " converge_threshold({self.converge_threshold})! If you want"
                      " the training continued, please reduce the converge_threshold!")
                break

            num_iter += 1

            # TODO: using validation data check the performance of the model

        return "The model has been after full iteration trained!"






    def predict(self, data):
        """
        Make predictions using the trained neural network.

        :param data: Input data for prediction.
        :return: Predicted output.
        """

        predicted, _ = self.forward_propagation(
            data=data,
            weights=self.weights,
            biases=self.biases
        )

        return predicted






    def save_model(self, filename):
        """
        Save the trained model to a file.
        :param filename: Name of the file to save the model.
        """
        model_params = {
            'biases': self.biases,
            'weights': self.weights,
        }

        with open(filename, 'wb') as file:
            pickle.dump(model_params, file)






    def load_model(self, filename):
        """
        Load a trained model from a file.
        :param filename: Name of the file containing the saved model.
        :return: None
        """
        with open(filename, 'rb') as file:
            model_params = pickle.load(file)

        # Assuming that model_params is a dictionary containing 'biases' and 'weights'
        if 'biases' in model_params and 'weights' in model_params:
            self.biases = model_params['biases']
            self.weights = model_params['weights']
        else:
            raise ValueError("The loaded model parameters are missing or incorrect.")

