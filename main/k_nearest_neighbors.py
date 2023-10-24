import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) algorithm for classification and regression.

    Parameters:
        train_data (list): The training data in the format [(y1,[x11,x12,...,x1n]), ...].
        num_neighbors (int): The number of neighbors to consider (default is 5).
        distance (str): The distance metric to use ('EU', 'MA, 'MI', 'CH', or 'CO').
        algorithm (str): The type of task ('classification' or 'regression').
        p_value (int): The p-value for the Minkowski distance (default is 2).

    Attributes:
        features (numpy.ndarray): The features of the training data.
        labels (numpy.ndarray): The labels of the training data.

    Methods:
        fit(): Fit the model to the training data.
        custom_distance(point1, point2): Calculate the custom distance between two data points.
        euclidean (point1, point2): Calculate the Euclidean distance between two points.
        manhattan(point1, point2): Calculate the Manhattan distance between two points.
        minkowski(point1, point2): Calculate the Minkowski distance between two points.
        chebyshev(point1, point2): Calculate the Chebyshev distance between two points.
        cosine(point1, point2): Calculate the Cosine distance between two points.
        predict(data): Make predictions for a list of data points.
        point_predict(point): Predict the class or value of a single data point.
    """
    def __init__(self, train_data, num_neighbors=5, distance='EU', algorithm='classification', p_value=2):
        self.train_data = train_data
        self.num_neighbors = num_neighbors
        self.distance = distance
        self.algorithm = algorithm
        self.p_value = p_value
        self.features = None
        self.labels = None

    def fit(self):
        """
        Fit the model to the training data.

        Returns:
            None
        """
        self.features = np.array([point[1] for point in self.train_data])
        self.labels = np.array([point[0] for point in self.train_data])

    def custom_distance(self, point1, point2):
        """
        Calculate the custom distance between two data points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The distance between the two points.
        """

        if self.distance == 'EU':
            return self.euclidean(point1=point1, point2=point2)

        elif self.distance == 'MA':
            return self.manhattan(point1=point1, point2=point2)

        elif self.distance == 'MI':
            return self.minkowski(point1=point1, point2=point2)

        elif self.distance == 'CH':
            return self.chebyshev(point1=point1, point2=point2)

        elif self.distance == 'CO':
            return self.cosine(point1=point1, point2=point2)

        else:
            raise ValueError("Invalid distance algorithm. available are: 'EU' (euclidean distance),"
                             " 'MA' (manhattan distance), 'MI'(minkowski distance), 'CH' (chebyshev distance),"
                             "'CO' (cosine distance).")

    def euclidean(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Euclidean distance between the two points.
        """

        distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

        return distance

    def manhattan(self, point1, point2):
        """
        Calculate the Manhattan distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Manhattan distance between the two points.
        """

        distance = np.sum(np.abs(np.array(point1) - np.array(point2)))

        return distance

    def minkowski(self, point1, point2):
        """
        Calculate the Minkowski distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Minkowski distance between the two points.
        """

        distance = (np.sum(np.abs(np.array(point1) - np.array(point2))
                           ** self.p_value)) ** (1 / self.p_value)

        return distance

    def chebyshev(self, point1, point2):
        """
        Calculate the Chebyshev distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Chebyshev distance between the two points.
        """

        distance = np.max(np.abs(np.array(point1) - np.array(point2)))

        return distance

    def cosine(self, point1, point2):
        """
        Calculate the Cosine distance between two points.

        Parameters:
            point1 (list): The first data point.
            point2 (list): The second data point.

        Returns:
            float: The Cosine distance between the two points.
        """

        dot_product = np.dot(point1, point2)
        norm_p = np.linalg.norm(point1)
        norm_q = np.linalg.norm(point2)
        distance = 1.0 - (dot_product / (norm_p * norm_q))
        return distance

    def predict(self, data):
        """
        Make predictions for a list of data points.

        Parameters:
            data (list): A list of data points to make predictions for.

        Returns:
            list: Predictions for each data point.
        """
        predictions = [self.point_predict(point=point) for point in data]
        return predictions

    def point_predict(self, point):
        """
        Predict the class or value of a single data point.

        Parameters:
            point (list): A single data point.

        Returns:
            int or float: The predicted class (for classification) or value (for regression).
        """
        distances = [self.custom_distance(point1=point, point2=train_point) for train_point in self.features]
        k_indices = np.argsort(distances)[:self.num_neighbors]
        k_nearest_labels = [self.labels[i] for i in k_indices]

        if self.algorithm == 'classification':
            most_common = Counter(k_nearest_labels).most_common()
            return most_common[0][0]
        elif self.algorithm == 'regression':
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Invalid algorithm. Supported values: 'classification' or 'regression'")



