import numpy as np


class NaiveBayes:
    def __init__(self, train_data, algorithm='classification'):
        """
        Initializes a NaiveBayes classifier/regressor.

        :param train_data: Input training data in the form of [(y1, [x11, x12, ..., x1n]), (y2, [x21, x22, x2n]), ...].
        :param algorithm: The type of algorithm, either 'classification' or 'regression' (default is 'classification').
        """
        self.train_data = train_data
        self.algorithm = algorithm

        self.classes = None
        self.mean = None  # For storing class means
        self.variance = None  # For storing class variances
        self.prior = None
        self.labels = None

    def train(self):
        """
        Trains the NaiveBayes model with the provided training data.
        """
        features = np.array([point[1] for point in self.train_data])
        labels = np.array([point[0] for point in self.train_data])

        num_points, num_features = features.shape
        self.classes = np.unique(labels)

        self.mean = np.zeros(shape=(len(self.classes), num_features), dtype=np.float64)
        self.variance = np.zeros(shape=(len(self.classes), num_features), dtype=np.float64)
        self.prior = np.zeros(len(self.classes), dtype=np.float64)

        for index, cls in enumerate(self.classes):
            feature_cls = features[labels == cls]
            self.mean[index, :] = feature_cls.mean(axis=0)
            self.variance[index, :] = feature_cls.var(axis=0)
            self.prior[index] = feature_cls.shape[0] / float(num_points)

        self.labels = labels

    def likelihood(self, data, mean, variance):
        """
        Calculates the likelihood of the data given the class mean and variance using Gaussian distribution.

        :param data: The data point for which to calculate the likelihood.
        :param mean: The mean of the class.
        :param variance: The variance of the class.
        :return: The likelihood of the data point.
        """

        epsilon = 1e-4
        coefficient = 1 / np.sqrt(2 * np.pi * variance + epsilon)

        exponent = np.exp(-((data - mean) ** 2 / (2 * variance + epsilon)))
        likelihood = coefficient * exponent

        return likelihood

    def predict(self, features):
        """
        Predicts the labels or values of the given features.

        :param features: The features to make predictions for.
        :return: Predicted labels or values for the features.
        """
        if self.algorithm == 'classification':

            predictions = [self.point_predict(feature=feature) for feature in features]

            return np.array(predictions)

        elif self.algorithm == 'regression':

            num_samples, _ = features.shape
            predictions = np.empty(num_samples)

            for index, feature in enumerate(features):

                posteriors = []

                for label_index, label in enumerate(self.classes):

                    prior = np.log((self.labels == label).mean())
                    pairs = zip(feature, self.mean[label_index], self.variance[label_index])

                    likelihood = np.sum([np.log(self.likelihood(data=data, mean=mean, variance=variance))
                                         for data, mean, variance in pairs])

                    posteriors.append(prior + likelihood)

                predictions[index] = self.classes[np.argmax(posteriors)]

            return predictions

        else:

            raise ValueError("Invalid algorithm. Supported values: 'classification' or 'regression")

    def point_predict(self, feature):
        """
        Predicts the label or value for a single data point (feature).

        :param feature: The data point (feature) to make a prediction for.
        :return: The predicted label or value for the data point.
        """

        posteriors = []

        for index, cls in enumerate(self.classes):

            prior = np.log(self.prior[index])
            posterior = np.sum(np.log(self.probability_density(class_index=index, feature=feature)))
            posterior = posterior + prior

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def probability_density(self, class_index, feature):
        """
        Calculates the probability density of a feature for a specific class.

        :param class_index: The index of the class.
        :param feature: The feature for which to calculate the probability density.
        :return: The probability density of the feature for the specified class.
        """

        mean = self.mean[class_index]
        var = self.variance[class_index]

        numerator = np.exp(-(np.power(feature - mean, 2)) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

