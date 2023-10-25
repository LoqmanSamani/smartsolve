import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, train_data, num_clusters, means=None, covariances=None, coefficients=None, max_iter=1000,
                 threshold=1e-4):
        """
        Initialize a Gaussian Mixture Model.

        :param train_data: Training data as a list of data points.
        :param num_clusters: Number of clusters in the model.
        :param means: Initial means for clusters (default is None).
        :param covariances: Initial covariances for clusters (default is None).
        :param coefficients: Initial coefficients for clusters (default is None).
        :param max_iter: Maximum number of iterations during training (default is 1000).
        :param threshold: Convergence threshold for log-likelihood (default is 1e-4).
        """

        self.train_data = train_data
        self.num_clusters = num_clusters
        self.means = means
        self.covariances = covariances
        self.coefficients = coefficients
        self.max_iter = max_iter
        self.threshold = threshold

        self.responsibilities = None
        self.loglikelihood = None
        self.loglikelihood_trace = []

    def train(self):
        """
        Train the Gaussian Mixture Model using the provided data and parameters.
        """

        if not self.means:

            np.random.seed(42)
            random_indices = np.random.choice(len(self.train_data), self.num_clusters, replace=False)
            self.means = [self.train_data[i] for i in random_indices]

        if not self.covariances:

            self.covariances = [np.identity(len(self.train_data[0])) for _ in range(self.num_clusters)]

        if not self.coefficients:
            self.coefficients = [1.0 / self.num_clusters] * self.num_clusters

        num_point = len(self.train_data)

        self.responsibilities = np.zeros((num_point, self.num_clusters))

        self.loglikelihood = self.compute_loglikelihood(
            data=self.train_data,
            coefficients=self.coefficients,
            means=self.means,
            covariances=self.covariances,
        )

        self.loglikelihood_trace = [self.loglikelihood]

        for i in range(self.max_iter):

            self.responsibilities = self.compute_response(
                data=self.train_data,
                coefficients=self.coefficients,
                means=self.means,
                covariances=self.covariances,
            )

            counts = self.soft_counts(
                responsibilities=self.responsibilities
            )

            self.coefficients = self.compute_coefficients(
                counts=counts
            )

            self.means = self.compute_means(
                data=self.train_data,
                responsibilities=self.responsibilities,
                counts=counts
            )

            covariances = self.compute_covariances(
                data=self.train_data,
                responsibilities=self.responsibilities,
                counts=counts,
                means=self.means
            )

            l_loglikelihood = self.compute_loglikelihood(
                data=self.train_data,
                coefficients=self.coefficients,
                means=self.means,
                covariances=covariances
            )

            self.loglikelihood_trace.append(l_loglikelihood)


    def compute_loglikelihood(self, data, coefficients, means, covariances):
        """
        Calculate the log-likelihood of the data given the model parameters.

        :param data: Data points for log-likelihood calculation.
        :param coefficients: Cluster coefficients.
        :param means: Cluster means.
        :param covariances: Cluster covariances.

        :return: Log-likelihood of the data.
        """

        num_clusters = len(means)
        num_dimension = len(data[0])

        loglikelihood = 0

        for point in data:

            results = np.zeros(num_clusters)

            for i in range(num_clusters):

                delta = np.array(point) - means[i]
                exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covariances[i]), delta))

                results[i] += np.log(coefficients[i])
                results[i] -= 1 / 2. * (num_dimension * np.log(2 * np.pi) + np.log(np.linalg.det(covariances[i])) + exponent_term)

            loglikelihood += self.log_sum_exp(results=results)

        return loglikelihood

    def log_sum_exp(self, results):
        """
        Calculate the log of the sum of exponentials.

        :param results: List of results to calculate the log-sum-exp for.

        :return: Log of the sum of exponentials.
        """

        loglikelihood = np.max(results) + np.log(np.sum(np.exp(results - np.max(results))))

        return loglikelihood

    def compute_response(self, data, coefficients, means, covariances):
        """
        Compute the responsibilities for each data point.

        :param data: Data points.
        :param coefficients: Cluster coefficients.
        :param means: Cluster means.
        :param covariances: Cluster covariances.

        :return: Responsibilities for each data point.
        """

        num_point = len(data)
        num_clusters = len(means)
        responsibilities = np.zeros((num_point, num_clusters))

        for i in range(num_point):
            for k in range(num_clusters):
                responsibilities[i, k] = coefficients[k] * multivariate_normal.pdf(data[i], means[k], covariances[k])

        row_sums = responsibilities.sum(axis=1)[:, np.newaxis]
        resp = responsibilities / row_sums

        return resp

    def soft_counts(self, responsibilities):
        """
        Calculate soft counts based on responsibilities.

        :param responsibilities: Responsibilities for data points.

        :return: Soft counts for each cluster.
        """

        counts = np.sum(responsibilities, axis=0)

        return counts

    def compute_coefficients(self, counts):
        """
        Calculate cluster coefficients based on soft counts.

        :param counts: Soft counts for each cluster.

        :return: Cluster coefficients.
        """

        num_clusters = len(counts)
        coefficients = [0.] * num_clusters

        for k in range(num_clusters):

            coefficients[k] = counts[k]

        return coefficients

    def compute_means(self, data, responsibilities, counts):
        """
        Calculate updated means for clusters.

        :param data: Data points.
        :param responsibilities: Responsibilities for data points.
        :param counts: Soft counts for each cluster.

        :return: Updated cluster means.
        """

        num_clusters = len(counts)
        num_data = len(data)
        means = [np.zeros(len(data[0]))] * num_clusters

        for k in range(num_clusters):

            weighted_sum = 0.
            for i in range(num_data):
                weighted_sum += responsibilities[i, k] * np.array(data[i])
            means[k] = weighted_sum / counts[k]

        return means

    def compute_covariances(self, data, responsibilities, counts, means):
        """
        Calculate updated covariances for clusters.

        :param data: Data points.
        :param responsibilities: Responsibilities for data points.
        :param counts: Soft counts for each cluster.
        :param means: Cluster means.

        :return: Updated cluster covariances.
        """

        num_clusters = len(counts)
        num_dim = len(data[0])
        num_data = len(data)
        covariances = [np.zeros((num_dim, num_dim))] * num_clusters

        for k in range(num_clusters):

            weighted_sum = np.zeros((num_dim, num_dim))

            for i in range(num_data):

                weighted_sum += responsibilities[i, k] * np.outer(data[i] - means[k], data[i] - means[k])

            covariances[k] = weighted_sum / counts[k]

        return covariances


