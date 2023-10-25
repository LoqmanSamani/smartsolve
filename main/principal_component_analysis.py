import numpy as np


class PrincipalComponentAnalysis:

    def __init__(self, features, num_components):
        """
        Initializes a Principal Component Analysis (PCA) object.

        :param features: Input features as a 2D array.
        :param num_components: Number of principal components to retain.

        """

        self.features = features
        self.num_components = num_components

        self.components = None
        self.mean = None

    def train(self):
        """
        Performs PCA training on the input features.
        Computes principal components and updates the mean.
        """

        self.mean = np.mean(self.features, axis=0)
        features = self.features - self.mean

        covariance = np.cov(features.T)

        eigenvectors, eigenvalues = np.linalg.eig(covariance)

        eigenvectors = eigenvectors.T

        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[index]

        self.components = eigenvectors[:self.num_components]

    def transform(self):
        """
        Transforms the input features using the learned principal components.

        :return: Transformed features with reduced dimensions.
        """

        features = self.features - self.mean
        transformed = np.dot(features, self.components.T)

        return transformed

