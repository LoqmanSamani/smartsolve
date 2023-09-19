class PrincipalComponentAnalysis:

    def __init__(self, var_threshold=0.95, num_components=None):
        """
        Initialize the PrincipalComponentAnalysis instance.
        var_threshold: The threshold for explained variance (default is 0.95).
        num_components: The number of principal components to select (default is None).
        """

        self.var_threshold = var_threshold
        self.num_components = num_components





    def feature_centering(self, feature):
        """
        Center the given feature by subtracting its mean.
        param feature: A numpy array representing a feature.
        return: Centered feature.
        """

        import numpy as np
        mean = np.mean(feature)

        centered_feature = feature - mean

        return centered_feature





    def covariance_matrix_calculator(self, centered_features):
        """
        Calculate the covariance matrix of centered features.
        param centered_features: List of numpy arrays, each representing a centered feature.
        return: Covariance matrix.
        """

        import numpy as np

        covariance_matrix = np.cov(centered_features)

        return covariance_matrix





    def eigenvalue_decomposition_calculator(self, covariance_matrix):
        """
        Perform eigenvalue decomposition on the given covariance matrix.
        param covariance_matrix: Covariance matrix.
        return: Eigenvalues and eigenvectors.
        """

        from numpy import linalg as la

        eigenvalues, eigenvectors = la.eig(covariance_matrix)

        return eigenvalues, eigenvectors





    def sum_variance(self, eigenvalues):

        """
        Calculate the total sum of eigenvalues.
        param eigenvalues: Eigenvalues.
        return: Total sum of eigenvalues.
        """

        import numpy as np

        sum_variance = np.sum(eigenvalues)

        return sum_variance






    def explained_variance_threshold(self, sum_variance):
        """
        Calculate the threshold for explained variance based on the total sum of eigenvalues.
        param sum_variance: Total sum of eigenvalues.
        return: Threshold for explained variance.
        """

        ex_var_threshold = int(self.var_threshold * sum_variance)

        return ex_var_threshold






    def cumulative_variance(self, cumulative_variance,eigenvalue):
        """
        Calculate cumulative variance by adding an eigenvalue to the existing cumulative variance.
        param cumulative_variance: Current cumulative variance.
        param eigenvalue: Eigenvalue to be added.
        return: Updated cumulative variance.
        """

        cum_variance = cumulative_variance + eigenvalue

        return cum_variance






    def select_principal_components(self, eigenvalues, eigenvectors):
        """
        Select the principal components based on explained variance threshold.
        param eigenvalues: Eigenvalues.
        param eigenvectors: Eigenvectors.
        return: Selected component vectors.
        """

        sum_variance = self.sum_variance(eigenvalues)

        ex_var_threshold = self.explained_variance_threshold(sum_variance)

        cumulative_variance = 0

        component_vectors = []

        num_components = 0

        for i in range(len(eigenvalues)):

            if cumulative_variance < ex_var_threshold:

                cum_variance = self.cumulative_variance(cumulative_variance, eigenvalues[i])

                cumulative_variance += cum_variance

                num_components += 1

                component_vectors.append(eigenvectors[i])

            else:
                break

        if self.num_components is None:

            self.num_components = num_components

        return component_vectors






    def point_component_dot_product(self, component_vector, data):
        """
        Calculate dot products between data points and a component vector.
        param component_vector: A principal component vector.
        param data: Input data.
        return: List of dot products.
        """

        import numpy as np

        dot_products = []
        for point in data:

            vectorized_point = np.array(point)
            dot_product = np.dot(component_vector, vectorized_point)
            dot_products.append(dot_product)

        return dot_products







    def project_data(self, data):
        """
        Project the input data onto the selected principal components.
        param data: Input data.
        return: Transformed data and eigenvectors.
        """

        import numpy as np

        centered_features = []

        try:
            for i in range(len(data[0])):

                feature = np.array([point[i] for point in data])

                centered_feature = self.feature_centering(feature)

                centered_features.append(centered_feature)


            covariance_matrix = self.covariance_matrix_calculator(np.array(centered_features))

            eigenvalues, eigenvectors = self.eigenvalue_decomposition_calculator(covariance_matrix)

            component_vectors = self.select_principal_components(eigenvalues, eigenvectors)

            projected_data = []

            for component_vector in component_vectors:

                dot_products = self.point_component_dot_product(component_vector, data)

                projected_data.append(dot_products)


            transformed_data = []

            for point in range(len(projected_data[0])):

                transformed_point = [component[point] for component in projected_data]

                transformed_data.append(transformed_point)

        except TypeError:

            return "The input data must have this structure : list = [[x11,x12,...,x1n],[x21,x22,x2n],...,[xm1,xm2,...,xmn]]"



        return transformed_data, eigenvectors







    def inverse_project_data(self):
        """
        Inverse project transformed data to reconstruct the original data.
        return: Inverse-projected original data.
        """

        import numpy as np

        transformed_data, eigenvectors = self.project_data()

        reverse_eigenvectors = np.linalg.inv(eigenvectors)

        original_data = reverse_eigenvectors.dot(np.array(transformed_data))

        return original_data






    def explained_variance(self, data):
        """
        Calculate explained variance for each principal component.
        param data: Input data.
        return: Explained variance ratios.
        """

        import numpy as np

        centered_features = []

        for i in range(len(data[0])):

            feature = np.array([point[i] for point in data])

            centered_feature = self.feature_centering(feature)

            centered_features.append(centered_feature)

        covariance_matrix = self.covariance_matrix_calculator(np.array(centered_features))

        eigenvalues, _ = self.eigenvalue_decomposition_calculator(covariance_matrix)

        # Calculate the total sum of eigenvalues
        sum_variance = self.sum_variance(eigenvalues)

        # Calculate the explained variance for each principal component
        explained_variance_ratios = eigenvalues / sum_variance

        return explained_variance_ratios




















































