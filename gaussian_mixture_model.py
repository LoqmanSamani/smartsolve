class GaussianMixtureModel:

    def __init__(self, training_data, num_components=2, max_iteration=1000, con_tolerance=1e-4):

        """
        the training_data & validation_data must both have this structure:
        list = [[x11,x12,...,x1n],[x21,x22,x2n],...,[xm1,xm2,...,xmn]]
        """

        self.training_data = training_data
        self.num_components = num_components
        self.max_iteration = max_iteration
        self.con_tolerance = con_tolerance
        self.means = None
        self.covariances = None
        self.mix_coefficients = None





    def parameter_initializer(self, data):

        import numpy as np

        num_features = data.shape[1]

        minimum = np.min(data, axis=0)  # finds the min number for each feature
        maximum = np.max(data, axis=0)  # finds the max number for each feature

        # generates for each component a mean
        means = np.random.uniform(minimum, maximum, size=(self.num_components, num_features))
        # generates covariances
        covariances = np.array([np.diag(np.random.rand(num_features)) for _ in range(self.num_components)])

        # Initialize mixing coefficients
        mix_coefficients = np.ones(self.num_components) / self.num_components

        return means, covariances, mix_coefficients








    def marginal_likelihood(self, point, mix_coefficients, component_distributions):

        marginal_likelihood = 0

        for j in range(len(mix_coefficients)):

            component_likelihood = component_distributions[j].pdf(point)
            marginal_likelihood += mix_coefficients[j] * component_likelihood


        return marginal_likelihood








    def component_distribution(self, means, covariances):

        from scipy.stats import multivariate_normal

        com_distributions = []
        for i in range(len(means)):

            com_distribution = multivariate_normal(mean=means[i], cov=covariances[i])
            com_distributions.append(com_distribution)

        return com_distributions








    def expectation_step(self, data, means, covariances, mix_coefficients):

        com_distributions = self.component_distribution(means, covariances)

        responsibilities = []

        for i in range(len(data)):

            point_responsibilities = []

            for j in range(len(mix_coefficients)):

                marginal_likelihood = self.marginal_likelihood(data[i], mix_coefficients, com_distributions)

                responsibility = (mix_coefficients[j] * com_distributions[j].pdf(data[i])) / marginal_likelihood

                point_responsibilities.append(responsibility)

            # Normalize responsibilities for this data point
            sum_responsibilities = sum(point_responsibilities)

            normalized_responsibilities = [r / sum_responsibilities for r in point_responsibilities]

            responsibilities.append(normalized_responsibilities)


        return responsibilities








    def mean_updating(self, data, responsibilities):

        updated_means = []
        for j in range(self.num_components):

            mean_j = sum([(responsibilities[i][j] * data[i]) / sum(responsibilities[i][j]) for i in range(len(data))])

            updated_means.append(mean_j)


        return updated_means









    def covariance_updating(self, data, responsibilities):

        updated_means = self.mean_updating(data, responsibilities)

        updated_covariances = []

        for j in range(self.num_components):

            denominator = sum(responsibilities[i][j] for i in range(len(data)))

            covariance_j = [(1 / denominator) * ((data[i] - updated_means[j]) @ (data[i] - updated_means[j]).T) for i in range(len(data))]

            updated_covariances.append(covariance_j)


        return updated_covariances







    def mix_coefficient_updating(self, responsibilities):

        updated_mix_coefficients = []

        for j in range(self.num_components):

            mix_coefficient_j = sum([responsibilities[i][j] for i in range(len(responsibilities))]) / len(responsibilities)

            updated_mix_coefficients.append(mix_coefficient_j)


        return updated_mix_coefficients







    def maximization_step(self, data, responsibilities):

        self.means = self.mean_updating(data, responsibilities)

        self.covariances = self.covariance_updating(data, responsibilities)

        self.mix_coefficients = self.mix_coefficient_updating(responsibilities)







    def log_likelihood(self, data):

        import numpy as np

        log_likelihood = 0

        for i in range(len(data)):

            point_likelihood = 0

            for j in range(self.num_components):

                com_distributions = self.component_distribution(self.means, self.covariances)

                component_likelihood = com_distributions[j].pdf(data[i])

                weighted_likelihood = self.mix_coefficients[j] * component_likelihood

                point_likelihood += weighted_likelihood

            log_likelihood += np.log(point_likelihood)


        return np.array(log_likelihood)







    def train(self):

        import numpy as np

        num_iter = 0

        prev_log_likelihood = None

        data = np.array(self.training_data)

        means, covariances, mix_coefficients = self.parameter_initializer(data)

        for i in range(self.max_iteration):

            responsibilities = self.expectation_step(data, means, covariances, mix_coefficients)

            self.maximization_step(data, responsibilities)

            log_likelihood = self.log_likelihood(data)

            if prev_log_likelihood is not None:

                if abs(log_likelihood - prev_log_likelihood) < self.con_tolerance:

                    print(f"Converged after {num_iter} iterations.")
                    break

            prev_log_likelihood = log_likelihood

            num_iter += 1


        print(f"Reached maximum iterations ({self.max_iteration}).")









    def predict(self, data):

        import numpy as np

        data = np.array(data)

        responsibilities = np.array(self.expectation_step(data, self.means, self.covariances, self.mix_coefficients))

        predicted = np.max(responsibilities, axis=1)


        return predicted





































