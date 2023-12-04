from machine_learning.linear_algebra import intro_numpy as np


class SingularValueDecomposition:

    def __init__(self, data, num_dimension):

        self.data = data  # the input matrix. can be a nested list,
        # which will be converted to a matrix during the process
        self.num_dimension = num_dimension  # number of dimensions to return after processing the input data
        self.U = None  # The left singular vectors
        self.S = None  # The singular values
        self.Vt = None  # The transpose of the right singular vectors
        self.variance = None  # Proportion of total variance explained
        self.t_matrix = None  # transformed matrix
        """
        Singular Value Decomposition (SVD) for dimensionality reduction.

        Args:
            data (list or numpy.ndarray): Input data in the form of a nested list or a numpy array.
            num_dimension (int): Number of dimensions to return after processing the input data.

        Attributes:
            data: The input matrix, which can be a nested list or a numpy array.
            num_dimension: The number of dimensions to return after processing the input data.
            U: The left singular vectors.
            S: The singular values.
            Vt: The transpose of the right singular vectors.
            variance: Proportion of total variance explained.
            t_matrix: Transformed matrix with reduced dimensions.
        """

    def matrix_decomposer(self, matrix):
        """
        Perform SVD decomposition on the input matrix.

        Args:
            matrix (numpy.ndarray): Input matrix.

        Returns:
            left_singular_vector (numpy.ndarray): The left singular vectors.
            singular_values (numpy.ndarray): The singular values.
            t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.
        """

        # using nampy.linalg.svd to calculate all three returned matrices

        left_singular_vector, singular_values, t_right_singular_vectors = np.linalg.svd(
            matrix, full_matrices=False, compute_uv=True
        )

        return left_singular_vector, singular_values, t_right_singular_vectors

    def dimension_reducer(self, left_singular_vector, singular_values, t_right_singular_vectors):
        """
        Reduce dimensions of the decomposed matrices.

        Args:
            left_singular_vector (numpy.ndarray): The left singular vectors.
            singular_values (numpy.ndarray): The singular values.
            t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.

        Returns:
            k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
            k_singular_matrix (numpy.ndarray): Reduced singular values as a diagonal matrix.
            k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.
        """
        # Reduce the dimensions of left_singular_vector
        k_left_singular_vector = left_singular_vector[:, :self.num_dimension]

        # Create a diagonal matrix using singular values and truncate it to the first num_dimension values
        k_singular_matrix = np.diag(singular_values[:self.num_dimension])

        # Reduce the dimensions of t_right_singular_vectors
        k_right_singular_vectors = t_right_singular_vectors[:self.num_dimension, :]

        return k_left_singular_vector, k_singular_matrix, k_right_singular_vectors



    def matrix_reconstructor(self, k_left_singular_vector, k_singular_matrix, k_right_singular_vectors):
        """
        Reconstruct the original matrix from reduced dimensions.

        Args:
            k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
            k_singular_matrix (numpy.ndarray): Reduced singular values as a diagonal matrix.
            k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.

        Returns:
            reconstructed_matrix (numpy.ndarray): Reconstructed matrix.
        """
        # reconstruct the matrix using this formula: A = U * S * Vt
        reconstructed_matrix = np.dot(k_left_singular_vector, k_singular_matrix).dot(k_right_singular_vectors)

        return reconstructed_matrix






    def transform(self):
        """
        Perform dimensionality reduction and reconstruction in a single step.

        Returns:
            reconstructed_matrix (numpy.ndarray): Transformed matrix with reduced dimensions.
        """
        # check the type of the input data
        if isinstance(self.data, np.ndarray):
            matrix = self.data
        elif isinstance(self.data, list):
            matrix = np.array(self.data)  # converts the input data to a matrix for further calculations
        else:
            raise ValueError("The input must be in form numpy.ndarray or a nested list of points")

        if not self.num_dimension:
            raise ValueError("Please define the number of dimensions (num_dimension=?)of the output matrix.")

        # call matrix_decomposer to decompose the input data(matrix)
        self.U, self.S, self.Vt = self.matrix_decomposer(matrix=matrix)

        # call dimension_reducer  to reduce dimensions of the decomposed vectors
        k_left_singular_vector, k_singular_matrix, k_right_singular_vectors = self.dimension_reducer(
            left_singular_vector=self.U,
            singular_values=self.S,
            t_right_singular_vectors=self.Vt
        )

        # call matrix_reconstructor to combine the decomposed vectors to build the dimension-reduced matrix
        self.t_matrix = self.matrix_reconstructor(
            k_left_singular_vector=k_left_singular_vector,
            k_singular_matrix=k_singular_matrix,
            k_right_singular_vectors=k_right_singular_vectors
        )
        self.variance = self.variance_explained(
            singular_values=self.S,
            num_dimension=self.num_dimension
        )

        return self.t_matrix





    def variance_explained(self, singular_values, num_dimension):
        """
        Calculate the variance explained by the first num_dimensions singular values.

        Args:
            singular_values (numpy.ndarray): The singular values.
            num_dimension (int): Number of dimensions to consider.

        Returns:
            variance_explained (float): Proportion of total variance explained.
        """
        singular_values = np.array(singular_values)  # if the input is not numpy.ndarray

        squared_values = np.power(singular_values[:num_dimension], 2)
        sum_all_squared_values = np.sum(np.power(singular_values, 2))

        variance_explained = squared_values / sum_all_squared_values

        self.variance = variance_explained

        return variance_explained












