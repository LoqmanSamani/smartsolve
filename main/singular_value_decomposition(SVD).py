from numpy import array
from numpy.linalg import svd
from numpy import diag
from numpy import dot
from numpy import power
from numpy import sum as s
from numpy import ndarray



class SingularValueDecomposition:

    """
    the input data must have this structure:
    a nested list or can be a numpy array = [[x11,x12,...,x1n], [x21,x22,x2n],...,[xm1,xm2,...,xmn]]
    """

    def __init__(self, data, num_dimensions):
        self.data = data # the input matrix of can be a nested list, which will be converted to a matrix during the process
        self.num_dimensions = num_dimensions  # number of dimensions to return after processing the input data
        self.U = None  # The left singular vectors
        self.S = None  # The singular values
        self.Vt = None  # The transpose of the right singular vectors
        self.variance = None  # Proportion of total variance explained


    def matrix_decomposer(self, matrix):
        """
        Perform SVD decomposition on the input training_data.
        Parameter:
        A matrix which is generated from input data
        Returns:
        left_singular_vector (numpy.ndarray): The left singular vectors.
        singular_values (numpy.ndarray): The singular values.
        t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.
        """

        # using nampy.linalg.svd to calculate all three returned matrices
        left_singular_vector, singular_values, t_right_singular_vectors = svd(matrix, full_matrices=False, compute_uv=True)
        self.U = left_singular_vector
        self.S = singular_values
        self.Vt = t_right_singular_vectors

        return left_singular_vector, singular_values, t_right_singular_vectors





    def dimension_reducer(self, left_singular_vector, singular_values, t_right_singular_vectors):
        """
        Reduce dimensions of the decomposed matrices.

        Parameters:
        left_singular_vector (numpy.ndarray): The left singular vectors.
        singular_values (numpy.ndarray): The singular values.
        t_right_singular_vectors (numpy.ndarray): The transpose of the right singular vectors.

        Returns:
        k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
        k_singular_matrix (numpy.ndarray): Reduced singular values.
        k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.
        """

        # reduce the dimensions of left_singular_vector
        lsv_lst = list(left_singular_vector)
        k_left_singular_vector = array([row[:self.num_dimensions] for row in lsv_lst])

        # using singular values to generate a diagonal matrix
        k_singular_matrix = diag(singular_values, self.num_dimensions)

        # reduce the dimensions of t_right_singular_vector
        rsv_lst = list(t_right_singular_vectors)
        k_right_singular_vectors = array([row[:self.num_dimensions] for row in rsv_lst])

        return k_left_singular_vector, k_singular_matrix, k_right_singular_vectors







    def matrix_reconstructor(self, k_left_singular_vector, k_singular_matrix, k_right_singular_vectors):
        """
        Reconstruct the original matrix from reduced dimensions.

        parameters:
        k_left_singular_vector (numpy.ndarray): Reduced left singular vectors.
        k_singular_matrix (numpy.ndarray): Reduced singular values.
        k_right_singular_vectors (numpy.ndarray): Reduced transpose of right singular vectors.

        Returns:
        reconstructed_matrix (numpy.ndarray): Reconstructed matrix.
        """
        # reconstruct the matrix using this formula: A = U * S * Vt
        reconstructed_matrix = dot(k_left_singular_vector, k_singular_matrix).dot(k_right_singular_vectors)

        return reconstructed_matrix






    def transform(self):
        """
        Perform dimensionality reduction and reconstruction in a single step.

        Returns:
        reconstructed_matrix (numpy.ndarray): Transformed matrix with reduced dimensions.
        """
        # check the type of the input data
        if isinstance(self.data, ndarray):
            matrix = self.data
        elif isinstance(self.data, list):
            matrix = array(self.data)  # converts the input data to a matrix for further calculations
        else:
            raise ValueError("The input must be in form numpy.ndarray or a nested list of points")

        # call matrix_decomposer to decompose the input data(matrix)
        left_singular_vector, singular_values, t_right_singular_vectors = self.matrix_decomposer(matrix)

        # call dimension_reducer  to reduce dimensions of the decomposed vectors
        k_left_singular_vector, k_singular_matrix, k_right_singular_vectors = self.dimension_reducer(left_singular_vector, singular_values, t_right_singular_vectors)

        # call matrix_reconstructor to combine the decomposed vectors to build the dimension-reduced matrix
        reconstructed_matrix = self.matrix_reconstructor(k_left_singular_vector, k_singular_matrix, k_right_singular_vectors)

        return reconstructed_matrix







    def variance_explained(self, singular_values, num_dimensions):
        """
        Calculate the variance explained by the first num_dimensions(number of dimensions) singular values.

        Parameters:
        singular_values (numpy.ndarray): The singular values.
        num_dimensions (int): Number of dimensions to consider.

        Returns:
        variance_explained (float): Proportion of total variance explained.

        the formula to calculate it:
        squares the first k values of singular values and then dividing by
        the sum of the squares of all the values in singular values.

        variance_explained = numpy.power(self.S[:k], 2) / numpy.sum(numpy.power(self.S[:k], 2))
        """
        singular_values = array(singular_values)  # if the input is not numpy.ndarray

        squared_values = power(singular_values[:num_dimensions], 2)
        sum_all_squared_values = s(power(singular_values, 2))

        variance_explained = squared_values / sum_all_squared_values

        self.variance = variance_explained

        return variance_explained






    def parameters(self, u_return=False, s_return=False, vt_return=False, variance=False):
        """
        Return selected parameters as a dictionary.

        Parameters:
        u_return (bool): Whether to return left singular vectors (U).
        s_return (bool): Whether to return singular values (S).
        vt_return (bool): Whether to return transpose of right singular vectors (Vt).
        variance (bool): Whether to return proportion of total variance explained.

        Returns:
        params (dict): Dictionary containing selected parameters.

        Example usage:
        params = svd_instance.parameters(u_return=True, s_return=True)
        """
        params = {}

        if u_return:
            params['U'] = self.U
        if s_return:
            params['S'] = self.S
        if vt_return:
            params['Vt'] = self.Vt
        if variance:
            params['Variance'] = self.variance

        if not any([u_return, s_return, vt_return, variance]):
            raise ValueError("At least one parameter (u_return, s_return, vt_return, variance) must be set to True.")

        return params
















