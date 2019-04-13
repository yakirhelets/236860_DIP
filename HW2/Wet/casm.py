import numpy as np
from scipy.linalg import toeplitz


def matrix_to_vector(matrix):
    """
    converting a matrix into a vector.
    """
    height, width = matrix.shape
    output_vector = np.zeros(height * width, dtype=matrix.dtype)
    # flip the input matrix up-down because last row should go first
    matrix = np.flipud(matrix)
    for i, row in enumerate(matrix):
        st = i * width
        nd = st + width
        output_vector[st:nd] = row
    return output_vector


def vector_to_matrix(vector, output_shape):
    """
    converting vector into a matrix in th shape of output_shape.
    """
    output_height, output_weight = output_shape
    matrix = np.zeros(output_shape)
    for i in range(output_height):
        st = i * output_weight
        nd = st + output_weight
        matrix[i, :] = vector[st:nd]
    # flip the output matrix up-down to get correct result
    matrix = np.flipud(matrix)
    return matrix


def convert_to_toeplitz_doubely_blocked(shape, in_matrix):
    """
    if: y=h*k
    shape is the shape of k
    in_matrix is h
    return h as toeplitz doubly blocked matrix.
    """
    # number of columns and rows of the matrix
    mat_row_num, mat_col_num = in_matrix.shape

    #  calculate the output dimensions
    output_row_num = shape[0] + mat_row_num - 1
    output_col_num = shape[1] + mat_col_num - 1

    # zero pad the matrix
    in_matrix_zero_padded = np.pad(in_matrix, ((output_row_num - mat_row_num, 0),(0, output_col_num - mat_col_num)), 'constant', constant_values=0)

    # use each row of the zero-padded matrix to create a toeplitz matrix.
    toeplitz_list = []
    for i in range(in_matrix_zero_padded.shape[0] - 1, -1, -1):
        c = in_matrix_zero_padded[i, :]
        r = np.r_[c[0], np.zeros(shape[1] - 1)]
        # the result is wrong
        toeplitz_m = toeplitz(c, r)
        toeplitz_list.append(toeplitz_m)

    # doubly blocked toeplitz indices:
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, in_matrix_zero_padded.shape[0] + 1)
    r = np.r_[c[0], np.zeros(shape[0] - 1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    # create doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape  # shape of one toeplitz matrix
    h = toeplitz_shape[0] * doubly_indices.shape[0]
    w = toeplitz_shape[1] * doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape  # height and widths of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i, j] - 1]

    return doubly_blocked

