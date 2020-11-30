def mult_matrix_vector(matrix, vector):
    """
    Multiplies matrix on vector
    :param matrix: 2-d array matrix with shapes (n, m)
    :param vector: 1-d array vector with shape (m)
    :return: vector v = matrix*vector
    """
    assert len(matrix[0]) == len(vector),\
        f"Matrix and vector sizes must be equal, but {len(matrix[0])} != {len(vector)}"
    n, m = len(matrix), len(matrix[0])  # shapes
    result = []
    for i in range(n):
        element = sum(matrix[i][j]*vector[j] for j in range(m))
        result.append(element)
    return result


def get_transposed(x):  # [ [1, 2], [3, 5], [7, 8] ] -> [ [1, 3, 7], [2, 5, 8] ]
    """
    Returns transposed matrix
    :param x: 2-d array matrix
    :return: 2-d array matrix
    """
    result = []
    n, m = len(x), len(x[0])
    for j in range(m):
        result.append([x[i][j]for i in range(n)])
    return result


def vec2matrix(v):  # [1, 2, 3] -> [ [1], [2], [3] ]
    """
    Makes matrix from given vector
    :param v: 1-d array vector
    :return: 2-d array matrix
    """
    return [[el] for el in v]


def matrix2vec(x):  # [ [1, 2], [3, 10] ] -> [1, 2, 3, 10]
    """
    Returns flatten matrix
    :param x: 2-d array matrix
    :return: 1-d array vector
    """
    return [el for line in x for el in line]
