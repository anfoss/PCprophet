import numpy as np
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix, csr_matrix
import sklearn.preprocessing
from fractions import Fraction
from itertools import permutations
from scipy.sparse import isspmatrix, dok_matrix, find
import sys
import networkx as nx
from matplotlib.pylab import show, cm, axis


def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    if isspmatrix(a) and isspmatrix(b):
        diff = (a - b).copy()
        diff.data = np.abs(diff.data)
        max_diff = diff.max()
        return max_diff <= atol + rtol * np.max(np.abs(b.data))
    else:
        return np.allclose(a, b, rtol=rtol, atol=atol)


def normalize(matrix):
    """
    Normalize the columns of the given matrix

    :param matrix: The matrix to be normalized
    :returns: The normalized matrix
    """
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)


def inflate(matrix, power):
    """
    Apply cluster inflation to the given matrix by raising
    each element to the given power.

    :param matrix: The matrix to be inflated
    :param power: Cluster inflation parameter
    :returns: The inflated matrix
    """
    if isspmatrix(matrix):
        return normalize(matrix.power(power))

    return normalize(np.power(matrix, power))


def expand(matrix, power):
    """
    Apply cluster expansion to the given matrix by raising
    the matrix to the given power using matrix multiplication.
    """
    if power < 1 or not isinstance(power, int):
        raise ValueError("Power must be a positive integer")
    if isspmatrix(matrix):
        result = matrix
        for _ in range(power - 1):
            result = result @ matrix  # sparse matrix multiplication
        return result
    else:
        return np.linalg.matrix_power(matrix, power)



def add_self_loops(matrix, loop_value):
    """
    Add self-loops to the matrix by setting the diagonal
    to loop_value

    :param matrix: The matrix to add loops to
    :param loop_value: Value to use for self-loops
    :returns: The matrix with self-loops
    """
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"
    if isspmatrix(matrix):
        # Convert to LIL for efficient assignment
        new_matrix = matrix.tolil()
        for i in range(shape[0]):
            new_matrix[i, i] = loop_value
        # Convert back to CSC for consistency
        return new_matrix.tocsc()
    else:
        new_matrix = matrix.copy()
        for i in range(shape[0]):
            new_matrix[i, i] = loop_value
        return new_matrix


def prune(matrix, threshold):
    """
    Prune the matrix so that very small edges are removed.
    The maximum value in each column is never pruned.

    :param matrix: The matrix to be pruned
    :param threshold: The value below which edges will be removed
    :returns: The pruned matrix
    """
    if isspmatrix(matrix):
        pruned = dok_matrix(matrix.shape)
        pruned[matrix >= threshold] = matrix[matrix >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    # keep max value in each column. same behaviour for dense/sparse
    num_cols = matrix.shape[1]
    row_indices = matrix.argmax(axis=0).reshape((num_cols,))
    col_indices = np.arange(num_cols)
    pruned[row_indices, col_indices] = matrix[row_indices, col_indices]

    return pruned


def converged(matrix1, matrix2):
    """
    Check for convergence by determining if
    matrix1 and matrix2 are approximately equal.

    :param matrix1: The matrix to compare with matrix2
    :param matrix2: The matrix to compare with matrix1
    :returns: True if matrix1 and matrix2 approximately equal
    """
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)



def get_clusters(matrix):
    """
    Retrieve clusters from the MCL matrix (optimized).
    """
    if not isspmatrix(matrix):
        matrix = csr_matrix(matrix)  # CSR for fast row slicing
    elif not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()

    attractors = matrix.diagonal().nonzero()[0]
    submatrix = matrix[attractors]
    clusters = []
    indptr = submatrix.indptr
    indices = submatrix.indices
    for i in range(len(attractors)):
        start, end = indptr[i], indptr[i+1]
        clusters.append(tuple(indices[start:end]))

    unique_clusters = sorted(set(clusters))

    return unique_clusters



def run_mcl(
    matrix,
    expansion=2,
    inflation=2,
    loop_value=1,
    iterations=100,
    pruning_threshold=0.001,
    pruning_frequency=1,
    convergence_check_frequency=1,
    verbose=False,
):
    """
    Perform MCL on the given similarity matrix

    :param matrix: The similarity matrix to cluster
    :param expansion: The cluster expansion factor
    :param inflation: The cluster inflation factor
    :param loop_value: Initialization value for self-loops
    :param iterations: Maximum number of iterations
           (actual number of iterations will be less if convergence is reached)
    :param pruning_threshold: Threshold below which matrix elements will be set
           set to 0
    :param pruning_frequency: Perform pruning every 'pruning_frequency'
           iterations.
    :param convergence_check_frequency: Perform the check for convergence
           every convergence_check_frequency iterations
    :param verbose: Print extra information to the console
    :returns: The final matrix
    """
    assert expansion > 1, "Invalid expansion parameter"
    assert inflation > 1, "Invalid inflation parameter"
    assert loop_value >= 0, "Invalid loop_value"
    assert iterations > 0, "Invalid number of iterations"
    assert pruning_threshold >= 0, "Invalid pruning_threshold"
    assert pruning_frequency > 0, "Invalid pruning_frequency"
    assert convergence_check_frequency > 0, "Invalid convergence_check_frequency"
    if loop_value > 0:
        matrix = add_self_loops(matrix, loop_value)
    matrix = normalize(matrix)
    for i in range(iterations):

        # store current matrix for convergence checking
        last_mat = matrix.copy()
        matrix = expand(matrix, expansion)
        matrix = inflate(matrix, inflation)
        if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
            matrix = prune(matrix, pruning_threshold)

        # Check for convergence
        if i % convergence_check_frequency == convergence_check_frequency - 1:
            if converged(matrix, last_mat):
                break

    return matrix


def is_undirected(matrix):
    """
    Determine if the matrix reprensents a directed graph
    :param matrix: The matrix to tested
    :returns: boolean
    """
    if isspmatrix(matrix):
        return sparse_allclose(matrix, matrix.transpose())

    return np.allclose(matrix, matrix.T)


def convert_to_adjacency_matrix(matrix):
    """
    Converts transition matrix into adjacency matrix
    :param matrix: The matrix to be converted
    :returns: adjacency matrix
    """
    for i in range(matrix.shape[0]):

        if isspmatrix(matrix):
            col = find(matrix[:, i])[2]
        else:
            col = matrix[:, i].T.tolist()[0]

        coeff = max(Fraction(c).limit_denominator().denominator for c in col)
        matrix[:, i] *= coeff

    return matrix


def modularity(matrix, clusters, undirected=True):
    # Ensure CSR for fast row slicing
    if isspmatrix(matrix):
        matrix = matrix.tocsr()

    m = matrix.sum()
    if undirected:
        m /= 2  # In undirected graphs, sum counts each edge twice

    # Precompute degrees
    if undirected:
        degrees = np.array(matrix.sum(axis=1)).flatten()
    else:
        out_deg = np.array(matrix.sum(axis=1)).flatten()
        in_deg = np.array(matrix.sum(axis=0)).flatten()

    Q = 0.0
    for cluster in clusters:
        cluster = list(cluster)
        submat = matrix[cluster, :][:, cluster]  # adjacency for cluster

        e_c = submat.sum()
        if undirected:
            e_c /= 2  # each edge counted twice

        a_c = degrees[cluster].sum() if undirected else out_deg[cluster].sum()
        Q += (e_c / m) - (a_c / (2 * m)) ** 2 if undirected else (e_c / m) - (a_c * in_deg[cluster].sum()) / (m**2)

    return Q