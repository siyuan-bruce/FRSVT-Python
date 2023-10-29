from __future__ import division

import numpy as np
import logging

import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds
from numpy.linalg import norm

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import time
import os

def ls_probs(m, n, A):

    r"""Function generating the length-squared (LS) probability distributions for sampling matrix A.

    Args:
        m (int): number of rows of matrix A
        n (int): row n of columns of matrix A
        A (array[complex]): most general case is a rectangular complex matrix

    Returns:
        tuple: Tuple containing the row-norms, LS probability distributions for rows and columns,
        and Frobenius norm.
    """

    # populates array with the row-norms squared of matrix A
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(row_norms))

    LS_prob_rows = np.zeros(m)

    # normalized length-square row probability distribution
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius**2

    # populates array with length-square column probability distributions
    column_norms = np.zeros(n)
    for j in range(n):
        column_norms[j] = np.abs(la.norm(A[:, j]))**2
    
    LS_prob_columns = np.zeros(n)

    # normalized length-square column probability distribution
    for j in range(n):
        LS_prob_columns[j] = column_norms[j] / A_Frobenius**2

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):

    r"""Function used to generate matrix C by performing LS sampling of rows and columns of matrix A.

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        m (int): number of rows of matrix A
        n (int): number of columns of matrix A
        r (int): number of sampled rows
        c (int): number of sampled columns
        row_norms (array[float]): norm of the rows of matrix A
        LS_prob_rows (array[float]): row LS probability distribution of matrix A
        LS_prob_columns (array[float]): column LS probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple containing the singular values (sigma), left- (w) and right-singular vectors (vh) of matrix C,
        the sampled rows (rows), the column LS prob. distribution (LS_prob_columns_R) of matrix R and split running
        times for the FKV algorithm.
    """

    tic = time.time()
    # sample row indices from row length_square distribution
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)

    columns = np.zeros(c, dtype=int)
    # sample column indices
    for j in range(c):
        # sample row index uniformly at random
        i = np.random.choice(rows, replace=True)
        # sample column from length-square distribution of row A[i]
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    toc = time.time()
    rt_sampling_C = toc - tic

    # building the lenght-squared distribution to sample columns from matrix R
    R_row = np.zeros(n)
    LS_prob_columns_R = np.zeros((r, n))

    tic = time.time()
    # creates empty array for R and C matrices. We treat R as r x c here, since we only need columns later
    R_C = np.zeros((r, c))
    C = np.zeros((r, c))

    # populates array for matrix R with the submatrix of A defined by sampled rows/columns
    for s in range(r):
        for t in range(c):
            R_C[s, t] = A[rows[s], columns[t]]

        # renormalize each row of R
        R_C[s,:] = R_C[s,:] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))

    # creates empty array of column norms
    column_norms = np.zeros(c)

    # computes column Euclidean norms
    for t in range(c):
        for s in range(r):
            column_norms[t] += np.abs(R_C[s, t])**2

    # renormalize columns of C
    for t in range(c):
        C[:, t] = R_C[:, t] * (A_Frobenius / np.sqrt(column_norms[t])) / np.sqrt(c)

    toc = time.time()
    rt_building_C = toc - tic

    tic = time.time()
    # Computing the SVD of sampled C matrix
    w, sigma, vh = la.svd(C, full_matrices=False)

    toc = time.time()
    rt_svd_C = toc - tic

    return w, rows, sigma, vh, rt_sampling_C, rt_building_C, rt_svd_C


def vl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        array[float]: reconstructed right-singular vector
    """

    n = len(A[1, :])
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    return v_approx


def uvl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple with arrays containing approximated singular vectors :math: '\bm{u}^l, \bm{v}^l'
    """

    m, n = A.shape
    u_approx = np.zeros(m)
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    u_approx = (A @ v_approx) / sigma[l]

    return u_approx, v_approx



def pmf_solve(A, mask, k, mu, epsilon=1e-2, max_iterations=1000):
    """
    Solve probabilistic matrix factorization using alternating least squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    k : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V

    epsilon : float
        convergence condition on the difference between iterative results

    max_iterations: int
        hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    m, n = A.shape
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)
    
    import time
    
    
    tic = time.time()
    
    for _ in range(max_iterations):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        #print(mean_diff)
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X
    
    toc = time.time()
    
    print("PMF Total Time:", toc - tic)
    
    return X


def pmf_solve_inspired(A, r, c, rank, mask, mu, max_iterations=1000, epsilon=1e-5):
    
    logger = logging.getLogger(__name__)
    m_rows, n_cols = np.shape(A)
    U = np.random.randn(m_rows, rank)
    V = np.random.randn(n_cols, rank)
    
    m, n = A.shape

    # 1- Generating LS probability distributions used to sample rows and columns indices of matrix A
    Y = mask * A
    LS = ls_probs(m_rows, n_cols, Y)
    rec_errors = []
    
    prev_X = np.dot(U, V.T)
    
    for k in range(max_iterations):
        # Sample and update a subset of rows in U
        sampled_rows = np.random.choice(m_rows, size=r, p=LS[1])  # LS[1] is the row sampling distribution
        #print(sampled_rows)
        for i in sampled_rows:
            #print(i)
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, V]) +
                                    mu * np.eye(rank),
                                    np.linalg.multi_dot([V.T, Y[i, :]]))

        # Sample and update a subset of columns in V
        sampled_columns = np.random.choice(n_cols, size=c, p=LS[2])  # Assume LS[2] is the column sampling distribution
        for j in sampled_columns:
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, U]) +
                                    mu * np.eye(rank),
                                    np.linalg.multi_dot([U.T, Y[:, j]]))
    
            # New solution
        X = np.linalg.multi_dot([U, V.T])
            #X = 0.9 * X + 0.1 * X_new
    
        mean_diff = np.linalg.norm(X - prev_X) / m / n
        print(mean_diff)
        if k % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (k + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X
        
    return X

