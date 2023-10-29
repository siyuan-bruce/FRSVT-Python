from __future__ import division
import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip, inspired_randomized_svd
from scipy.sparse.linalg import svds
from numpy.linalg import norm

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import time
import os
import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds


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

    LS_prob_columns = np.zeros((m, n))

    # populates array with length-square column probability distributions
    # LS_prob_columns[i]: LS probability distribution for selecting columns from row A[i]
    for i in range(m):
        LS_prob_columns[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

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


    return C


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
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l])
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    u_approx = (A @ v_approx) / sigma[l]

    return u_approx, v_approx


def _my_svd(M, k, algorithm):
    if algorithm == 'randomized':
        (U, S, V) = randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=5)
    elif algorithm == 'inspired':
        (U, S, V) = inspired_randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=5)
    elif algorithm == 'arpack':
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("unknown algorithm")
    return (U, S, V)

def svt_solve(
        A, 
        mask, 
        tau=None, 
        delta=None, 
        epsilon=1e-1,
        rel_improvement=-0.01,
        max_iterations=100,
        algorithm='arpack'):
    """
    Solve using iterative singular value thresholding.

    [ Cai, Candes, and Shen 2010 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    tau : float
        singular value thresholding amount;, default to 5 * (m + n) / 2

    delta : float
        step size per iteration; default to 1.2 times the undersampling ratio

    epsilon : float
        convergence condition on the relative reconstruction error

    max_iterations: int
        hard limit on maximum number of iterations

    algorithm: str, 'arpack' or 'randomized' (default='arpack')
        SVD solver to use. Either 'arpack' for the ARPACK wrapper in 
        SciPy (scipy.sparse.linalg.svds), or 'randomized' for the 
        randomized algorithm due to Halko (2009).

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    if algorithm not in ['randomized', 'arpack', 'inspired']:
        raise ValueError("unknown algorithm %r" % algorithm)
    Y = np.zeros_like(A)

    if not tau:
        tau = np.sum(A.shape) / 10
    if not delta:
        delta = np.prod(A.shape) / np.sum(mask)

    r_previous = 0
    import time
    
        
    
    
    tic = time.time()

    for k in range(max_iterations):
        if k == 0:
            X = np.zeros_like(A)
        else:
            sk = r_previous + 1
            #print(sk)
            (U, S, V) = _my_svd(Y, sk, algorithm)
            while np.min(S) >= tau:
                sk = sk + 5
                (U, S, V) = _my_svd(Y, sk, algorithm)
            shrink_S = np.maximum(S - tau, 0)
            #print(shrink_S)
            r_previous = np.count_nonzero(shrink_S)
            
            
            #print(shrink_S)
            diag_shrink_S = np.diag(shrink_S)
            X = np.linalg.multi_dot([U, diag_shrink_S, V])
            #print(S)
            #print(tau)
        Y += delta * mask * (A - X)
        
        

        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        # print("recon_error: ",recon_error)
        if k % 1 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
        if recon_error < epsilon:
            break
    
    toc = time.time()
    
    print("SVT Total Time:", toc - tic)
    return X

