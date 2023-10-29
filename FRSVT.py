import numpy as np
from numpy.linalg import norm
import time

import warnings

warnings.filterwarnings('ignore')
# seed
np.random.seed(42)

rng = np.random.default_rng(2023)  # Create a new random number generator instance
#random_number = rng.random()  # Generate a random number

def Polar(C):
    U, S, V = np.linalg.svd(C)
    W = np.dot(U, V.T)
    # transform S into a diagonal matrix
    S = np.diag(S)
    P = np.dot(np.dot(V, S), V.T)
    return W, P

def QR_CP(Y):
    B = Y.astype(float)
    m, n = B.shape
    Q = np.zeros((m, n))
    bmax = B[:, 0]
    flag = 0

    for i in range(1, n):
        if norm(bmax) < norm(B[:, i]):
            bmax = B[:, i]
            flag = i

    B[:, flag] = B[:, 0]
    B[:, 0] = bmax
    Q[:, 0] = B[:, 0] / norm(B[:, 0])

    for j in range(1, n):
        bmax2 = np.zeros((m, 1))
        flag2 = j
        

        for i in range(j):
            
            temp = np.dot(Q[:, i].T, B[:, j]) * Q[:, i]

            temp = np.reshape(temp, (m, 1))
            
            bmax2 += temp
        
        bmax2 = np.reshape(B[:, j], (m,1)) - bmax2
        

        for u in range(j + 1, n):
            c = np.zeros((m, 1))

            for i in range(j):
                c += np.reshape((np.dot(Q[:, i].T, B[:, u]) * Q[:, i]), (m,1))

            if norm(bmax2) < norm(B[:, u] - c):
                bmax2 = np.reshape(B[:, u],(m, 1)) - c
                flag2 = u

        d = B[:, flag2]
        B[:, flag2] = B[:, j]
        B[:, j] = d
        Q[:, j] = np.squeeze(bmax2) / norm(bmax2)
        #bmax2 / norm(bmax2)

    return Q

def S_tau(D, tau):
    n = D.shape[0]
    S_D = np.zeros((n, n))
    for i in range(n):
        S_D[i, i] = np.sign(D[i, i]) * max([abs(D[i, i]) - tau, 0])
    return S_D

def Helper(A, tau, l, p, Q):
    for i in range(20):
        Q, R = np.linalg.qr(np.dot(A, np.dot(A.T, Q)))
        Q = Q[:, :R.shape[1]]
        R = R[:R.shape[1], :]

    H, C = np.linalg.qr(np.dot(A.T, Q))
    H = H[:, :C.shape[1]]
    C = C[:C.shape[1], :]
    W, P = Polar(C)
    D, V = np.linalg.eig(P)
    D = np.diag(D)
    S_D = np.dot(np.dot(Q, V), S_tau(D, tau))
    X = np.dot(S_D, np.dot(np.dot(H, W), V).T)
    Q = np.dot(Q, V)
    return X, Q

import numpy as np
def PartialOrthogonalization(Q, Y):
    A = np.hstack((Q, Y))
    m, n = A.shape
    #print(A.shape)
    for i in range(Q.shape[1], n):
        count = 0
        while np.linalg.matrix_rank(A[:, :i]) != min(A[:, :i].shape):
            count += 1
            # print("Submatrix A[:, :{}]".format(i))
            # print(A[:, :i].shape)
            #print("Rank:", np.linalg.matrix_rank(A[:, :i-1]))
            #A[:, i] = np.random.rand(m)
            A[:, i] = rng.random(m)
            
            if count > 5:
                break

    Q = A
    R = np.zeros((n, n))
    for i in range(n):
        Q[:, i] = A[:, i]
    for i in range(Q.shape[1], n):
        R[i, i] = np.linalg.norm(Q[:, i], 2)
        Q[:, i] = Q[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i].T, Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
    return Q

def FRSVT(A, tau=None, l=None, p=None):
    import time
    tic = time.time()

    if A is None:
        # Read an image if A is not provided
        import cv2
        A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    if tau is None:
        tau = 1 / np.linalg.norm(A, 2)
    if l is None:
        l = 20
    if p is None:
        p = 10

    if A.shape[0] < A.shape[1]:
        A = A.T

    m, n = A.shape
    
    tic = time.time()
    Omega = np.random.randn(n, l)
    Y = np.dot(A, Omega)
    #toc = time.time()
    #print("FRSVT Sampling Elapsed Time 1:", toc - tic)
    
    #tic = time.time()
    Q = QR_CP(Y)
    
    #toc = time.time()
    #print("FRSVT QRCP Elapsed Time:", toc - tic)
    #tic = time.time()
    
    X, Q = Helper(A, tau, l, p, Q)

    #tic = time.time()
    
    Omega = np.random.randn(n, p)
    Y = np.dot(A, Omega)
    
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)
    
    toc = time.time()
    print("FRSVT Time:", toc - tic)

    #toc = time.time()
    #print("Elapsed Time:", toc - tic)

    X_real = np.real(X)
    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X_real), cmap='gray')
    # save
    plt.savefig('FRSVT.jpg')
    return X_real
    

import numpy as np
import cv2
import matplotlib.pyplot as plt

def TestSVD():
    A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
    I = A.copy()
    plt.imshow(np.uint8(A), cmap='gray')
    plt.show()

    tic = time.time()

    S, V, D = np.linalg.svd(I)

    toc = time.time()
    print("Elapsed Time:", toc - tic)
    plt.imshow(np.uint8(I), cmap='gray')
    plt.show()

    k = 10
    reconstructed_image = np.dot(np.dot(S[:, :k], np.diag(V[:k])), D[:, :k].T)
    plt.imshow(np.uint8(reconstructed_image), cmap='gray')
    plt.show()

# Call the test function
TestSVD()

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


def variance_scaled_ls_probs(m, n, A):
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

    # New part: compute variances and adjust the probabilities
    row_vars = np.var(A, axis=1)  # row variances
    col_vars = np.var(A, axis=0)  # column variances

    row_adj = 1 / (1 + row_vars)  # adjust row probabilities - lower for high variance
    col_adj = 1 / (1 + col_vars)  # adjust column probabilities - lower for high variance

    LS_prob_rows *= row_adj  # adjust row probabilities
    LS_prob_rows /= np.sum(LS_prob_rows)  # renormalize

    for i in range(m):
        LS_prob_columns[i, :] *= col_adj  # adjust column probabilities
        LS_prob_columns[i, :] /= np.sum(LS_prob_columns[i, :])  # renormalize each row

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius



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



def ls_probs_columns(m, n, A):

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
    column_norms = np.zeros(n)
    for i in range(n):
        column_norms[i] = np.abs(la.norm(A[:, i]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(column_norms))

    LS_prob_columns = np.zeros(n)

    # normalized length-square row probability distribution
    for i in range(n):
        LS_prob_columns[i] = column_norms[i] / A_Frobenius**2

    return column_norms, LS_prob_columns, A_Frobenius


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

    return C, rows, columns



def sample_C_columns(A, m, n, c, column_norms, LS_prob_columns, A_Frobenius):

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
    columns = np.random.choice(n, c, replace=True, p=LS_prob_columns)

    #columns = np.zeros(c, dtype=int)
    # sample column indices
    i = np.random.choice(columns, replace=True)

    toc = time.time()
    rt_sampling_C = toc - tic

    # building the lenght-squared distribution to sample columns from matrix R
    R_row = np.zeros(n)

    tic = time.time()
    # creates empty array for R and C matrices. We treat R as r x c here, since we only need columns later
    C_Columns = np.zeros((m, c))

    # populates array for matrix R with the submatrix of A defined by sampled rows/columns
    for s in range(c):
        C_Columns[:, s] = A[:, columns[s]]

        # renormalize each row of R
        C_Columns[:,s] = C_Columns[:,s] * A_Frobenius / (np.sqrt(c) * np.sqrt(column_norms[columns[s]]))

    return C_Columns, columns



def quantum_inspired_FRSVT(A, tau=None, l=None, p=None):
    import time
    
    if A is None:
        # Read an image if A is not provided
        import cv2
        A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    if tau is None:
        tau = 1 / np.linalg.norm(A, 2)
    if l is None:
        l = 20
    if p is None:
        p = 10

    if A.shape[0] < A.shape[1]:
        A = A.T

    row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius =  ls_probs(A.shape[0], A.shape[1], A)

    m, n = A.shape
    
    tic = time.time()
    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, l, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    
    Q = QR_CP(Y)
    X, Q = Helper(A, tau, l, p, Q)

    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, p, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
   
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)
    
    toc = time.time()
    print("QISVT Elapsed Time:", toc - tic)

    X_real = np.real(X)
    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X_real), cmap='gray')
    # save
    plt.savefig('Inspired_Q.jpg')
    return X_real



def quantum_inspired_FRSVT_columns(A, tau=None, l=None, p=None):
    import time
    
    if A is None:
        # Read an image if A is not provided
        import cv2
        A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    if tau is None:
        tau = 1 / np.linalg.norm(A, 2)
    if l is None:
        l = 20
    if p is None:
        p = 10

    if A.shape[0] < A.shape[1]:
        A = A.T

    column_norms, LS_prob_columns, A_Frobenius = ls_probs_columns(A.shape[0], A.shape[1], A)
    #print(LS_prob_columns)
    m, n = A.shape
   
    tic = time.time()
    
    Y, columns = sample_C_columns(A, m, n, l, column_norms, LS_prob_columns, A_Frobenius)

    Q = QR_CP(Y)
    
    #toc = time.time()
    #print("Inspired QRCP Elapsed Time:", toc - tic)
    #tic = time.time()
    
    X, Q = Helper(A, tau, l, p, Q)

    #toc = time.time()
    #print("Helper Elapsed Time:", toc - tic)
    
    #tic = time.time()
    
    Y, columns = sample_C_columns(A, m, n, p, column_norms, LS_prob_columns, A_Frobenius)
    
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("QISVT Time:", toc - tic)
    
    X_real = np.real(X)
    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X_real), cmap='gray')
    # save
    plt.savefig('Inspired_Q_columns.jpg')
    return X_real


# A = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

# FRSVT(A, l = 50, p = 20)

# quantum_inspired_FRSVT(A, l = 50, p = 20)


A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)


FRSVT(A, l = 20, p = 10)

quantum_inspired_FRSVT(A, l = 20, p = 10)


quantum_inspired_FRSVT_columns(A, l = 20, p = 10)


import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming FRSVT and quantum_inspired_FRSVT are functions defined elsewhere


def gen_factorization(m, n, k):
    """
    Generate noisy data for m users and n movies with k latent factors.
    Gaussian noise with variance sigma^2 is added to U V^T.
    Effect is a matrix with a few large singular values and many close to zero.
    """
    
    # Create an array of norms in a random increasing way
    norms = np.cumsum(np.abs(np.random.randn(k)))
    
    # Randomize the order of the norms
    np.random.shuffle(norms)
    
    # Generate a random matrix and use QR decomposition to get a matrix with full column rank
    U = np.linalg.qr(np.random.randn(m, k))[0]
    
    # Generate another random matrix, use QR decomposition to get a matrix with full column rank,
    # and then scale each column
    V = np.linalg.qr(np.random.randn(n, k))[0]
    for i in range(k):
        V[:, i] *= norms[i]
    
    # Compute R
    R = np.dot(U, V.T) + np.random.randn(m, n) * 0.01
    
    return R


def TestSVD(A, k, tau = None):
    #A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
    if tau is None:
        tau = 1 / np.linalg.norm(A, 2)
    I = A.copy()
    plt.imshow(np.uint8(A), cmap='gray')
    plt.show()

    tic = time.time()

    S, V, D = np.linalg.svd(I)
    
    D = S_tau(D, tau)

    toc = time.time()
    print("SVD Elapsed Time:", toc - tic)
    plt.imshow(np.uint8(I), cmap='gray')
    plt.show()

    k = 10
    reconstructed_image = np.dot(np.dot(S[:, :k], np.diag(V[:k])), D[:, :k].T)
    
    return reconstructed_image



def SVT(A, tau=None):
    import time
    
    if A is None:
        # Read an image if A is not provided
        import cv2
        A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    if tau is None:
        tau = 1 / np.linalg.norm(A, 2)

    if A.shape[0] < A.shape[1]:
        A = A.T

    m, n = A.shape
   
    tic = time.time()
    Y = A
    
    Q = QR_CP(Y)
    
    X, Q = Helper(A, tau, l, p, Q)

    #Q = PartialOrthogonalization(Q, Y)
    
    #X, Q = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("SVT Time:", toc - tic)
    X_real = np.real(X)
    return X_real

# List of ranks of matrices to test
ranks = [5, 8, 10]

# List of values for l and p to test
l_values = [5, 10, 15, 20]
p_values = [2,2]

# Loop over all ranks
for rank in ranks:
    # Generate a random matrix of given rank
    A = gen_factorization(2000, 2000, rank)

    # Loop over all combinations of l and p values
    for l in l_values:
        for p in p_values:
            # Run FRSVT function and compute RMSE
            result_FRSVT = FRSVT(A, p = p, l = l)
            rmse_FRSVT = sqrt(mean_squared_error(A, result_FRSVT))
            print(f"FRSVT RMSE for rank {rank} with l={l}, p={p}: {rmse_FRSVT}")

            # Run quantum_inspired_FRSVT function and compute RMSE
            result_qi_FRSVT = quantum_inspired_FRSVT(A, p = p, l = l)
            rmse_qi_FRSVT = sqrt(mean_squared_error(A, result_qi_FRSVT))
            print(f"QISVT RMSE for rank {rank} with l={l}, p={p}: {rmse_qi_FRSVT}")
            
            # Run FRSVT function and compute RMSE
            TestSVD_result = TestSVD(A, k = ranks * 2)
            rmse_SVD = sqrt(mean_squared_error(A, TestSVD_result))
            print(f"SVD RMSE for rank {rank} with l={l}, p={p}: {rmse_SVD}")
                        
            # Run FRSVT function and compute RMSE
            # SVT_result = SVT(A)
            # rmse_SVT = sqrt(mean_squared_error(A, SVT_result))
            # print(f"SVT RMSE for rank {rank} with l={l}, p={p}: {rmse_SVT}")