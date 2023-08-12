import numpy as np
from numpy.linalg import norm
import time

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

            # reshape temp to (m, 1)
            temp = np.reshape(temp, (m, 1))
            
            bmax2 += temp
            #print(bmax2.shape)
            # bmax2 += np.dot(Q[:, i].T, B[:, j]) * Q[:, i]
            #for i in range(j):
            #bmax2 += np.outer(Q[:, i], np.squeeze(B[:, j]))
        
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
        S_D[i, i] = np.sign(D[i, i]) * max([abs(D[i, i] - tau), 0])
    return S_D

def Helper(A, tau, l, p, Q):
    for i in range(2):
        Q, R = np.linalg.qr(np.dot(A, np.dot(A.T, Q)))
        Q = Q[:, :R.shape[1]]
        R = R[:R.shape[1], :]

    H, C = np.linalg.qr(np.dot(A.T, Q))
    H = H[:, :C.shape[1]]
    C = C[:C.shape[1], :]
    W, P = Polar(C)
    D, V = np.linalg.eig(P)
    D = np.diag(D)
    X = np.dot(np.dot(Q, V), S_tau(D, tau))
    X = np.dot(X, np.dot(np.dot(H, W), V).T)
    Q = np.dot(Q, V)
    return X, Q

def PartialOrthogonalization(Q, Y):
    A = np.hstack((Q, Y))
    m, n = A.shape
    for i in range(Q.shape[1], n):
        while np.linalg.matrix_rank(A[:, :i]) != min(A[:, :i].shape):
            A[:, i] = np.random.rand(m, 1)
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
    Omega = np.random.randn(n, l)
    Y = np.dot(A, Omega)
    print(Y.shape)
    Q = QR_CP(Y)
    X, Q = Helper(A, tau, l, p, Q)

    Omega = np.random.randn(n, p)
    Y = np.dot(A, Omega)
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("Elapsed Time:", toc - tic)

    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X), cmap='gray')
    # save
    plt.savefig('Q.jpg')
    

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

A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

FRSVT(A)


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

    tic = time.time()

    m, n = A.shape
    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, l, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    print(Y.shape)
    Q = QR_CP(Y)
    X, Q = Helper(A, tau, l, p, Q)

    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, p, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("Elapsed Time:", toc - tic)

    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X), cmap='gray')
    # save
    plt.savefig('Inspired_Q.jpg')
    
quantum_inspired_FRSVT(A)