from __future__ import division
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

import numpy as np
from numpy.linalg import norm
import time

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
    S_D = np.dot(np.dot(Q, V), S_tau(D, tau))
    X = np.dot(S_D, np.dot(np.dot(H, W), V).T)
    Q = np.dot(Q, V)
    return X, Q, S_D

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
    Omega = np.random.randn(n, l)
    Y = np.dot(A, Omega)
    print(Y.shape)
    Q = QR_CP(Y)
    X, Q, D = Helper(A, tau, l, p, Q)

    Omega = np.random.randn(n, p)
    Y = np.dot(A, Omega)
    Q = PartialOrthogonalization(Q, Y)
    X, Q, D = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("Elapsed Time:", toc - tic)

    return X
    


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


def quantum_inspired_FRSVT(A, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius, tau=None, l=None, p=None):
    
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
    tic = time.time()

    m, n = A.shape
    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, l, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    
    print(Y.shape)
    
    Q = QR_CP(Y)

    X, Q, D = Helper(A, tau, l, p, Q)

    Y, rows, columns = sample_C(A, A.shape[0], A.shape[1], m, p, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    
    Q = PartialOrthogonalization(Q, Y)

    X, Q, D = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("Elapsed Time:", toc - tic)

    return X, D

    
# A = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

# FRSVT(A, l = 50, p = 20)

# quantum_inspired_FRSVT(A, l = 50, p = 20)


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
    print("vl_vector")
    print(v_approx)
    for s in range(r):
        print(s)
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
    #print(sigma)
    u_approx = (A @ v_approx) / sigma[l]
    
    #print(v_approx)

    return u_approx, v_approx


def _my_svd(M, k, algorithm):
    if algorithm == 'randomized':
        (U, S, V) = randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=20)
    elif algorithm == 'arpack':
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("unknown algorithm")
    return (U, S, V)



def FRSVT_solve(A, r, c, rank, mask, delta = None, tau=None, max_iterations=1000, epsilon=1e-5):

    r""" Function to solve the the linear system of equations :math:'A \bm{x} = b' using FKV algorithm
    and a direct calculation of the coefficients :math: '\lambda_l' and solution vector :math: '\bm{x}'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A

    Returns:
        array[float]: array containing the components of the solution vector :math: '\bm{x}'
    """
    logger = logging.getLogger(__name__)
    Y = mask * A
    
    m_rows, n_cols = np.shape(A)
    
    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * r * c / np.sum(mask)

    # 1- Generating LS probability distributions used to sample rows and columns indices of matrix A
    tic = time.time()

    # save reconstruction error for drawing
    rec_errors = []
    rec_errors.append(1)
    
    X_output = np.zeros_like(A)

    for k in range(100):
        
        toc = time.time()
        
        rt_ls_prob = toc - tic
        
        if k == 0:
            X = np.zeros_like(A)
        else:
            X =  FRSVT(Y,l = c, p = r)
            
        Y += delta * mask * (A - X)
        
        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        print(recon_error)        
        if recon_error < rec_errors[-1]:
            print(recon_error)
            rec_errors.append(recon_error)
            X_output = X

        if k % 10 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
    
        if recon_error < epsilon:
            break
        
    # draw reconstruction error with iterations
    plt.figure()
    print(rec_errors)
    plt.plot(rec_errors)
    plt.xlabel('Iterations')
    plt.ylabel('Reconstruction error')
    plt.savefig('reconstruction_error.png')

    return X_output


import numpy as np
from scipy.sparse import spdiags

def rSVDBKIr(A, k, i, Q=None):
    s = 5
    if Q is None:
        m, n = A.shape
        B = np.random.randn(n, k+s)
        H = np.zeros((m, (k+s)*i))
        H[:, :k+s], _ = np.linalg.lu(A @ B)
        for j in range(2, i+1):
            H[:, (k+s)*(j-1):(k+s)*j], _ = np.linalg.lu(A @ (A.T @ H[:, (k+s)*(j-2):(k+s)*(j-1)]))
        Q, _ = np.linalg.qr(H, mode='reduced')
        kn = i*(k+s)
    else:
        kn = Q.shape[1]

    T = A.T @ Q
    v, d = np.linalg.eig(T.T @ T)
    ss = np.sqrt(np.diag(d))
    S = spdiags(ss, 0, kn, kn)
    u = np.linalg.solve(S, T @ v.T).T
    V = u
    x = slice(kn-k, kn)
    S = ss[x]
    U = Q @ v[:, x]
    V = V[:, x]

    return U, S, V, Q


import numpy as np
from scipy.sparse import spdiags, coo_matrix
from scipy.sparse.linalg import svds

def fastSVT_Q(M, tol, ran, i_reuse, q_reuse, delta):
    m, n = M.shape
    Omega = M != 0
    Ns = np.sum(Omega)
    
    if delta is None:
        delta = 1.2 * m * n / Ns
    
    xi, yi = np.where(Omega)
    
    tau = 5 * n
    l = 5
    i_max = 1000
    PM = M.copy()
    normPM2 = svds(PM, k=1)[0][0]
    normPM = np.linalg.norm(PM, 'fro')
    k0 = np.ceil(tau / (delta * normPM2))
    Y0 = k0 * delta * PM

    dec = 0
    r = 0
    p = 2
    q = 0
    err_before = 1000

    for i in range(1, i_max + 1):
        r_before = r
        r += 1
        
        if i % 50 == 0:
            delta /= 1.1
        
        if i > i_reuse and q < q_reuse:
            U, S, Vt = rSVDBKIr(Y0, r, p, Q)
            q += 1
        else:
            U, S, Vt = rSVDBKIr(Y0, r, p)
            q = 0
        
        while S[0] > tau:
            r += l
            U, S, Vt = rSVDBKIr(Y0, r, p)
        
        j = 0
        while S[j] <= tau:
            j += 1
        
        r_max = r
        r = max(r_max - j + 1, r_before)
        x = slice(r_max - r, r_max)
        S[x] = S[x] - tau
        U[:, x] *= S[x]
        
        x_now = np.zeros(Ns)
        for j in range(Ns):
            temp = np.dot(U[xi[j], x], Vt[yi[j], x].T)
            if temp < ran[0]:
                x_now[j] = ran[0]
            elif temp > ran[1]:
                x_now[j] = ran[1]
            else:
                x_now[j] = temp
        
        X = coo_matrix((x_now, (xi, yi)), shape=(m, n)).tocsr()
        PX = X - PM
        err = np.linalg.norm(PX, 'fro') / normPM
        
        if err <= tol:
            X = np.dot(U[:, x], Vt[:, x].T)
            X[X < ran[0]] = ran[0]
            X[X > ran[1]] = ran[1]
            k = r
            iters = i
            break
        
        if err > err_before:
            dec = 0
            p += 1
            q = 10
        else:
            if p <= 5:
                dec = 0
            else:
                dec += 1
                if dec == 10:
                    p -= 1
                    dec = 0
        
        err_before = err
        print([i, r, err, p])
        Y0 -= delta * PX
    
    return X, iters, k


def svt_solve_inspired(A, r, c, rank, mask, delta = None, tau=None, max_iterations=1000, epsilon=1e-5):

    
    r""" Functiontion to solve the the linear system of equations :math:'A \bm{x} = b' using FKV algorithm
    and a direct calculation of the coefficients :math: '\lambda_l' and solution vector :math: '\bm{x}'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A

    Returns:
        array[float]: array containing the components of the solution vector :math: '\bm{x}'
    """
    
    import time 
    logger = logging.getLogger(__name__)
    Y = mask * A
    
    m_rows, n_cols = np.shape(A)
    
    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * r * c / np.sum(mask)

    # 1- Generating LS probability distributions used to sample rows and columns indices of matrix A
    tic = time.time()

    row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius =  ls_probs(A.shape[0], A.shape[1], Y)

    
    # save reconstruction error for drawing
    rec_errors = []
    rec_errors.append(1)
    
    X_output = np.zeros_like(A)

    import time
    
    if r is None:
        r = 20
    if c is None:
        c = 10
        
        
    r_previous = r
    for k in range(100):
        
        toc = time.time()
        
        rt_ls_prob = toc - tic
        
        if k == 0:
            X = np.zeros_like(A)
        else:

            X, D = quantum_inspired_FRSVT(Y, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius, tau=tau, l=r, p=c)
            
        Y += delta * mask * (A - X)
        
        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        
        print(recon_error)        
        if recon_error < rec_errors[-1]:
            print(recon_error)
            rec_errors.append(recon_error)
            X_output = X

        if k % 10 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
    
        if recon_error < epsilon:
            break
        
    # draw reconstruction error with iterations
    plt.figure()
    print(rec_errors)
    plt.plot(rec_errors)
    plt.xlabel('Iterations')
    plt.ylabel('Reconstruction error')
    plt.savefig('reconstruction_error.png')

    return X_output
        

