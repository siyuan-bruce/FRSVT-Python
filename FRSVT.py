import numpy as np
from numpy.linalg import norm
import time

def Polar(C):
    print(C.shape)
    U, S, V = np.linalg.svd(C)
    W = np.dot(U, V.T)
    print(W.shape)
    print(V.shape)
    print(S.shape)
    # transform S into a diagonal matrix
    S = np.diag(S)
    P = np.dot(np.dot(V, S), V.T)
    print("P Shape", P.shape)
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
        
        print(bmax2.shape)

        for i in range(j):
            
            temp = np.dot(Q[:, i].T, B[:, j]) * Q[:, i]

            # reshape temp to (m, 1)
            temp = np.reshape(temp, (m, 1))
            
            bmax2 += temp
            #print(bmax2.shape)
            # bmax2 += np.dot(Q[:, i].T, B[:, j]) * Q[:, i]
            #for i in range(j):
            #bmax2 += np.outer(Q[:, i], np.squeeze(B[:, j]))
        
        print(B.shape)
        
        print(bmax2.shape)
        
        print(B[:, j].shape)
        
        bmax2 = np.reshape(B[:, j], (m,1)) - bmax2
        
        print(bmax2.shape)

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
    print("Qshape:", Q.shape)
    print("Vshape", V.shape)
    print("Dshape", D.shape)
    print(D)
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
    print(A.shape)
    Omega = np.random.randn(n, l)
    print(Omega.shape)
    Y = np.dot(A, Omega)
    print(Y.shape)
    Q = QR_CP(Y)
    print(Q.shape)
    X, Q = Helper(A, tau, l, p, Q)

    Omega = np.random.randn(n, p)
    Y = np.dot(A, Omega)
    Q = PartialOrthogonalization(Q, Y)
    X, Q = Helper(A, tau, l, p, Q)

    toc = time.time()
    print("Elapsed Time:", toc - tic)
    

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
    plt.imshow(np.uint8(I), cmap='gray')
    plt.show()

    k = 10
    reconstructed_image = np.dot(np.dot(S[:, :k], np.diag(V[:k])), D[:, :k].T)
    plt.imshow(np.uint8(reconstructed_image), cmap='gray')
    plt.show()

    toc = time.time()
    print("Elapsed Time:", toc - tic)

# Call the test function
TestSVD()

A = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

FRSVT(A)
