import numpy as np
import logging

def calculate_prob(U):
    row_sums = np.linalg.norm(U, axis=1)**2
    return row_sums / np.sum(row_sums)

def quantum_inspired_pmf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve using Quantum-Inspired Probabilistic Matrix Factorization.

    Parameters and returns are same as the pmf_solve function.
    """
    logger = logging.getLogger(__name__)
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in range(max_iterations):

        # quantum-inspired sampling
        p_U = calculate_prob(U)
        p_V = calculate_prob(V)
        U_sampled = np.random.choice(U, p=p_U)
        V_sampled = np.random.choice(V, p=p_V)

        for i in U_sampled:
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in V_sampled:
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X

    return X
