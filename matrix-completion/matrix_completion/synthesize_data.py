import numpy as np
from scipy.stats import bernoulli


def gen_mask(m, n, prob_masked=0.5):
    """
    Generate a binary mask for m users and n movies.
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))


def gen_factorization_without_noise(m, n, k):
    """
    Generate non-noisy data for m users and n movies with k latent factors.
    Draws factors U, V from Gaussian noise and returns U Vᵀ.
    """
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    R = np.dot(U, V.T)
    return U, V, R


def gen_factorization_with_noise(m, n, k, sigma):
    
    # Create an array of norms in a random increasing way
    norms = np.cumsum(np.abs(np.random.randn(k))) * 10
    
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
    R = np.dot(U, V.T) + np.random.randn(m, n) * sigma
    return U, V, R
