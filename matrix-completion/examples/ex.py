import sys
sys.path.append('../')
from matrix_completion import *
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2
import matplotlib.pyplot as plt
# random seed
np.random.seed(10)



def plot_image(A, name):
    plt.imshow(A.T)
    plt.savefig(name)


if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--m", default=1000, type=int)
    argparse.add_argument("--n", default=1000, type=int)
    argparse.add_argument("--k", default=5, type=int)
    argparse.add_argument("--noise", default=0.01, type=float)
    argparse.add_argument("--mask-prob", default=0.5, type=float)

    args = argparse.parse_args()

    U, V, R = gen_factorization_with_noise(args.m, args.n, args.k, args.noise)
    mask = gen_mask(args.m, args.n, args.mask_prob)

    plot_image(R, "R.png")
    plot_image(mask, "mask.png")

    # print("== No Matrix Completion")
    # print("RMSE:", calc_unobserved_rmse(U, V, R * mask, mask))

    
    # R_hat = FRSVT_solve(
    #     R * mask, 10, 5, args.k, mask)
    # print("== quantum_inspired_svt_solve")
    # print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    # plot_image(R_hat, "quantum_inspired_svt_solve.png")

    # R_hat = pmf_solve_inspired(R * mask, 50, 50, args.k, mask, 1e-2)
    # print("== pmf_solve_inspired")
    # print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    # plot_image(R_hat, "pmf_solve.png")

    R_hat = pmf_solve(R * mask, mask, args.k, 1e-2)
    print("== PMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "pmf_solve.png")

    R_hat = biased_mf_solve(R * mask, mask, args.k, 1e-2)
    print("== BMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "biased_mf_solve.png")

    
    R_hat = svt_solve(R, mask)
    print("== SVT")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "SVT_R_hat.png")
    
    R_hat = svt_solve(R, mask, algorithm = "inspired")
    print("== Inspired_SVT")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "SVT_R_hat.png")
    
    R_hat = svt_solve(R, mask, algorithm = "randomized")
    print("== Randomized_SVT")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "SVT_R_hat.png")
    
    
    
    A = cv2.imread('../../1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius =  ls_probs(A.shape[0], A.shape[1], A)
    X =  quantum_inspired_FRSVT(A, row_norms=row_norms, LS_prob_rows=LS_prob_rows, LS_prob_columns=LS_prob_columns, A_Frobenius=A_Frobenius)
    # save image
    import matplotlib.pyplot as plt
    plt.imshow(np.uint8(X), cmap='gray')
    # save
    plt.savefig('Inspired_Q.jpg')       
