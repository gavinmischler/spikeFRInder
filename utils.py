"""
Utility functions to aid the frequency domain FRI method of estimating
a stream of decaying exponentials.

Authors: Benjamin Bejar, Gavin Mischler
"""

import numpy as np
from numpy.fft import fftshift
from scipy.linalg import toeplitz, svd


def F(alpha,Zk,wk,K,cadzow=False):
    """
    Cost function for finding the optimal decay factor.
    """ 
    # estimated Fourier coefficients
    Sk = fftshift(Zk) * (alpha + 1j*wk)
    
    if cadzow:
        N = len(Sk)
        Sk = Cadzow(Sk, K=K, N=N)
    
    # find annihilating filter
    s = svd( toeplitz(Sk[K:],Sk[np.arange(K,-1,-1)]) )[1]
    
    # return smallest singular value
    return s[-1]

# def Prony(Yk, K, cadzow=False):
#     """
#     Estimation of locations using Prony's method.
#     """
#     Y = Yk.copy()
#     if cadzow:
#         N = len(Yk)
#         Y = Cadzow(Y, K=K, N=N)

#     # find annihilating filter
#     Qh = svd( toeplitz(Y[K:],Y[np.arange(K,-1,-1)]) )[2]
#     h  = Qh[-1,:].conj()
#     h  = h/h[0]
    
#     # estimate time locations from the roots of the polynomial
#     tk_hat = np.sort( np.mod( np.angle( np.roots( h[::-1] ) ) / 2.0 / pi, 1 ) ).reshape((K,1))
        
#     # return estimates
#     return tk_hat

def Cadzow(Xk, K, N, tol_ratio=10000, max_iter=10):
    """
    Implement Cadzow denoising

    Parameters
    ----------
    Xk : signal to denoise
    K : number of most significant members to take
    N : number of samples in the signal
    tol_ratio : min ratio of (K+1)th singular value / Kth singular value
                to stop iterations
    max_iter : maximum number of iterations to run
    
    Returns
    -------
    X : denoised signal
    """
    
    X = Xk.copy()
    ratio = 0
    iters = 0
    
    while (ratio < tol_ratio and iters < max_iter):
        iters += 1

        # perform svd
        #print(toeplitz(X[K:],X[np.arange(K,-1,-1)]).shape)
        U, s, Vh = svd(toeplitz(X[K:],X[np.arange(K,-1,-1)]))

        # update ratio of singular values for cutoff
        ratio = s[K-1] / s[K]

        # build S' : first K diagonals of S
        s_ = s[:K]
        

        sz1 = U.shape[1]
        sz2 = Vh.shape[0]
        S_ = np.zeros(shape=(sz1, sz2))
        for elem,(i,j) in enumerate(zip(np.arange(K),np.arange(K))):
            S_[i,j] = s_[elem]

        # least squares approx. for A
        A_ = U @ S_ @ Vh

        # denoised Xk is the average of the diagonals
        for idx, off in enumerate(np.arange(K,K-N,-1)):
            temp = np.mean(np.diagonal(A_,offset=off))
            X[idx] = temp
    
    return X
    