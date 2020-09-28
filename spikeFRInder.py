"""
Functions for using frequency-domain FRI method for spike inference.

Authors: Benjamin Bejar, Gavin Mischler
"""

import numpy as np
from numpy.fft import fft, fftshift, ifft
from math import pi
from scipy.linalg import toeplitz, svd, pinv, inv
from scipy.optimize import golden
from scipy.signal import hamming
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from utils import F, Cadzow


# ============================================================================
# Calcium signal sliding window estimation
# ============================================================================
def sliding_window_predict(signal,
                           Fs,
                           K_full_signal_estimate,
                           alpha_grid_ends=[7, 1.75],
                           window_lengths=[301, 601, 801, 1101],
                           jump_size=25,
                           OF=4,
                           smoothing_sigma=5):
    """
    Use multiple sliding windows to estimate spikes. Returns dictionary
    containing aggregated array of spike likelihoods, same length as
    input signal.

    Parameters
    ----------
    signal : numpy 1-D array
        Calcium signal (preprocessed).
    Fs : int
        Sampling frequency of the signal
    K_full_signal_estimate : int
        Estimate of the number of spikes in the entire signal
    alpha_grid_ends : list (length=2), default=[7, 1.75]
        Endpoints of the grid within which to search for the time constant,
        alpha, for decaying exponentials of the form exp(-t*alpha).
    window_lengths : list, default=[301, 601, 801, 1101]
        The window lengths to use. Each length is used with a sliding
        window approach and the results are summed together.
    jump_size : int, default=25
        Jump size of the sliding window approach (number of samples)
    OF : int, default=4
        Oversampling factor, used to truncate the Fourier series.
    smoothing_sigma: float, default=5
        Sigma parameter for final smoothing of the joint histogram using
        scipy.ndimage.gaussian_filter1d. If `None`, no smoothing is performed
        and the joint histogram is returned.

    Returns
    -------
    final_histogram : numpy 1-D array with same shape as signal
        Joint histogram spiking estimate, same shape as input signal.

    """

    likelihood_counts = np.zeros_like(signal)

    for window_len in window_lengths:
        # estimate K for this window
        K = round(window_len / len(signal) * K_full_signal_estimate)

        N = len(signal)
        start_pt = 0
        end_pt = window_len

        while end_pt < N:
            sig = signal[start_pt:np.clip(end_pt, 0, N-1)]

            d = estimate_tk_ak(signal=sig, Fs=Fs, K=K, alpha_grid_ends=alpha_grid_ends, OF=OF)

            tk_indices = d['tk_indices']
            ak_hat = d['ak_hat']

            relative_indices = tk_indices + start_pt

            # add to likelihood counts
            likelihood_counts[relative_indices] += np.sqrt(np.maximum(ak_hat, 0))

            start_pt = start_pt + jump_size
            end_pt = end_pt + jump_size


    if smoothing_sigma is not None:
        return gaussian_filter1d(likelihood_counts, sigma=smoothing_sigma)

    return likelihood_counts


def estimate_tk_ak(signal, Fs, K, alpha_grid_ends, OF=4):
    """
    Estimate the indices of most likely spikes in the signal using the FRI method.
    
    Parameters
    ----------
    signal : signal to process, shape=(N,)
        Signal to estimate K spikes within.
    Fs : float
        Sampling frequency of signal
    K : int
        Number of spikes assumed to be in the signal
    alpha_grid_ends : list (length=2), default=[7, 1.75]
        Endpoints of the grid within which to search for the time constant,
        alpha, for decaying exponentials of the form exp(-t*alpha).
    OF : int, positive
        oversampling factor to define the number of Fourier coefficients
        to use from the signal, by L = (OF * K * 2) + 1
    
    Returns
    -------
    d : Dictionary of outputs containing the following
        'tk_indices' : array of indices corresponding to the predicted spike locations, shape=(K,)
        'ak_hat'    : estimated amplitudes of detected spikes, shape=(K,)
    """
    
    # scale the alpha estimate to the length of the signal, since the model
    # assumes the signal is 1 second long
    N = len(signal)
    N_full = N
    seconds = N / Fs # duration of signal in seconds
    alpha_grid_ends = [alpha * seconds for alpha in alpha_grid_ends]

    # compute Fourier series
    Zk_tilde = fft(signal)
    
    if OF is None:
        L = (N - 1) / 2
    else:
        L = OF * K * 2 + 1
        N = 2 * L + 1
        Zk_tilde = np.concatenate((Zk_tilde[:L+1], Zk_tilde[-L-1:-1]))

    # grid search for candidate solutions
    alpha_grid = np.linspace(alpha_grid_ends[0], alpha_grid_ends[1], 6)
    
    # frequencies used for estimation
    wk = 2*pi*np.arange(-L,L+1)

    # error variable
    e = np.zeros(alpha_grid.size)

    for ii, alpha in enumerate(alpha_grid):

        # estimated Fourier coefficients
        Sk = fftshift(Zk_tilde) * (alpha + 1j*wk)

        Sk = Cadzow(Sk,K,N)

        # find annihilating filter
        s = svd( toeplitz(Sk[K:],Sk[np.arange(K,-1,-1)]) )[1]

        # error computation
        e[ii] = s[-1]

    # minindex = 0
    minindx = np.argmin(e)
    if minindx == 0 or minindx == len(e)-1:
        bracket = (alpha_grid[0],alpha_grid[0])
    else:
        bracket = tuple(alpha_grid[minindx-1:minindx+2])

    # refine estimate with golden search, using Cadzow
    alpha_hat = golden(F,(Zk_tilde,wk,K,True),bracket)
    alpha_hat = np.clip(alpha_hat,a_min=alpha_grid[0],a_max=alpha_grid[-1])

    # estimated Fourier coefficients
    Sk = fftshift(Zk_tilde) * (alpha_hat + 1j*wk)

    Sk = Cadzow(Sk,K,N)

    ## Prony's method to estimate tk

    # find annihilating filter
    Qh = svd( toeplitz(Sk[K:],Sk[np.arange(K,-1,-1)]) )[2]
    h  = Qh[-1,:].conj()
    h  = h/h[0]

    # estimate time locations from the roots of the polynomial
    tk_hat = np.sort( np.mod( np.angle( np.roots( h[::-1] ) ) / 2.0 / pi, 1 ) ).reshape((K,1))
    
    # estimate amplitudes by solving linear system
    uk = np.exp( -1j * 2 * np.pi * tk_hat ).reshape((K,))

    V = np.flipud(np.vander(uk, len(Sk)//2+1).transpose())[1:,:]

    if len(Sk)%2==0:
        Z  = np.concatenate((np.flipud(V.conj()),np.ones((1,K)),V[:-1,:]),axis=0) 
    else:
        Z  = np.concatenate((np.flipud(V.conj()),np.ones((1,K)),V),axis=0)

    # least-squares estimate
    ak_hat = np.real(np.dot(pinv(Z),Sk)).reshape((K,))    
    
    tk_hat_indices = (tk_hat * N_full).astype(int).squeeze()
    
    d = {'tk_indices': tk_hat_indices,
         'ak_hat' : ak_hat}
    
    return d
