""" This module define convolution related functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numba
import numpy as np
import torch
from torch.nn.functional import conv1d
from scipy.linalg import toeplitz


def make_toeplitz(v, n_times_valid):
    """ Make Toeplitz matrix from given kernel to perform valid
    convolution.

    Parameters
    ----------
    v : array, shape (n_times_atom), HRF
    n_times_valid : int, length of the temporal components

    Return
    ------
    H : array, shape (n_times, n_times_valid), Toeplitz matrix, recall that
        n_times = n_times_valid + n_times_atom -1
    """
    padd = np.zeros((1, n_times_valid - 1))
    return toeplitz(np.c_[v[None, :], padd], np.c_[1.0, padd])


def hu_numpy(h, u):
    """ Convolution of h with each line of u. Numpy implementation.

    Parameters
    ----------
    h : array, shape (n_time_hrf), HRF
    u : array, shape (n_samples, n_time_valid), neural activity signals

    Return
    ------
    h_conv_u : array, shape (n_samples, n_times), convolved signals
    """
    u = np.atleast_2d(u)
    n_samples, _ = u.shape
    return np.r_[[np.convolve(h, u[i]) for i in range(n_samples)]]


def hu_tensor(h, u):
    """ Convolution of h with each line of u. Torch implementation.

    Parameters
    ----------
    h : tensor, shape (n_time_hrf), HRF
    u : tensor, shape (n_samples, n_time_valid), neural activity signals

    Return
    ------
    h_conv_u : tensor, shape (n_samples, n_times), convolved signals
    """
    n_time_hrf = len(h)
    padd = n_time_hrf - 1
    n_samples, _ = u.shape
    h = torch.flip(h, (0,))
    h_conv_u = conv1d(u.view(n_samples, 1, -1), h.view(1, 1, -1), padding=padd)
    return h_conv_u.view(n_samples, -1)


def htx_numpy(h, x):
    """ Convolution of reversed h with each line of u. Numpy implementation.

    Parameters
    ----------
    h : array, shape (n_time_hrf), HRF
    x : array, shape (n_samples, n_time), neural activity signals

    Return
    ------
    h_conv_x : array, shape (n_samples, n_time_valid), convolved signals
    """
    n_samples, _ = x.shape
    return np.r_[[np.convolve(h[::-1], x[i], mode='valid')
                 for i in range(n_samples)]]


def htx_tensor(h, x):
    """ Convolution of reversed h with each line of u. Torch implementation.

    Parameters
    ----------
    h : tensor, shape (n_time_hrf), HRF
    x : tensor, shape (n_samples, n_time), neural activity signals

    Return
    ------
    h_conv_x : tensor, shape (n_samples, n_time_valid), convolved signals
    """
    padd = 0
    n_samples, _ = x.shape
    ht_conv_x = conv1d(x.view(n_samples, 1, -1), h.view(1, 1, -1),
                       padding=padd)
    return ht_conv_x.view(n_samples, -1)


@numba.jit((numba.float64[:], numba.float64[:, :], numba.int64, numba.int64),
           cache=True, nopython=True)
def _compute_hth_u(hth, u, n, m):
    """ Helper to HRF convolution and time-reversed HRF convolution operator.
    Numba implementation.

    Parameters
    ----------
    hth : array, shape (n_time_hrf), HRF
    u : array, shape (n_samples, n_time), neural activity signals
    n : int, n_samples
    m : int, n_time_valid

    Return
    ------
    hth_conv_u : array, shape (n_samples, n_time_valid), convolved signals
    """
    hth_conv_u = np.empty((n, m), dtype=np.float64)
    len_hth = len(hth)
    for i in range(n):
        for t in range(m):
            hth_conv_u[i, t] = np.sum(hth * u[i, t: t + len_hth])
    return hth_conv_u


def hthu_numpy(hth, u):
    """ HRF convolution and time-reversed HRF convolution operator. Numpy
    implementation.

    Parameters
    ----------
    hth : array, shape (n_time_hrf), HRF
    u : array, shape (n_samples, n_time), neural activity signals

    Return
    ------
    hth_conv_u : array, shape (n_samples, n_time_valid), convolved signals
    """
    n_samples, n_time_valid = u.shape
    padd = np.zeros((n_samples, int(len(hth)/2)), dtype=np.float64)
    u_padded = np.c_[padd, u, padd]
    return _compute_hth_u(hth, u_padded, n_samples, n_time_valid)


def hthu_tensor(hth, u):
    """ HRF convolution and time-reversed HRF convolution operator. Torch
    implementation.

    Parameters
    ----------
    hth : tensor, shape (n_time_hrf), HRF
    u : tensor, shape (n_samples, n_time), neural activity signals

    Return
    ------
    hth_conv_u : tensor, shape (n_samples, n_time_valid), convolved signals
    """
    n_kernel = len(hth)
    padd = int((n_kernel - 1) / 2)
    n_samples, _ = u.shape
    hth_conv_u = conv1d(u.view(n_samples, 1, -1), hth.view(1, 1, -1),
                        padding=padd)
    return hth_conv_u.view(n_samples, -1)


def hth_id_u_numpy(hth, u):
    """ HRF convolution and time-reversed HRF convolution operator plus
    identity operator. Numpy implementation.

    Parameters
    ----------
    hth : array, shape (n_time_hrf), HRF
    u : array, shape (n_samples, n_time), neural activity signals

    Return
    ------
    hth_conv_u : array, shape (n_samples, n_time_valid), convolved signals
    """
    n_samples, n_time_valid = u.shape
    padd = np.zeros((n_samples, int(len(hth)/2)), dtype=np.float64)
    u_padded = np.c_[padd, u, padd]
    # simple implementation to provide a reference
    return u - _compute_hth_u(hth, u_padded, n_samples, n_time_valid)


def hth_id_u_tensor(hth, u):
    """ HRF convolution and time-reversed HRF convolution operator plus
    identity operator. Torch implementation.

    Parameters
    ----------
    hth : tensor, shape (n_time_hrf), HRF
    u : tensor, shape (n_samples, n_time), neural activity signals

    Return
    ------
    hth_conv_u : tensor, shape (n_samples, n_time_valid), convolved signals
    """
    id = torch.zeros_like(hth)
    id[int(len(hth)/2)] = 1.0
    return hthu_tensor(id - hth, u)
