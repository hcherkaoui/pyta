# coding: utf-8
""" Loss and gradients functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np
import numba


@numba.jit((numba.float64[:], numba.float64[:], numba.int64), cache=True,
           nopython=True)
def hth_(hth, x, n):
    """
    """
    hth_x = np.empty(n)
    for t in range(n):
        hth_x[t] = np.sum(hth * x[t: t + len(hth)])
    return hth_x


def _grad_t(x, hth, hty=None):
    """ Gradient for the temporal prox for one voxels.
    """
    Lx = np.r_[np.zeros(int(len(hth)/2)),
               np.cumsum(x),
               np.zeros(int(len(hth)/2))]
    grad = hth_(hth, Lx, len(x))

    if hty is not None:
        grad -= hty

    return np.cumsum(grad[::-1])[::-1]


def _obj_t(x, y, h, lbda):
    """ Cost func for the temporal prox for one voxels.
    """
    return (0.5 * np.sum(np.square(np.convolve(h, np.cumsum(x)) - y))
            + lbda * np.sum(np.abs(x)))
