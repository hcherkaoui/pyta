""" This module loss and gradient functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from .convolution import hthu_numpy, hu_numpy


def _grad_t_analysis(u, hth, htY=None):
    """ Gradient for the temporal prox for one voxels. """
    u = np.atleast_2d(u)
    grad = hthu_numpy(hth, u)
    if htY is not None:
        grad -= htY
    return grad


def _obj_t_analysis(u, x, h, lbda):
    """ Cost func for the temporal prox for one voxels. """
    u = np.atleast_2d(u)
    n_samples = u.shape[0]
    res = (hu_numpy(h, u) - x).ravel()
    loss = 0.5 * res.dot(res) + lbda * np.sum(np.abs(np.diff(u)))
    return loss / n_samples
