""" This module loss and gradient functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def _grad_t_analysis(x, HtH, Hty=None):
    """ Gradient for the temporal prox for one voxels. """
    x = np.atleast_2d(x)
    grad = x.dot(HtH)
    if Hty is not None:
        grad -= Hty
    return grad


def _obj_t_analysis(u, y, H, lbda):
    """ Cost func for the temporal prox for one voxels. """
    u = np.atleast_2d(u)
    n_samples = u.shape[0]
    res = (u.dot(H) - y).ravel()
    reg = np.sum(np.abs(np.diff(u, axis=1)))
    return (0.5 * res.dot(res) + lbda * reg) / n_samples
