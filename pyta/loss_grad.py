# coding: utf-8
""" Loss and gradients functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np


def _grad_t(x, hth, hty=None, h=None):
    """ Gradient for the temporal prox for one voxels.
    """
    x = np.cumsum(x)
    grad = np.empty_like(x)
    # TODO debug this to reduce computation
    # for t in range(len(x) - len(hth) + 1):
    #     grad[t] = np.sum(hth * x[t: t + len(hth)])
    grad = np.convolve(h[::-1], np.convolve(h, x), mode='valid')
    if hty is not None:
        grad -= hty
    return np.cumsum(grad[::-1])[::-1]


def _obj_t(x, y, h, lbda):
    """ Cost func for the temporal prox for one voxels.
    """
    return (0.5 * np.sum(np.square(np.convolve(h, np.cumsum(x)) - y))
            + lbda * np.sum(np.abs(x)))
