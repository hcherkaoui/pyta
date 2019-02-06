# coding: utf-8
""" Proximal operator functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np


def _soft_th(x, lbda, step_size):
    """ Prox for the temporal prox for one voxels.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lbda * step_size, 0)
